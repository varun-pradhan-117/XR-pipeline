import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.optim as optim
from Utils import MetricOrthLoss, get_orthodromic_distance_cartesian

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=1000):
        super().__init__()
        pe=torch.zeros(max_len,d_model)
        position=torch.arange(0,max_len,dtype=torch.float).unsqueeze(1)
        
        div_term=torch.pow(10000,-torch.arange(0,d_model,2).float()/d_model)
        pe[:,0::2]=torch.sin(position*div_term)
        pe[:,1::2]=torch.cos(position*div_term)
        pe=pe.unsqueeze(0).transpose(0,1)
        self.register_buffer('pe',pe)
        
    def forward(self,x):
        x=x+self.pe[:x.size(0),:]
        return x

class AdaptiveMultiHeadAttention(nn.Module):
    def __init__(self,d_model,num_heads, dropout, M_WINDOW=None):
        super().__init__()
        self.num_heads=num_heads
        self.d_model=d_model
        assert d_model%num_heads==0, 'Embedding dimension must be divisible by the number of heads'
        self.d_k=d_model//num_heads
        self.scale=self.d_k**0.5
        self.W_q=nn.Linear(d_model,d_model)
        self.W_k=nn.Linear(d_model,d_model)
        self.W_v=nn.Linear(d_model,d_model)
        self.W_o=nn.Linear(d_model,d_model)
        if M_WINDOW:
            self.W_e=nn.Parameter(torch.empty(M_WINDOW).normal_(mean=1,std=0.01))
            
        else:
            self.W_e=nn.Parameter(torch.randn(1))
        
        self.attention_dropout=nn.Dropout(dropout)
        self.output_dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(num_heads * self.d_k, d_model)
        self.layer_norm=nn.LayerNorm(d_model)
        
    def forward(self,Q,K,V,entropy):
        batch_size,seq_len,embed_dim=Q.size()
        res=Q
        # Separate different heads b x seq_len x h x dk and reshape to b x h x seq_len x dk
        Q=self.W_q(Q).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        K=self.W_k(K).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        V=self.W_v(V).view(batch_size,seq_len,self.num_heads,self.d_k).transpose(1,2)
        #print(K.shape)
        #print(Q.shape)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        entropy_weights=torch.exp(self.W_e.unsqueeze(0)*entropy.squeeze(2)).unsqueeze(1).unsqueeze(1)
        #print(entropy_weights.shape)
        adjusted_scores = attn_scores * entropy_weights
        adjusted_scores=self.attention_dropout(adjusted_scores)
        #print(adjusted_scores.shape)
        attention_weights=F.softmax(adjusted_scores,dim=-1)
        attention_weights=self.attention_dropout(attention_weights)
        output=torch.matmul(attention_weights,V)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.output_dropout(self.fc(output))
        output+=res
        output=self.layer_norm(output)
        return output
        
class PositionWiseFeedForward(nn.Module):
    def __init__(self,d_in,d_hid,dropout=0.1):
        super().__init__()
        self.w_1=nn.Linear(d_in,d_hid)
        self.w_2=nn.Linear(d_hid,d_in)
        self.layer_norm=nn.LayerNorm(d_in,eps=1e-6)
        self.dropout=nn.Dropout(dropout)
        
    def forward(self,x):
        res=x
        x=self.w_2(F.relu(self.w_1(x)))
        x=self.dropout(x)
        x+=res
        x=self.layer_norm(x)
        return x
        
class AMH(nn.Module):
    def __init__(self, M_WINDOW, H_WINDOW, input_dim=2, hidden_size=512, num_heads=8, num_layers=1, max_len=1000, dropout=0.1, full_vec=False):
        super().__init__()
        self.H_WINDOW = H_WINDOW
        self.M_WINDOW = M_WINDOW
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_len = max_len
        self.input_embedding = nn.Linear(self.input_dim, self.hidden_size)
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=self.max_len)
        
        if full_vec:
            # Use multiple AdaptiveAttention layers with dropout for each layer
            self.encoder_layers = nn.ModuleList([AdaptiveMultiHeadAttention(hidden_size, num_heads, dropout=dropout, M_WINDOW=M_WINDOW) for _ in range(num_layers)])
        else:
            self.encoder_layers = nn.ModuleList([AdaptiveMultiHeadAttention(hidden_size, num_heads, dropout=dropout) for _ in range(num_layers)])
        
        self.pos_ffn=PositionWiseFeedForward(d_in=hidden_size,d_hid=hidden_size*4,dropout=0.1)
        
        self.output_layer = nn.Linear(hidden_size, H_WINDOW * input_dim)
        
        # Dropout layer for the input embeddings and final layer
        self.dropout = nn.Dropout(dropout)
        self.layer_norm=nn.LayerNorm(self.hidden_size,eps=1e-6)
        
    def forward(self, ip):
        x = ip[0]
        entropy=ip[1]
        batch_size, seq_len, _ = x.shape
        x = self.input_embedding(x)
        x = self.pos_encoder(x)
        # Apply dropout to embeddings
        x = self.dropout(x)
        
        x=self.layer_norm(x)
        
        # Pass through adaptive attention layers
        for layer in self.encoder_layers:
            x = layer(x, x, x, entropy)  # Pass entropy to each layer

        x=self.pos_ffn(x)
        
        output = self.output_layer(x[:, -1, :])
        
        output = output.view(batch_size, self.H_WINDOW, self.input_dim)
        return output
    
class CombinationLoss(nn.Module):
    def __init__(self,alpha=0.75, beta=0.25):
        super().__init__()
        self.alpha=alpha
        self.beta=beta
    
    def forward(self,pred_pos,true_pos,pred_vel,true_vel):
        #print(pred_pos.shape)
        #print(true_pos.shape)
        mse_pos=F.mse_loss(pred_pos,true_pos)
        #print(mse_pos)
        pos_loss=MetricOrthLoss(pred_pos,true_pos)
        #print(pos_loss)
        #print(mse_pos)
        #print(mse_pos.shape)
        #print(pos_loss.shape)
        mse_vel = F.mse_loss(pred_vel, true_vel)
        #print(mse_vel)
        #print(mse_vel.shape)
        #sys.exit()
        return self.alpha*mse_pos + self.beta*mse_vel
    
def create_AMH_model(M_WINDOW,H_WINDOW,input_size_pos=3, lr=0.0001,hidden_size=512,num_heads=8,num_layers=1,max_len=1000,
                     full_vec=False,dropout=0.1, device='cpu', mode=None):
    model=AMH(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,input_dim=input_size_pos,hidden_size=hidden_size,
                 num_heads=num_heads,num_layers=num_layers,max_len=max_len, full_vec=full_vec,
                 dropout=dropout).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=lr, weight_decay=0.99)
    criterion=CombinationLoss()
    return model, optimizer, criterion