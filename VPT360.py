import torch
import torch.nn as nn

import torch.nn.init as init
import torch.nn.functional as F
from torchviz import make_dot
import torch.optim as optim
from Utils import MetricOrthLoss, get_orthodromic_distance_cartesian
import sys

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
    
class VPT360(nn.Module):
    def __init__(self,M_WINDOW, H_WINDOW,input_dim=2,hidden_size=512, num_heads=8,num_layers=1,max_len=1000,dropout=0.1):
        super().__init__()
        self.H_WINDOW=H_WINDOW
        self.M_WINDOW=M_WINDOW
        self.input_dim=input_dim
        self.hidden_size=hidden_size
        self.num_heads=num_heads
        self.num_layers=num_layers
        self.max_len=max_len
        self.input_embedding=nn.Linear(self.input_dim,self.hidden_size)
        self.pos_encoder=PositionalEncoding(hidden_size,max_len=self.max_len)
        encoder_layer=nn.TransformerEncoderLayer(d_model=hidden_size,nhead=num_heads,
                                                 dim_feedforward=4*hidden_size,dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layer=nn.Linear(hidden_size,H_WINDOW*input_dim)
        
    def forward(self,ip):
        x=ip[0]
        batch_size,seq_len,_=x.shape
        x=self.input_embedding(x)
        x=self.pos_encoder(x)

        output=self.transformer_encoder(x)
        
        output=self.output_layer(output[:,-1,:])
        output=output.view(batch_size,self.H_WINDOW,self.input_dim)
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
        
def create_VPT360_model(M_WINDOW,H_WINDOW,input_size_pos=3, lr=2e-7,hidden_size=512,num_heads=8,num_layers=1,max_len=1000,dropout=0.1, device='cpu'):
    model=VPT360(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,input_dim=input_size_pos,hidden_size=hidden_size,
                 num_heads=num_heads,num_layers=num_layers,max_len=max_len,
                 dropout=dropout).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=lr, weight_decay=0.99)
    criterion=CombinationLoss()
    return model, optimizer, criterion

if __name__=="__main__":
    d_model=128
    past_vp=torch.randn(1,32,5,2)
    model=VPT360(M_WINDOW=5,H_WINDOW=25,hidden_size=d_model,input_dim=2, num_layers=1,max_len=200)
    pred=model(past_vp)
    #dot=make_dot(pred,params=dict(model.named_parameters()))
    #dot.format = 'png'  # Optional: set the format to PNG
    #dot.render("network_visualization") 
    
    #print(pred.shape)
    