import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torchinfo import summary
import os

def toPosition(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return (orientation + motion)


class AttentionLayer(nn.Module):
    def __init__(self,hidden_size,M_WINDOW, entropy=True, mode='IE'):
        super().__init__()
        self.mode=mode
        self.W_K=nn.Linear(hidden_size,hidden_size)
        self.W_Q=nn.Linear(hidden_size,hidden_size)
        self.W_V = nn.Linear(hidden_size,hidden_size)
        self.W_e=nn.Parameter(torch.empty(M_WINDOW).normal_(mean=1,std=0.01))
        self.entropy=entropy
        
    def forward(self,last_state,all_states, IEs=None):
        Q=self.W_Q(last_state)
        K=self.W_K(all_states)
        V=self.W_V(all_states)
        #print(Q.shape)
        #print(V.shape)
        
        scores=torch.bmm(Q.unsqueeze(1),K.transpose(1,2))
        scores=scores/(K.size(-1)**0.5)
        #print(all_states.shape)
        #print(self.W_e.shape)
        if self.entropy:
            if self.mode=='SE':
                M_t=torch.tanh(self.W_e.unsqueeze(0)*IEs.squeeze(2)).unsqueeze(1)
            else:
                M_t=torch.exp(-self.W_e.unsqueeze(0)*IEs.squeeze(2)).unsqueeze(1)
            scores=scores*M_t
        attention_weights=F.softmax(scores,dim=-1)
        context=torch.bmm(attention_weights,V)
        return context.squeeze(1),attention_weights.squeeze(1)

class AdaptiveLSTM(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=3,hidden_size=1024, entropy=True, mode='IE'):
        super().__init__()
        self.hidden_size=hidden_size
        self.encoder_lstm=nn.LSTM(input_size,hidden_size=hidden_size,batch_first=True)
        self.decoder_lstm=nn.LSTM(hidden_size,hidden_size,batch_first=True)
        self.fc_out=nn.Linear(hidden_size,input_size)
        #self.decoder_dense_dir=nn.Linear(hidden_size,2)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
        self.attention_layer=AttentionLayer(hidden_size, self.input_window,entropy=entropy, mode=mode)
        self.entropy=entropy
        self.layer_norm = nn.LayerNorm(hidden_size)
    
    def forward(self,X):
        encoder_pos_inputs,encoder_ent, decoder_ent=X
        entropies=torch.cat((encoder_ent,decoder_ent),dim=1)
        encoder_outputs,(hidden,cell)=self.encoder_lstm(encoder_pos_inputs)
        encoder_outputs = self.layer_norm(encoder_outputs)

        #print(encoder_outputs.shape)
        #print(hidden.view(-1,self.hidden_size).shape)
        #print(torch.eq(hidden.view(-1,self.hidden_size),encoder_outputs[:,-1,:]))
        #exit()
        all_outputs=[]
        outputs=encoder_outputs
        inputs=encoder_pos_inputs[:,-1,:].unsqueeze(dim=1)
        for i in range(self.output_horizon):
            if self.entropy:
                context,attention_weights=self.attention_layer(hidden[-1],outputs, entropies[:,i:i+self.input_window,:])
            else:
                context,attention_weights=self.attention_layer(hidden[-1],outputs)
            decoder_input=context.unsqueeze(1)
            decoder_output,(hidden,cell)=self.decoder_lstm(decoder_input,(hidden,cell))
            decoder_output = self.layer_norm(decoder_output)
            #print(decoder_output.shape)
            outputs_delta=self.fc_out(decoder_output)
            #outputs_delta_dir=self.decoder_dense_dir(decoder_output)
            outputs_pos=toPosition([inputs,outputs_delta])
            outputs=torch.cat((outputs,decoder_output),dim=1)[:,1:,:]
            all_outputs.append(outputs_pos)
            inputs=outputs_pos
            #print(inputs.shape)
        all_outputs=torch.cat(all_outputs,dim=1)
        return all_outputs

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
        #pos_loss=MetricOrthLoss(pred_pos,true_pos)
        #print(pos_loss)
        #print(mse_pos)
        #print(mse_pos.shape)
        #print(pos_loss.shape)
        mse_vel = F.mse_loss(pred_vel, true_vel)
        #print(mse_vel)
        #print(mse_vel.shape)
        #sys.exit()
        return self.alpha*mse_pos + self.beta*mse_vel
    
def create_ALSTM_model(M_WINDOW,H_WINDOW,device='cpu',lr=1e-3, entropy=True, mode=None):
    model=AdaptiveLSTM(M_WINDOW,H_WINDOW,entropy=entropy,mode=mode).float().to(device)
    optimizer=optim.AdamW(model.parameters(),lr=lr,weight_decay=0.01)
    criterion=torch.nn.MSELoss()
    #optimizer=optim.AdamW(model.parameters(),lr=lr)
    criterion=CombinationLoss()
    return model,optimizer,criterion