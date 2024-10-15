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
    def __init__(self,hidden_size):
        super().__init__()
        self.W_K=nn.Linear(hidden_size,hidden_size)
        self.W_Q=nn.Linear(hidden_size,hidden_size)
        self.W_V = nn.Linear(hidden_size,hidden_size)
        
    def forward(self,decoder_hidden,encoder_outputs):
        Q=self.W_Q(decoder_hidden)
        K=self.W_K(encoder_outputs)
        V=self.W_V(encoder_outputs)
        
        scores=torch.bmm(Q.unsqueeze(1),K.transpose(1,2))
        scores=scores/(K.size(-1)**0.5)
        attention_weights=F.softmax(scores,dim=-1)
        context=torch.bmm(attention_weights,V)
        return context.squeeze(1),attention_weights.squeeze(1)

class AdaptiveLSTM(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=3,hidden_size=1024):
        super().__init__()
        self.encoder_lstm=nn.LSTM(input_size,hidden_size=hidden_size,batch_first=True)
        self.decoder_lstm=nn.LSTM(hidden_size,hidden_size,batch_first=True)
        self.fc_out=nn.Linear(hidden_size,input_size)
        #self.decoder_dense_dir=nn.Linear(hidden_size,2)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
    
    def forward(self,X):
        encoder_inputs,decoder_inputs=X
        encoder_outputs,(hidden,cell)=self.encoder_lstmlstm(encoder_inputs)
        all_outputs=[]
        inputs=decoder_inputs
        for _ in range(self.output_horizon):
            decoder_output,(hidden,cell)=self.decoder_lstm(inputs,(hidden,cell))
            outputs_delta=self.fc_out(decoder_output)
            #outputs_delta_dir=self.decoder_dense_dir(decoder_output)
            outputs_pos=toPosition([inputs,outputs_delta])
            all_outputs.append(outputs_pos)
            inputs=outputs_pos
            #print(inputs.shape)
        all_outputs=torch.cat(all_outputs,dim=1)
        return all_outputs