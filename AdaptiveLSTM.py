import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchinfo import summary
import os

class TRACK_POS(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=2,hidden_size=1024):
        super().__init__()
        self.lstm_layer=nn.LSTM(input_size,hidden_size=hidden_size,batch_first=True)
        self.decoder_dense_mot=nn.Linear(hidden_size,2)
        self.decoder_dense_dir=nn.Linear(hidden_size,2)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
    
    def forward(self,X):
        encoder_inputs,decoder_inputs=X
        encoder_outputs,states=self.lstm_layer(encoder_inputs)
        all_outputs=[]
        inputs=decoder_inputs
        for _ in range(self.output_horizon):
            decoder_output,states=self.lstm_layer(inputs,states)
            outputs_delta=self.decoder_dense_mot(decoder_output)
            outputs_delta_dir=self.decoder_dense_dir(decoder_output)
            outputs_pos=to_position(inputs,outputs_delta,outputs_delta_dir)
            all_outputs.append(outputs_pos)
            inputs=outputs_pos
            #print(inputs.shape)
        
        return torch.cat(all_outputs,dim=1)