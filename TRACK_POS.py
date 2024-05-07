import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchinfo import summary

def MetricOrthLoss(true_position, pred_position):
    yaw_true = (true_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
    delta_long = torch.abs(torch.atan2(torch.sin(yaw_true - yaw_pred), torch.cos(yaw_true - yaw_pred)))
    numerator = torch.sqrt(torch.pow(torch.cos(pitch_pred)*torch.sin(delta_long), 2.0) + torch.pow(torch.cos(pitch_true)*torch.sin(pitch_pred)-torch.sin(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long), 2.0))
    denominator = torch.sin(pitch_true)*torch.sin(pitch_pred)+torch.cos(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long)
    great_circle_distance = torch.abs(torch.atan2(numerator, denominator))
    return great_circle_distance.mean()

def to_position(inputs,outputs_delta,outputs_delta_dir):
    orientation=inputs
    magnitudes=outputs_delta
    directions=outputs_delta_dir
    motion=magnitudes*directions
    
    yaw_pred_wo_corr=orientation[:,:,0:1]+motion[:,:,0:1]
    pitch_pred_wo_corr = orientation[:, :, 1:2] + motion[:, :, 1:2]
    cond_above = (pitch_pred_wo_corr > 1.0).float()
    cond_correct = ((pitch_pred_wo_corr <= 1.0) & (pitch_pred_wo_corr >= 0.0)).float()
    cond_below = (pitch_pred_wo_corr < 0.0).float()

    pitch_pred = cond_above * (1.0 - (pitch_pred_wo_corr - 1.0)) + cond_correct * pitch_pred_wo_corr + cond_below * (-pitch_pred_wo_corr)
    yaw_pred = torch.fmod(cond_above * (yaw_pred_wo_corr - 0.5) + cond_correct * yaw_pred_wo_corr + cond_below * (yaw_pred_wo_corr - 0.5), 1.0)
    return torch.cat([yaw_pred, pitch_pred], -1)

class seq2seq(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=2,hidden_size=1024):
        super().__init__()
        self.lstm_layer=nn.LSTM(input_size,hidden_size=hidden_size,batch_first=True)
        self.decoder_dense_mot=nn.Linear(hidden_size,2)
        self.decoder_dense_dir=nn.Linear(hidden_size,2)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
    
    def forward(self,encoder_inputs,decoder_inputs):
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
    
def train_pos_only(model,train_loader,validation_loader,optimizer=None,criterion=MetricOrthLoss,epochs=100):
    if optimizer==None:
        optimizer=optim.Adam(model.parameters(),lr=0.0005)
    losses=[]
    for epoch in range(epochs):
        model.train()
        epoch_losses=[]
        #print(model.state_dict())
        for ip,targets in train_loader:
            optimizer.zero_grad()
            encoder_inputs,decoder_inputs=ip
            encoder_inputs=encoder_inputs.squeeze()
            decoder_inputs=decoder_inputs.squeeze(axis=1)
            #print(encoder_inputs)
            prediction=model(encoder_inputs,decoder_inputs)
            loss=criterion(prediction,targets)
            loss.backward()
            optimizer.step()
            #print("-------")
            #print(model.state_dict())
            #return 0
            epoch_losses.append(loss.item())
        epoch_loss=sum(epoch_losses)/len(epoch_losses)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss:{epoch_loss}")
    return losses
    
    
def create_pos_only_model(M_WINDOW,H_WINDOW,input_size=2):
    model=seq2seq(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,input_size=input_size)
    optimizer=optim.Adam(model.parameters(),lr=0.0005)
    criterion=MetricOrthLoss
    return model,optimizer,criterion