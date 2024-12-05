import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchinfo import summary
import os

def MetricOrthLoss(pred_position, true_position):
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

class TRACK_POS(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=2,hidden_size=1024, output_size=None):
        super().__init__()
        if output_size:
            self.output_size=output_size
        else:
            self.output_size=input_size
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
    
    
class TRACK_POS_augmented(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,input_size=2,hidden_size=1024, output_size=None):
        super().__init__()
        if output_size:
            self.output_size=output_size
        else:
            self.output_size=input_size
        self.lstm_layer=nn.LSTM(input_size,hidden_size=hidden_size,batch_first=True)
        self.decoder_dense_mot=nn.Linear(hidden_size,2)
        self.decoder_dense_dir=nn.Linear(hidden_size,2)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
    
    def forward(self,X):
        encoder_pos_inputs,encoder_ent,decoder_pos_inputs,decoder_ent=X
        encoder_ent=encoder_ent.unsqueeze(2)
        decoder_ent=decoder_ent.unsqueeze(2)
        encoder_inputs=torch.cat([encoder_pos_inputs,encoder_ent],dim=2)
        #decoder_inputs=torch.cat([decoder_pos_inputs,decoder_ent[:,0,:].unsqueeze(dim=1)],dim=2)
        #decoder_ent=decoder_ent[:,1,:].unsqueeze(dim=1)
        #print(encoder_inputs.shape)
        #print(decoder_inputs.shape)
        #print(decoder_ent.shape)
        #exit()
        encoder_outputs,states=self.lstm_layer(encoder_inputs)
        all_outputs=[]
        inputs=decoder_pos_inputs
        for i in range(self.output_horizon):
            inputs=torch.cat([inputs,decoder_ent[:,i,:].unsqueeze(dim=1)],dim=2)
            decoder_output,states=self.lstm_layer(inputs,states)
            outputs_delta=self.decoder_dense_mot(decoder_output)
            outputs_delta_dir=self.decoder_dense_dir(decoder_output)
            outputs_pos=to_position(inputs,outputs_delta,outputs_delta_dir)
            all_outputs.append(outputs_pos)
            inputs=outputs_pos
            #print(inputs.shape)
        
        return torch.cat(all_outputs,dim=1)

def weighted_MetricOrthLoss(pred_position, true_position,IE,alpha):
    yaw_true = (true_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_true = (true_position[:, :, 1:2] - 0.5) * np.pi
    # Transform it to range -pi, pi for yaw and -pi/2, pi/2 for pitch
    yaw_pred = (pred_position[:, :, 0:1] - 0.5) * 2*np.pi
    pitch_pred = (pred_position[:, :, 1:2] - 0.5) * np.pi
    delta_long = torch.abs(torch.atan2(torch.sin(yaw_true - yaw_pred), torch.cos(yaw_true - yaw_pred)))
    numerator = torch.sqrt(torch.pow(torch.cos(pitch_pred)*torch.sin(delta_long), 2.0) + torch.pow(torch.cos(pitch_true)*torch.sin(pitch_pred)-torch.sin(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long), 2.0))
    denominator = torch.sin(pitch_true)*torch.sin(pitch_pred)+torch.cos(pitch_true)*torch.cos(pitch_pred)*torch.cos(delta_long)
    great_circle_distance = torch.abs(torch.atan2(numerator, denominator))
    weighted_loss=great_circle_distance*(IE*(1+alpha))
    return weighted_loss.mean()
 
def create_pos_only_model(M_WINDOW,H_WINDOW,input_size=2, lr=0.0005, device='cpu', loss=None):
    model=TRACK_POS(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,input_size=input_size).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=lr)
    if loss is not None:
        criterion=weighted_MetricOrthLoss
    else:
        criterion=MetricOrthLoss
    return model,optimizer,criterion

def create_pos_only_augmented_model(M_WINDOW,H_WINDOW,input_size=3, lr=0.0005, device='cpu',mode=None):
    model=TRACK_POS_augmented(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,input_size=input_size).to(device)
    optimizer=optim.AdamW(model.parameters(),lr=lr)
    criterion=MetricOrthLoss
    return model,optimizer,criterion


def train_pos_only(model,train_loader,validation_loader,optimizer=None,criterion=MetricOrthLoss,epochs=100,device="cpu", path=None):
    best_val_loss=float('inf')
    device=torch.device(device)
    model.to(device)
    if optimizer==None:
        optimizer=optim.Adam(model.parameters(),lr=0.0005)
    losses=[]
    val_losses=[]
    last_saved=0
    for epoch in range(epochs):
        model.train()
        epoch_losses=[]
        #print(model.state_dict())
        for ip,targets in train_loader:
            optimizer.zero_grad()
            ip=[t.squeeze(axis=1).to(device) for t in ip]
            targets=targets.squeeze(axis=1)
            #print(encoder_inputs)
            prediction=model(ip)
            loss=criterion(prediction,targets.to(device))
            loss.backward()
            optimizer.step()
            #print("-------")
            #print(model.state_dict())
            #return 0
            epoch_losses.append(loss.item())
        epoch_loss=sum(epoch_losses)/len(epoch_losses)
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss:{epoch_loss}")
        
        model.eval()
        epoch_val_losses=[]
        for ip,targets in validation_loader:
            ip=[t.squeeze(axis=1).to(device) for t in ip]
            targets=targets.squeeze(axis=1)
            #print(encoder_inputs)
            prediction=model(ip)
            loss=criterion(prediction,targets.to(device))
            #print("-------")
            #print(model.state_dict())
            #return 0
            epoch_val_losses.append(loss.item())
        epoch_val_loss=sum(epoch_val_losses)/len(epoch_val_losses)
        val_losses.append(epoch_val_loss)
        
        print(f"Epoch {epoch+1}/{epochs}, Train Loss:{epoch_loss:.4f}, Validation Loss:{epoch_val_loss:.4f}")
        last_saved+=1
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            last_saved=0
            checkpoint={
                'model_state_dict':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'losses':losses,
                'val_losses':val_losses,
                'epoch':'epoch'
            }
            save_path=os.path.join(path,f'class_trainer_Epoch_{epoch}.pth')
            torch.save(checkpoint,save_path)
            print(f"Model saved at {epoch+1} with validation loss: {epoch_val_loss:.4f}")
        if last_saved>20:
            break
    return losses, val_losses
    
    
