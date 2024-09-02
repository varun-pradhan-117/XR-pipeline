import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
#import numpy as np
#from torchinfo import summary
import os

def MetricOrthLoss(position_a, position_b,epsilon=1e-8):
    # Transform into directional vector in Cartesian Coordinate System
    # Transform into directional vector in Cartesian Coordinate System
    norm_a = torch.sqrt(torch.square(position_a[..., 0:1]) + torch.square(position_a[..., 1:2])
                        + torch.square(position_a[..., 2:3]))+epsilon
    norm_b = torch.sqrt(torch.square(position_b[..., 0:1]) + torch.square(position_b[..., 1:2])
                        + torch.square(position_b[..., 2:3]))+epsilon
    x_true = position_a[..., 0:1] / norm_a
    y_true = position_a[..., 1:2] / norm_a
    z_true = position_a[..., 2:3] / norm_a
    x_pred = position_b[..., 0:1] / norm_b
    y_pred = position_b[..., 1:2] / norm_b
    z_pred = position_b[..., 2:3] / norm_b
    # Finally compute orthodromic distance
    # great_circle_distance = np.arccos(x_true*x_pred+y_true*y_pred+z_true*z_pred)
    # To keep the values in bound between -1 and 1
    great_circle_distance = torch.acos(torch.clamp(x_true * x_pred + y_true * y_pred + z_true * z_pred, -1.0, 1.0))
    return great_circle_distance.mean()

# This way we ensure that the network learns to predict the delta angle
def toPosition(values):
    orientation = values[0]
    # The network returns values between 0 and 1, we force it to be between -1/2 and 1/2
    motion = values[1]
    return (orientation + motion)

def check_for_nans_or_infs(tensor, name='tensor'):
    if torch.isnan(tensor).any():
        print(f'NaN detected in {name}')
    if torch.isinf(tensor).any():
        print(f'Inf detected in {name}')

class TRACK_MODEL(nn.Module):
    def __init__(self,M_WINDOW,H_WINDOW,NUM_TILES_HEIGHT,NUM_TILES_WIDTH,input_size_pos=3,input_size_saliency=1,hidden_size=256):
        super().__init__()
        # Define encoders
        self.pos_enc=nn.LSTM(input_size_pos,hidden_size=hidden_size,batch_first=True)
        self.sal_enc=nn.LSTM(input_size_saliency*NUM_TILES_HEIGHT*NUM_TILES_WIDTH, hidden_size=hidden_size,batch_first=True)
        self.fuse_1_enc=nn.LSTM(hidden_size*2,hidden_size=hidden_size,batch_first=True)
        
        # Define decoders
        self.pos_dec=nn.LSTM(input_size_pos,hidden_size=hidden_size,batch_first=True)
        self.sal_dec=nn.LSTM(input_size_saliency*NUM_TILES_HEIGHT*NUM_TILES_WIDTH, hidden_size=hidden_size,batch_first=True)
        self.fuse_1_dec=nn.LSTM(hidden_size*2,hidden_size=hidden_size,batch_first=True)
        
        self.fc_1=nn.Linear(hidden_size,hidden_size)
        self.fc_layer_out=nn.Linear(hidden_size,3)
        self.output_horizon=H_WINDOW
        self.input_window=M_WINDOW
        
        #self._init_weights()
    
    def forward(self, X):
        encoder_pos_inputs,encoder_sal_inputs,decoder_pos_inputs,decoder_sal_inputs=X
        
        # Encode position inputs
        out_enc_pos, (h_n_pos,c_n_pos)= self.pos_enc(encoder_pos_inputs)
        states_pos=[h_n_pos,c_n_pos]
        #out_enc_pos = self.dropout(out_enc_pos)
        #check_for_nans_or_infs(out_enc_pos, 'out_enc_pos')
        
        # Flatten saliency input
        flat_enc_sal_inputs = torch.flatten(encoder_sal_inputs, start_dim=-2)
        out_enc_sal, (h_n_sal,c_n_sal) = self.sal_enc(flat_enc_sal_inputs)
        states_sal=[h_n_sal,c_n_sal]
        #out_enc_sal = self.dropout(out_enc_sal)
        #check_for_nans_or_infs(out_enc_sal, 'out_enc_sal')
        
        # Concatenate encoder outputs
        #with torch.autograd.detect_anomaly():
        conc_out_enc = torch.cat([out_enc_sal, out_enc_pos], dim=-1)
        fuse_out_enc, (h_n_fuse,c_n_fuse) = self.fuse_1_enc(conc_out_enc)
        states_fuse=[h_n_fuse,c_n_fuse]
        #fuse_out_enc = self.dropout(fuse_out_enc)
        #check_for_nans_or_infs(conc_out_enc, 'conc_out_enc')
        #check_for_nans_or_infs(fuse_out_enc, 'fuse_out_enc')
        
        dec_input = decoder_pos_inputs
        all_pos_outputs = []
        for t in range(self.output_horizon):
            
            # Decode pos at timestep
            dec_pos_out, (h_n_pos,c_n_pos)= self.pos_dec(dec_input,states_pos)
            states_pos=[h_n_pos,c_n_pos]
            #dec_pos_out = self.dropout(dec_pos_out)
            #check_for_nans_or_infs(dec_pos_out, 'dec_pos_out')
            
            # Decode saliency at current timestep
            selected_timestep_saliency = decoder_sal_inputs[:, t:t + 1]
            flatten_timestep_saliency = torch.flatten(selected_timestep_saliency, start_dim=-2)
            dec_sal_out, (h_n_sal,c_n_sal) = self.sal_dec(flatten_timestep_saliency, states_sal)
            states_sal=[h_n_sal,c_n_sal]
            #check_for_nans_or_infs(dec_sal_out, 'dec_sal_out')
            
            # Decode concatenated values
            #with torch.autograd.detect_anomaly():
            dec_out = torch.cat((dec_sal_out, dec_pos_out), dim=-1)
            fuse_out_dec_1, (h_n_fuse, c_n_fuse) = self.fuse_1_dec(dec_out, states_fuse)
            states_fuse=[h_n_fuse,c_n_fuse]
            #fuse_out_dec_1 = self.dropout(fuse_out_dec_1)
            #check_for_nans_or_infs(dec_out, 'dec_out')
            #check_for_nans_or_infs(fuse_out_dec_1, 'fuse_out_dec_1')
            
            # FC layers
            dec_fuse_out = self.fc_1(fuse_out_dec_1)

            #dec_fuse_out = self.dropout(dec_fuse_out)
            outputs_delta = self.fc_layer_out(dec_fuse_out)
            # Apply toposition
            decoder_pred=toPosition([dec_input,outputs_delta])
            #check_for_nans_or_infs(dec_fuse_out, 'dec_fuse_out')
            #check_for_nans_or_infs(outputs_delta, 'outputs_delta')
            #check_for_nans_or_infs(decoder_pred, 'decoder_pred')
            all_pos_outputs.append(decoder_pred)
            dec_input=decoder_pred
        #with torch.autograd.detect_anomaly():    
        decoder_outputs_pos=torch.cat(all_pos_outputs,dim=1)
        return decoder_outputs_pos
    
    def _init_weights(self):
        def init_lstm(lstm):
            for name, param in lstm.named_parameters():
                if 'weight_ih' in name:
                    init.xavier_uniform_(param.data)  # glorot_uniform initializer
                elif 'weight_hh' in name:
                    init.orthogonal_(param.data)  # orthogonal initializer
                elif 'bias' in name:
                    param.data.fill_(0)  # zeros initializer
                    # Optionally set forget gate biases to 1
                    n = param.size(0)
                    param.data[n//4:n//2].fill_(1.0)
            
        init_lstm(self.pos_enc)
        init_lstm(self.sal_enc)
        init_lstm(self.fuse_1_enc)
        init_lstm(self.pos_dec)
        init_lstm(self.sal_dec)
        init_lstm(self.fuse_1_dec)
        # Initialize Linear layers
        init.xavier_uniform_(self.fc_1.weight)
        self.fc_1.bias.data.fill_(0)
        init.xavier_uniform_(self.fc_layer_out.weight)
        self.fc_layer_out.bias.data.fill_(0)
    



    
def train_model(model,train_loader,validation_loader,optimizer=None,criterion=torch.nn.MSELoss(),epochs=100,device="cpu", path=None, metric=None):
    best_val_loss=float('inf')
    device=torch.device(device)
    os.makedirs(path, exist_ok=True)
    #torch.autograd.set_detect_anomaly(True)
    if optimizer is None:
        optimizer=optim.AdamW(model.parameters(),lr=5e-4)
        
    losses=[]
    val_losses=[]
    last_saved=0
    epoch_losses={}
    metric_vals={}
    if metric is not None:
        for name,func in metric.items():
            metric_vals[name]={}
    for epoch in range(epochs):
        model.train()
        epoch_losses[epoch]=[]
        #print(model.state_dict())
        if metric is not None:
            for name,func in metric.items():
                metric_vals[name][epoch]=[]
        metric_val={}
        for idx,(ip,targets) in enumerate(train_loader):
            optimizer.zero_grad()
            ip=(t.squeeze(axis=1).float().to(device) for t in ip)
            targets=targets.squeeze(axis=1).float().to(device)
            #print(encoder_inputs)
            prediction=model(ip)
            loss=criterion(prediction,targets)
            loss.backward()

                    
            if metric is not None:
                for name,func in metric.items():
                    metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                    metric_vals[epoch].append(metric_val)
            #prev_grads={name: param.grad for name, param in model.named_parameters() if param.grad is not None}
                    
            # Check for NaN loss
            if torch.isnan(loss).item():
                print(f'batch={idx}')
                print("NaN loss encountered during backward pass")
                print(f"prediction: {prediction}")
                print(f"targets: {targets}")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'inputs': ip,
                    'prediction': prediction,
                    'targets': targets
                }, os.path.join(path, 'batch_nan_loss.pth') if path else 'batch_nan_loss.pth')
                print("NAN Loss")
                return 0, 0
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses[epoch].append(loss.item())
            print(f"Batch {idx}/{len(train_loader)} - loss:{epoch_losses[epoch][-1]:.4f}", end=' ')
            if metric is not None:
                for name, value in metric_val.items():
                        print(f" - {name}: {value:.4f}",end=' ')
            print()
            #print(50*"*")
        #print(sum(epoch_losses))
        #print(len(epoch_losses))
        epoch_loss=sum(epoch_losses[epoch])/len(epoch_losses[epoch])
        losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss:{epoch_loss:.4f}",end=' ')
        for name, value_list in metric_vals.items():
                print(f" - {name}: {torch.mean(value_list):.4f}")
                
                
                
        model.eval()
        epoch_val_losses=[]
        eval_metrics={}
        if metric is not None:
            for name,func in metric.items():
                eval_metrics[name]=[]
        for ip,targets in validation_loader:
            ip=(t.squeeze(axis=1).to(device) for t in ip)    
            targets=targets.squeeze(axis=1)     
            #print(encoder_inputs)
            prediction=model(ip)
            loss=criterion(prediction,targets.to(device))
            #print("-------")
            #print(model.state_dict())
            #return 0
            if metric is not None:
                for name,func in metric.items():
                    metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                    metric_vals[name].append(metric_val)
            epoch_val_losses.append(loss.item())
        epoch_val_loss=sum(epoch_val_losses)/len(epoch_val_losses)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss:{epoch_loss:.4f}, Validation Loss:{epoch_val_loss:.4f}",end=' ')
        if metric is not None:
            for name, value_list in metric_vals.items():
                print(f" - {name}: {torch.mean(value_list):.4f}",end=' ')
        print()    
        last_saved+=1
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            torch.save(model.state_dict(),f'{path}.pth')
            last_saved=0
            print(f"Model saved at {epoch+1} with validation loss: {epoch_val_loss:.4f}")
        if last_saved>5:
            break
    return losses, val_losses

def create_sal_model(M_WINDOW,H_WINDOW,NUM_TILES_HEIGHT,NUM_TILES_WIDTH,device='cpu',lr=1e-3):
    model=TRACK_MODEL(M_WINDOW,H_WINDOW,NUM_TILES_HEIGHT,NUM_TILES_WIDTH).float().to(device)
    criterion=torch.nn.MSELoss()
    optimizer=optim.AdamW(model.parameters(),lr=lr)
    return model,optimizer,criterion