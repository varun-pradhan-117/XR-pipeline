import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from torchinfo import summary
import os



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