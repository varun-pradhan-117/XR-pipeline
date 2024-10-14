import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import numpy as np
from torchinfo import summary
import os
import shutil
import matplotlib.pyplot as plt
from Utils import get_velocities


def train_model(model,train_loader,validation_loader,optimizer=None,criterion=torch.nn.MSELoss(),epochs=100,device="cpu", path=None, metric=None, tolerance=5, verbose=False, model_name=None):
    best_val_loss=float('inf')
    device=torch.device(device)
    if os.path.isdir(path):
        print(f"Clearing contents of the existing folder: {path}")
        shutil.rmtree(path)  # Deletes all contents in the folder
        print(f"Creating folder: {path}")
        os.makedirs(path)
    if not os.path.isdir(path):
        print(f"Making folder {path}")
        os.makedirs(path)
    #torch.autograd.set_detect_anomaly(True)
    if optimizer is None:
        optimizer=optim.AdamW(model.parameters(),lr=5e-4)
        
    losses=[]
    val_losses=[]
    last_saved=0
    epoch_losses={}
    metric_vals={}
    
    for epoch in range(epochs):
        model.train()
        epoch_losses[epoch]=[]
        #print(model.state_dict())
        metric_vals[epoch]={}
        if metric is not None:
            for name,func in metric.items():
                metric_vals[epoch][name]=[]
        metric_val={}
        for idx,(ip,targets) in enumerate(train_loader):
            optimizer.zero_grad()
            #print(ip)
            ip=[t.squeeze(axis=1).float().to(device) for t in ip]
            targets=targets.squeeze(axis=1).float().to(device)
            #print(encoder_inputs)
            if model_name=='DVMS':
                prediction=model(ip,targets)
                loss=criterion(*prediction)['loss']
            else:
                prediction=model(ip)
                
                if model_name in ['VPT360','AMH']:
                    norm = torch.norm(prediction, dim=-1, keepdim=True) + 1e-8  # Avoid division by zero
                    prediction = prediction / norm 
                    """ magnitude=torch.norm(prediction,dim=-1)
                    is_unit_vector = torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-6)

                    # Print the results
                    print("Magnitude of predictions:", magnitude.shape)
                    print("Are predictions unit vectors?:", is_unit_vector)
                    return """
                    pred_vels=get_velocities(ip[0],prediction)
                    target_vels=get_velocities(ip[0],targets)
                    
                    loss=criterion(prediction,targets,pred_vels,target_vels)
                    #print(criterion)
                    #print(loss)
                    #return
                else:
                    loss=criterion(prediction,targets)
            loss.backward()
            
            
            
            if metric is not None:
                for name,func in metric.items():
                    if name=='combinatorial_loss':
                        pred_vels=get_velocities(ip[0],prediction.detach())
                        target_vels=get_velocities(ip[0],targets.detach())
                        metric_val[name]=torch.mean(func(prediction.detach(),targets,pred_vels,target_vels)).item()
                    else:
                        metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                    metric_vals[epoch][name].append(metric_val[name])
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
            if verbose:
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
        #print(metric_vals)
        for name, value_list in metric_vals[epoch].items():
                print(f" - {name}: {np.mean(value_list):.4f}")
                
                
                
        model.eval()
        epoch_val_losses=[]
        eval_metrics={}
        if metric is not None:
            for name,func in metric.items():
                eval_metrics[name]=[]
        for ip,targets in validation_loader:
            ip=[t.squeeze(axis=1).to(device) for t in ip]    
            targets=targets.squeeze(axis=1).to(device)     
            #print(encoder_inputs)
            if model_name=='DVMS':
                prediction=model(ip,targets)
                loss=criterion(*prediction)['loss']
            else:
                prediction=model(ip)
                if model_name in ['VPT360','AMH']:
                    pred_vels=get_velocities(ip[0],prediction)
                    target_vels=get_velocities(ip[0],targets)
                    
                    loss=criterion(prediction,targets,pred_vels,target_vels)
                else:
                    loss=criterion(prediction,targets)
            #print("-------")
            #print(model.state_dict())
            #return 0
            if metric is not None:
                for name,func in metric.items():
                    if name=='combinatorial_loss':
                        pred_vels=get_velocities(ip[0],prediction.detach())
                        target_vels=get_velocities(ip[0],targets.detach())
                        metric_val[name]=torch.mean(func(prediction.detach(),targets,pred_vels,target_vels)).item()
                    else:
                        metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                    eval_metrics[name].append(metric_val[name])
            epoch_val_losses.append(loss.item())
        epoch_val_loss=sum(epoch_val_losses)/len(epoch_val_losses)
        val_losses.append(epoch_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss:{epoch_loss:.4f}, Validation Loss:{epoch_val_loss:.4f}",end=' ')
        if metric is not None:
            for name, value_list in eval_metrics.items():
                print(f" - {name}: {np.mean(value_list):.4f}",end=' ')
        print()    
        last_saved+=1
        if epoch_val_loss<best_val_loss:
            best_val_loss=epoch_val_loss
            checkpoint={
                'model_state_dict':model.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'losses':losses,
                'val_losses':val_losses,
                'epoch':'epoch'
            }
            save_path=os.path.join(path,f'Epoch_{epoch}.pth')

            torch.save(checkpoint,save_path)
            last_saved=0
            print(f"Model saved at {epoch+1} with validation loss: {epoch_val_loss:.4f}")
        if last_saved>tolerance:
            break
    return losses, val_losses


def test_model(model, validation_loader, criterion=torch.nn.MSELoss(), device='cpu',path=None, metric=None, model_name=None, K=None, vid_name=None):
    model.eval()
    if vid_name is not None:
        path=os.path.join(path,vid_name)
    print(path)
    if not os.path.isdir(path):
        print(f"Making folder {path}")
        os.makedirs(path)
    eval_metrics={}
    if metric is not None:
        for name,func in metric.items():
            eval_metrics[name]=[]
    for ip, targets in validation_loader:
        ip=[t.squeeze(axis=1).to(device) for t in ip] 
        targets=targets.squeeze(axis=1).to(device)  
        #print(targets.shape)
        if model_name=='DVMS':
            prediction=model.sample(ip)
            #print(prediction.shape)
        else:
            prediction=model(ip)
        
        #print(prediction.shape)

        metric_val={}
        if metric is not None:
            for name,func in metric.items():
                if model_name=='DVMS':
                    metric_val=func(prediction.detach(),targets,k=K)
                else:
                    metric_val=func(prediction.detach(),targets).squeeze(-1)
                eval_metrics[name].append(metric_val)
                #avg_metrics=func(prediction.detach(),targets).mean(dim=0).squeeze(-1)
                #metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                #metric_vals[name].append(metric_val)
    for name in eval_metrics:
        eval_metrics[name]=torch.cat(eval_metrics[name],dim=0).mean(dim=0).squeeze(-1)
        # Save the metrics values to a file
        metric_values = eval_metrics[name].cpu().numpy()
        values_save_path = os.path.join(path, f'{name}_values.npy')
        np.save(values_save_path, metric_values)
        print(f"Saved metric values for {name} at {values_save_path}")

        # Plot the metric and save the figure
        timestamps = (np.arange(1, 26) / 5)  # [0.2, 0.4, ..., 5.0] assuming 5 fps
        plt.figure(figsize=(10, 5))
        plt.plot(timestamps, metric_values, marker='o', label=f'{name}')
        plt.title(f'Metric: {name}')
        plt.xlabel('Time (seconds)')
        plt.ylabel(f'{name} Value')
        plt.grid(True)
        plt.legend()

        # Save the plot
        plot_save_path = os.path.join(path, f'{name}_plot.png')
        plt.savefig(plot_save_path)
        plt.close()
        print(f"Saved plot for {name} at {plot_save_path}")
    
    
def test_full_vid(model, validation_loader, criterion=torch.nn.MSELoss(), device='cpu',path=None,
                  metric=None, model_name=None, K=None, vid_name=None, user_name=None):
    model.eval()
    if vid_name is not None:
        path=os.path.join(path,vid_name,user_name)
    print(path)
    if not os.path.isdir(path):
        print(f"Making folder {path}")
        os.makedirs(path)
    eval_metrics={}

    indices={"next": 0, 1: 4, 2: 9, 3: 14, 4: 19, 5: 24} 
    if metric is not None:
        for name,func in metric.items():
            eval_metrics[name]=[]
    for ip, targets in validation_loader:
        ip=[t.squeeze(axis=1).to(device) for t in ip] 
        targets=targets.squeeze(axis=1).to(device)  
        #print(targets.shape)
        if model_name=='DVMS':
            prediction=model.sample(ip)
            #print(prediction.shape)
        else:
            prediction=model(ip)
        
        #print(prediction.shape)

        metric_val={}
        if metric is not None:
            for name,func in metric.items():
                if model_name=='DVMS':
                    metric_val=func(prediction.detach(),targets,k=K)
                else:
                    metric_val=func(prediction.detach(),targets).squeeze(-1)
                
                eval_metrics[name].append(metric_val)
                #avg_metrics=func(prediction.detach(),targets).mean(dim=0).squeeze(-1)
                #metric_val[name]=torch.mean(func(prediction.detach(),targets)).item()
                #metric_vals[name].append(metric_val)
    for name in eval_metrics:
        all_metrics=torch.cat(eval_metrics[name],dim=0)
        #print(eval_metrics[name].shape)
        for time,idx in indices.items():
            specific_values = all_metrics[:, idx].cpu().numpy()

            # Save the metrics values to a file
            values_save_path = os.path.join(path, f'{name}_{time}_values.npy')
            np.save(values_save_path, specific_values)
            print(f"Saved metric values for {name} and timestep {time} at {values_save_path}")

            # Plot the metric and save the figure
            """ timestamps = (np.arange(1, 26) / 5)  # [0.2, 0.4, ..., 5.0] assuming 5 fps
            plt.figure(figsize=(10, 5))
            plt.plot(timestamps, metric_values, marker='o', label=f'{name}')
            plt.title(f'Metric: {name}')
            plt.xlabel('Time (seconds)')
            plt.ylabel(f'{name} Value')
            plt.grid(True)
            plt.legend()

            # Save the plot
            plot_save_path = os.path.join(path, f'{name}_plot.png')
            plt.savefig(plot_save_path)
            plt.close()
            print(f"Saved plot for {name} at {plot_save_path}") """
        