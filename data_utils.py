import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos,load_dict_from_csv,all_metrics
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split

NOSSDAV_sample_folder="Fan_NOSSDAV_17/sampled_dataset"
def fan_nossdav_split(NOSSDAV_sample_folder,user_ratio=0.6, video_ratio=0.6):
    list_of_videos = [o for o in os.listdir(NOSSDAV_sample_folder) if not o.endswith('.gitkeep')]
    list_of_users=[o for o in os.listdir(os.path.join(NOSSDAV_sample_folder,list_of_videos[0]))]
    
    # Determine the number of users and videos for training
    num_train_users = int(len(list_of_users) * user_ratio)
    num_train_videos = int(len(list_of_videos) * video_ratio)
    
    # Shuffle the users and videos to ensure randomness
    random.shuffle(list_of_users)
    random.shuffle(list_of_videos)
    
    # Split users and videos into training and testing sets
    train_users = list_of_users[:num_train_users]
    test_users = list_of_users[num_train_users:]
    train_videos = list_of_videos[:num_train_videos]
    test_videos = list_of_videos[num_train_videos:]
    # Randomly select users and videos for the special cases
    # where either the user or video is new in the test set
    special_test_users = test_users
    special_test_videos = test_videos
    
    new_user_new_video=[[user,video] for user in test_users for video in test_videos]
    old_user_new_video=[[random.choice(train_users),video] for video in special_test_videos]
    new_user_old_video=[[user,random.choice(train_videos)] for user in special_test_users]
    train_data=[[user,video] for user in train_users for video in train_videos]
    
    return np.array(train_data),np.array(new_user_new_video),np.array(old_user_new_video),np.array(new_user_old_video)
 
def transform_batches_cartesian_to_normalized_eulerian(positions_in_batch):
    positions_in_batch = np.array(positions_in_batch)
    eulerian_batches = [[cartesian_to_eulerian(pos[0], pos[1], pos[2]) for pos in batch] for batch in positions_in_batch]
    eulerian_batches = np.array(eulerian_batches) / np.array([2*np.pi, np.pi])
    return eulerian_batches

def transform_normalized_eulerian_to_cartesian(positions):
    positions = positions * np.array([2*np.pi, np.pi])
    eulerian_samples = [eulerian_to_cartesian(pos[0], pos[1]) for pos in positions]
    return np.array(eulerian_samples)

def reshape_ip(input):
    input=[ip.squeeze(dim=1) for ip in input]
    return input

def get_trace_pairs(users_per_video,video_list,user_list):
    trace_pairs=[]
    for video in video_list:
        for user in user_list:
            if user in users_per_video[video]:
                trace_pairs.append([user,video])
    return np.array(trace_pairs)

def var_norm(data, epsilon=1e-5):
    return np.var(data)/max(np.mean(data),epsilon)
def split_videos(dataset_dir,bins=2, video_test_size=0.4):
    data_dir=os.path.join(dataset_dir,'video_data')
    videos=os.listdir(data_dir)
    AE={}
    SI={}
    TI={}
    CE={}
    for video in videos:
        vid_path=os.path.join(data_dir,video)
        siti_path=os.path.join(data_dir,video)
        AE[video]=np.load(os.path.join(vid_path,f'{video}_AEs.npy'))
        AE[video]=np.mean(AE[video])
        CE[video]=np.load(os.path.join(vid_path,f'{video}_content_entropy.npy'))
        CE[video]=np.mean(CE[video])
        SI[video]=np.load(os.path.join(siti_path,f'{video}_SI.npy'))
        SI[video]=np.mean(SI[video])
        TI[video]=np.load(os.path.join(siti_path,f'{video}_TI.npy'))
        TI[video]=np.mean(TI[video])
    ae_bins = pd.qcut(list(AE.values()), q=bins, labels=False)
    ce_bins = pd.qcut(list(CE.values()), q=bins, labels=False)
    si_bins = pd.qcut(list(SI.values()), q=bins, labels=False)
    ti_bins = pd.qcut(list(TI.values()), q=bins, labels=False)
    
    strat_keys = [f"{ae}_{ce}" for ae, ce in zip(ae_bins, ce_bins)]
    
    try:
        train_videos, test_videos = train_test_split(videos, test_size=video_test_size, stratify=strat_keys)
    except ValueError :
        print("Can't use combined stratified key")
        ae_variance = var_norm(list(AE.values()))
        ce_variance = var_norm(list(CE.values()))
        if ae_variance > ce_variance:
            strat_keys = ae_bins
            print("Using AE for stratification")
        else:
            strat_keys = ce_bins
            print("Using CE for stratification")
        
        try:
            train_videos, test_videos = train_test_split(videos, test_size=video_test_size, stratify=strat_keys)
        except ValueError:
            print("Can't use stratified keys, using random split instead.")
            train_videos, test_videos = train_test_split(videos, test_size=video_test_size)
    
    """ # Separate the metrics into train and test sets
    train_AE = [AE[video] for video in train_videos]
    train_CE = [CE[video] for video in train_videos]

    test_AE = [AE[video] for video in test_videos]
    test_CE = [CE[video] for video in test_videos]
    # Step 4: Plot the metrics

    plt.figure(figsize=(12, 6))

    # Plot AE vs CE
    plt.scatter(train_AE, train_CE, color='blue', label='Train')
    plt.scatter(test_AE, test_CE, color='red', label='Test')
    plt.xlabel('AE')
    plt.ylabel('CE')
    plt.title('AE vs CE')
    plt.legend()

    plt.show() """
    return train_videos,test_videos

def split_data_all_users(dataset_dir,total_users,users_per_video,bins=2,video_test_size=0.4,user_test_size=0.4):
    train_vids,test_vids=split_videos(dataset_dir=dataset_dir,bins=bins,video_test_size=video_test_size)
    random.shuffle(total_users)
    num_train_vids=len(train_vids)
    
    # Get train and test users
    num_train_users=int(len(total_users)*(1-video_test_size))
    train_users = total_users[:num_train_users]
    test_users = total_users[num_train_users:]
    
    train_traces=get_trace_pairs(users_per_video,train_vids,train_users)
    test_traces=get_trace_pairs(users_per_video,test_vids,test_users)
    new_video_old_user_traces=get_trace_pairs(users_per_video,test_vids,train_users)
    old_video_new_user_traces=get_trace_pairs(users_per_video,train_vids,test_users)
    return train_traces,test_traces,new_video_old_user_traces,old_video_new_user_traces, train_vids,test_vids
    
def save_unique_videos_to_csv(unique_videos, output_csv_path):
        # Prepare data for writing
        data_to_write = [['video']] + [[video] for video in unique_videos]

        # Write the data to a CSV file
        with open(output_csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(data_to_write)

        print(f"Unique videos saved to {output_csv_path}")
        
class PositionDataset(Dataset):
    def __init__(self,list_IDs,future_window,M_WINDOW, model_name, all_traces,all_saliencies=None,all_headmaps=None, video_name=None):
        self.list_IDs=list_IDs
        self.all_saliencies=all_saliencies
        self.all_traces=all_traces
        self.all_headmaps=all_headmaps
        self.M_WINDOW=M_WINDOW
        self.model_name=model_name
        self.future_window=future_window
        # Filter list_IDs by video_name if specified
        if video_name is not None:
            self.list_IDs = [ID for ID in self.list_IDs if ID['video'] == video_name]
        
    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self,index):
        ID=self.list_IDs[index]
        user=ID['user']
        video=ID['video']
        tstamp=ID['time-stamp']
        
        encoder_pos_inputs_for_batch = []
        encoder_sal_inputs_for_batch = []
        decoder_pos_inputs_for_batch = []
        decoder_sal_inputs_for_batch = []
        decoder_outputs_for_batch = []
        
        if self.model_name not in ['pos_only', 'pos_only_3d_loss', 'MM18','DVMS']:
            encoder_sal_inputs_for_batch.append(self.all_saliencies[video][tstamp-self.M_WINDOW+1:tstamp+1])
            decoder_sal_inputs_for_batch.append(self.all_saliencies[video][tstamp+1:tstamp+self.future_window+1])
        if self.model_name == 'CVPR18_orig':
            encoder_pos_inputs_for_batch.append(self.all_traces[video][user][tstamp-self.M_WINDOW+1:tstamp+1])
            decoder_outputs_for_batch.append(self.all_traces[video][user][tstamp+1:tstamp+1+1])
        elif self.model_name == 'MM18':
            encoder_sal_inputs_for_batch.append(torch.cat((self.all_saliencies[video][tstamp-self.M_WINDOW+1:tstamp+1], self.all_headmaps[video][user][tstamp-self.M_WINDOW+1:tstamp+1]), dim=1))
            decoder_outputs_for_batch.append(self.all_headmaps[video][user][tstamp+self.future_window+1])
        else:
            encoder_pos_inputs_for_batch.append(self.all_traces[video][user][tstamp-self.M_WINDOW:tstamp])
            decoder_pos_inputs_for_batch.append(self.all_traces[video][user][tstamp:tstamp+1])
            decoder_outputs_for_batch.append(self.all_traces[video][user][tstamp+1:tstamp+self.future_window+1])

        encoder_pos_inputs_for_batch=np.array(encoder_pos_inputs_for_batch)
        encoder_sal_inputs_for_batch=np.array(encoder_sal_inputs_for_batch)
        decoder_pos_inputs_for_batch=np.array(decoder_pos_inputs_for_batch)
        decoder_sal_inputs_for_batch=np.array(decoder_sal_inputs_for_batch)
        decoder_outputs_for_batch=np.array(decoder_outputs_for_batch)
        if self.model_name == 'TRACK' or self.model_name == 'TRACK_AblatSal' or self.model_name == 'TRACK_AblatFuse':
            return [torch.tensor(encoder_pos_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(encoder_sal_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(decoder_pos_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(decoder_sal_inputs_for_batch,dtype=torch.float32)], torch.tensor(decoder_outputs_for_batch,dtype=torch.float32)
        elif self.model_name == 'CVPR18':
            return [torch.tensor(encoder_pos_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(decoder_pos_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(decoder_sal_inputs_for_batch,dtype=torch.float32)], torch.tensor(decoder_outputs_for_batch,dtype=torch.float32)
        elif self.model_name == 'pos_only':
            return [torch.tensor(transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),dtype=torch.float32), 
                    torch.tensor(transform_batches_cartesian_to_normalized_eulerian(decoder_pos_inputs_for_batch),dtype=torch.float32)], torch.tensor(transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch),dtype=torch.float32)
        elif self.model_name in ['pos_only_3d_loss','DVMS']:
            return [torch.tensor(encoder_pos_inputs_for_batch,dtype=torch.float32), 
                    torch.tensor(decoder_pos_inputs_for_batch,dtype=torch.float32)], torch.tensor(decoder_outputs_for_batch,dtype=torch.float32)
        elif self.model_name == 'CVPR18_orig':
            return [torch.tensor(transform_batches_cartesian_to_normalized_eulerian(encoder_pos_inputs_for_batch),dtype=torch.float32), torch.tensor(decoder_sal_inputs_for_batch,dtype=torch.float32)[:, 0, :, :, 0]], torch.tensor(transform_batches_cartesian_to_normalized_eulerian(decoder_outputs_for_batch),dtype=torch.float32)[:, 0]
        elif self.model_name == 'MM18':
            return torch.tensor(encoder_sal_inputs_for_batch,dtype=torch.float32), torch.tensor(decoder_outputs_for_batch,dtype=torch.float32)
    
    
if __name__=="__main__":
    d1,d2,d3,d4=fan_nossdav_split()
    print(d1.shape)
    print(len(d2))
    print(len(d3))
    print(len(d4))
    #print(d2)