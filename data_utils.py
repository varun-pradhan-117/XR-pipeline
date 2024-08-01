import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos,load_dict_from_csv,all_metrics


NOSSDAV_sample_folder="Fan_NOSSDAV_17/sampled_dataset"
def fan_nossdav_split(user_ratio=0.6, video_ratio=0.6):
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


class PositionDataset(Dataset):
    def __init__(self,list_IDs,future_window,M_WINDOW, model_name, all_traces,all_saliencies=None,all_headmaps=None):
        self.list_IDs=list_IDs
        self.all_saliencies=all_saliencies
        self.all_traces=all_traces
        self.all_headmaps=all_headmaps
        self.M_WINDOW=M_WINDOW
        self.model_name=model_name
        self.future_window=future_window
        
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
        
        if self.model_name not in ['pos_only', 'pos_only_3d_loss', 'MM18']:
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
        elif self.model_name == 'pos_only_3d_loss':
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