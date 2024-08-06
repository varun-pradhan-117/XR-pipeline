import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import sys
import argparse


from SampledDataset import read_sampled_positions_for_trace, load_saliency, load_true_saliency, get_video_ids, get_user_ids, get_users_per_video, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, partition_in_train_and_test_without_video_intersection, partition_in_train_and_test
from Utils import cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos,load_dict_from_csv,all_metrics, store_list_as_csv, MetricOrthLoss, OrthDist
from data_utils import fan_nossdav_split, PositionDataset
import TRACK_POS, TRACK_SAL

if torch.cuda.is_available():
    device=torch.device("cuda")
    print("Using GPU")
else:
    device=torch.device("cpu")
    print("Using CPU")

np.random.seed(19680801)
parser = argparse.ArgumentParser(description='Process the input parameters to train the network.')

parser.add_argument('-train', action="store_true", dest='train_flag', help='Flag that tells if we will run the training procedure.')
parser.add_argument('-evaluate', action="store_true", dest='evaluate_flag', help='Flag that tells if we will run the evaluation procedure.')
parser.add_argument('-dataset_name', action='store', dest='dataset_name', help='The name of the dataset used to train this network.')
parser.add_argument('-model_name', action='store', dest='model_name', help='The name of the model used to reference the network structure used.')
parser.add_argument('-init_window', action='store', dest='init_window', help='(Optional) Initial buffer window (to avoid stationary part).', type=int)
parser.add_argument('-m_window', action='store', dest='m_window', help='Past history window.', type=int)
parser.add_argument('-h_window', action='store', dest='h_window', help='Prediction window.', type=int)
parser.add_argument('-end_window', action='store', dest='end_window', help='(Optional) Final buffer (to avoid having samples with less outputs).', type=int)
parser.add_argument('-exp_folder', action='store', dest='exp_folder', help='Used when the dataset folder of the experiment is different than the default dataset.')
parser.add_argument('-provided_videos', action="store_true", dest='provided_videos', help='Flag that tells whether the list of videos is provided in a global variable.')
parser.add_argument('-use_true_saliency', action="store_true", dest='use_true_saliency', help='Flag that tells whether to use true saliency (if not set, then the content based saliency is used).')
parser.add_argument('-num_of_peaks', action="store", dest='num_of_peaks', help='Value used to get number of peaks from the true_saliency baseline.')
parser.add_argument('-video_test_chinacom', action="store", dest='video_test_chinacom', help='Which video will be used to test in ChinaCom, the rest of the videos will be used to train')
parser.add_argument('-metric', action="store", dest='metric', help='Which metric to use, by default, orthodromic distance is used.')

args = parser.parse_args()

# Parse arguments (or assign default)
if args.dataset_name is None:
    dataset_name="Fan_NOSSDAV_17"
else:
    dataset_name=args.dataset_name
    
if args.m_window is None:
    M_WINDOW=5
else:
    M_WINDOW=args.m_window

if args.h_window is None:
    H_WINDOW=25
else:
    H_WINDOW=args.h_window
    
if args.init_window is None:
    INIT_WINDOW=M_WINDOW
else:
    INIT_WINDOW=args.init_window

if args.end_window is None:
    END_WINDOW=H_WINDOW
else:
    END_WINDOW=args.end_window
    
if args.model_name is None:
    model_name="pos_only"
else:
    model_name=args.model_name

if args.metric is None:
    metric = all_metrics['orthodromic']
else:
    assert args.metric in all_metrics.keys()
    metric = all_metrics[args.metric]

# Fixed parameters
EPOCHS=500
NUM_TILES_WIDTH=480
NUM_TILES_HEIGHT=240
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256
RATE = 0.2
PERC_VIDEOS_TRAIN = 0.6
PERC_USERS_TRAIN = 0.6
BATCH_SIZE = 128
TRAIN_MODEL = False
EVALUATE_MODEL = False
if args.train_flag:
    TRAIN_MODEL = True
if args.evaluate_flag:
    EVALUATE_MODEL = True

root_dataset_folder = os.path.join('/media/Blue2TB1', dataset_name)
EXP_NAME=f"_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}"
SAMPLED_DATASET_FOLDER=os.path.join(root_dataset_folder,'sampled_dataset')

EXP_FOLDER = args.exp_folder
# If EXP_FOLDER is defined, add "Paper_exp" to the results name and use the folder in EXP_FOLDER as dataset folder
if EXP_FOLDER is None:
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, 'sampled_dataset')
else:
    EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)


SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'extract_saliency/saliency')

if model_name == 'MM18':
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'mm18_true_saliency')
else:
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'true_saliency')

# Define mdoel and results folder 
if model_name == 'TRACK':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
        model,optimizer,criterion=TRACK_SAL.create_sal_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,
                                                             NUM_TILES_HEIGHT=NUM_TILES_HEIGHT,
                                                             NUM_TILES_WIDTH=NUM_TILES_WIDTH, device=device)
if model_name == 'TRACK_AblatSal':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatSal/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
if model_name == 'TRACK_AblatFuse':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK_AblatFuse/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
elif model_name == 'CVPR18':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
elif model_name == 'pos_only':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only/Models_EncDec_eulerian' + EXP_NAME)
    model,optimizer,criterion=TRACK_POS.create_pos_only_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW)
elif model_name == 'pos_only_3d_loss':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_3d_loss/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_3d_loss/Models_EncDec_eulerian' + EXP_NAME)
elif model_name == 'CVPR18_orig':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Results_EncDec_2DNormalized_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Models_EncDec_2DNormalized_TrueSal' + EXP_NAME)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Results_Enc_2DNormalized_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'CVPR18_orig/Models_Enc_2DNormalized_ContSal' + EXP_NAME)
elif model_name == 'MM18':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'MM18/Results_Seq2One_2DNormalized_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'MM18/Models_Seq2One_2DNormalized_TrueSal' + EXP_NAME)


#print(model)
#summary(model,input_size=(128,1,5,2))
if __name__=='__main__':

    videos = get_video_ids(SAMPLED_DATASET_FOLDER)
    users = get_user_ids(SAMPLED_DATASET_FOLDER)
    users_per_video = get_users_per_video(SAMPLED_DATASET_FOLDER)

    if dataset_name=="Fan_NOSSDAV_17":
        split_path=os.path.join(dataset_name,"splits")
        if os.path.exists(os.path.join(split_path,'train_set')):
            train_traces=load_dict_from_csv(os.path.join(split_path,'train_set'),columns=['user','video'])
            test_traces=load_dict_from_csv(os.path.join(split_path,'test_set'),columns=['user','video'])
            user_test_traces=load_dict_from_csv(os.path.join(split_path,'user_test_set'),columns=['user','video'])
            video_test_traces=load_dict_from_csv(os.path.join(split_path,'video_test_set'),columns=['user','video'])
        else:
            train_traces,test_traces,video_test_traces,user_test_traces=fan_nossdav_split(video_ratio=PERC_VIDEOS_TRAIN,user_ratio=PERC_USERS_TRAIN)
            store_list_as_csv(os.path.join(split_path,'train_set'),['user','video'],train_traces)
            store_list_as_csv(os.path.join(split_path,'test_set'),['user','video'],test_traces)
            store_list_as_csv(os.path.join(split_path,'user_test_set'),['user','video'],user_test_traces)
            store_list_as_csv(os.path.join(split_path,'video_test_set'),['user','video'],video_test_traces)
        partitions=partition_in_train_and_test(SAMPLED_DATASET_FOLDER,INIT_WINDOW,END_WINDOW,train_traces,test_traces,user_test_traces=user_test_traces,video_test_traces=video_test_traces)

    #print(train_traces.shape)
    #print(test_traces.shape)
    #print(partitions)

    # Dictionary that stores the traces per video and user
    all_traces = {}
    for video in videos:
        all_traces[video] = {}
        for user in users_per_video[video]:
            all_traces[video][user] = read_sampled_positions_for_trace(SAMPLED_DATASET_FOLDER, str(video), str(user))

    #print(users_per_video[videos[0]])        
    #print(all_traces[videos[0]][users_per_video[videos[0]][0]])
    #print(all_traces['coaster']['user21'].shape)
    #sys.exit()


    all_saliencies = {}
    if model_name not in ['pos_only', 'pos_only_3d_loss', 'no_motion', 'true_saliency', 'content_based_saliency']:
        for video in videos:
            print(f"Loading {video} saliencies")
            all_saliencies[video]=load_saliency(SALIENCY_FOLDER,video)


    train_data=PositionDataset(partitions['train'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies)
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True, pin_memory=True, num_workers=0)
    test_data=PositionDataset(partitions['test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies)
    test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)

    if model_name == 'pos_only':
        criterion = OrthDist()
    elif model_name == 'TRACK':
        metrics = {"orth_dist": MetricOrthLoss}

    EPOCHS=500
    model_save_path=os.path.join('SavedModels',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
    print(model_save_path)
    #torch.autograd.set_detect_anomaly(True)
    if model_name=='pos_only':
        losses, val_losses=TRACK_POS.train_pos_only(model,train_loader,test_loader,optimizer,criterion,epochs=EPOCHS, device=device,path=model_save_path)
        #print(model.state_dict())
        print(f"Final Loss:{losses[-1]:.4f}")
        if not os.path.exists(os.path.join('Losses',dataset_name)):
            os.makedirs(os.path.join('LOsses',dataset_name))
        torch.save(losses,os.path.join('Losses',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}"))
    elif model_name=='TRACK':
        losses,val_losses=TRACK_SAL.train_model(model,train_loader,test_loader,optimizer,criterion,epochs=EPOCHS,device=device,path=model_save_path)
        print(f"Final Loss:{losses[-1]:.4f}")
        if not os.path.exists(os.path.join('Losses',dataset_name)):
            os.makedirs(os.path.join('LOsses',dataset_name))
        torch.save(losses,os.path.join('Losses',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}"))