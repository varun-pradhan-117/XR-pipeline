import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchinfo import summary
import os
import sys
import csv
import random
import argparse

from DatasetHelper import partition_in_train_and_test, get_video_ids, get_user_ids, get_users_per_video, read_sampled_unit_vectors_by_users, load_true_saliency,load_saliency
#from SampledDataset import read_sampled_positions_for_trace, split_list_by_percentage, partition_in_train_and_test_without_any_intersection, partition_in_train_and_test_without_video_intersection
from Utils import get_orthodromic_distance_cartesian,get_orthodromic_distance_euler,cartesian_to_eulerian, eulerian_to_cartesian, get_max_sal_pos,load_dict_from_csv,all_metrics, store_list_as_csv, MetricOrthLoss, OrthDist
from data_utils import fan_nossdav_split, PositionDataset, split_data_all_users, save_unique_videos_to_csv, fetch_entropies
from trainers import train_model, test_model, test_full_vid
import TRACK_POS, TRACK_SAL, DVMS, VPT360, adaptiveMultiHead, AdaptiveLSTM



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
parser.add_argument('-evaluate_old_vid', action="store_true", dest='evaluate_old_vid_flag', help='Flag that tells if we will run the evaluation procedure for previously seen videos.')
parser.add_argument('-evaluate_old_user', action="store_true", dest='evaluate_old_user_flag', help='Flag that tells if we will run the evaluation procedure for previously seen user.')
parser.add_argument('-evaluate_vid', action="store_true", dest='evaluate_vid_flag', help='Flag that tells if we will run the evaluation procedure per video.')
parser.add_argument('-evaluate_full_vid', action="store_true", dest='evaluate_full_vid_flag', help='Flag that tells if we will run the evaluation procedure over the entire video.')
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
parser.add_argument('-K', action="store", dest='K',
                    help='(Optional) Number of predicted trajectories (default to 2).')
parser.add_argument('-a', action="store", dest='alpha',
                    help='(Optional) Weight for IE in adjusted loss.')
parser.add_argument('-mode', action="store", dest='mode',
                    help='(Optional) SE or AE.')



args = parser.parse_args()
# Parse arguments (or assign default)
if args.dataset_name is None:
    dataset_name="Fan_NOSSDAV_17"
else:
    dataset_name=args.dataset_name

if args.alpha is not None:
    alpha=float(args.alpha)
else:
    alpha=0
    
if args.mode is not None:
    mode='SE'
else:
    mode='IE'

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
if args.K is not None:
    K = int(args.K)
else:
    K=2


# Fixed parameters
EPOCHS=500
NUM_TILES_WIDTH=480
NUM_TILES_HEIGHT=240
NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256
RATE = 0.2
PERC_VIDEOS_TEST = 0.4
PERC_USERS_TEST = 0.4
BATCH_SIZE = 128

TRAIN_MODEL = False
EVALUATE_MODEL,EVALUATE_OLD_VIDEOS,EVALUATE_VIDEOS, EVALUATE_OLD_USERS, EVALUATE_FULL_VIDEOS = False,False,False,False,False
if args.train_flag:
    TRAIN_MODEL = True
if args.evaluate_flag:
    EVALUATE_MODEL = True
if args.evaluate_vid_flag:
    EVALUATE_VIDEOS=True
if args.evaluate_old_vid_flag:
    EVALUATE_OLD_VIDEOS=True    
if args.evaluate_old_user_flag:
    EVALUATE_OLD_USERS=True
if args.evaluate_full_vid_flag:
    EVALUATE_FULL_VIDEOS=True
root_folder='/media/Blue2TB1'
root_dataset_folder = os.path.join('/media/Blue2TB1', dataset_name)
EXP_NAME=f"_init_{INIT_WINDOW}_in_{M_WINDOW}_out_{H_WINDOW}_end_{END_WINDOW}"
SAMPLED_DATASET_FOLDER=os.path.join(root_dataset_folder,'sampled_dataset')
VIDEO_DATA_FOLDER=os.path.join(root_dataset_folder,'video_data')

EXP_FOLDER = args.exp_folder
# If EXP_FOLDER is defined, add "Paper_exp" to the results name and use the folder in EXP_FOLDER as dataset folder
if EXP_FOLDER is None:
    EXP_NAME = '_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    if model_name=='DVMS':
        EXP_NAME=f'{EXP_NAME}_K{K}'
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, 'sampled_dataset')
else:
    EXP_NAME = '_Paper_Exp_init_' + str(INIT_WINDOW) + '_in_' + str(M_WINDOW) + '_out_' + str(H_WINDOW) + '_end_' + str(END_WINDOW)
    SAMPLED_DATASET_FOLDER = os.path.join(root_dataset_folder, EXP_FOLDER)


SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'extract_saliency/saliency')

if model_name == 'MM18':
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'mm18_true_saliency')
else:
    TRUE_SALIENCY_FOLDER = os.path.join(root_dataset_folder, 'true_saliency')
eval_metrics={}
# Define mdoel and results folder 
if model_name == 'TRACK':
    if args.use_true_saliency:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_TrueSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_TrueSal' + EXP_NAME)
        model,optimizer,criterion=TRACK_SAL.create_sal_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,
                                                             NUM_TILES_HEIGHT=NUM_TILES_HEIGHT_TRUE_SAL,
                                                             NUM_TILES_WIDTH=NUM_TILES_WIDTH_TRUE_SAL, device=device)
    else:
        RESULTS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Results_EncDec_3DCoords_ContSal' + EXP_NAME)
        MODELS_FOLDER = os.path.join(root_dataset_folder, 'TRACK/Models_EncDec_3DCoords_ContSal' + EXP_NAME)
        model,optimizer,criterion=TRACK_SAL.create_sal_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,
                                                             NUM_TILES_HEIGHT=NUM_TILES_HEIGHT,
                                                             NUM_TILES_WIDTH=NUM_TILES_WIDTH, device=device)
    eval_metrics['orth_dist']=get_orthodromic_distance_cartesian
if model_name=='DVMS':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, f'DVMS/Results_K{K}_EncDec_3DCoord' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, f'DVMS/Models_K{K}_EncDec_3DCoord' + EXP_NAME)
    model,optimizer,criterion=DVMS.create_DVMS_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,
                                                     K=K,device=device)
    eval_metrics['orth_dist']=DVMS.flat_top_k_orth_dist
if model_name == 'VPT360':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, f'VPT360/Results_K{K}_EncDec_3DCoord' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, f'VPT360/Models_K{K}_EncDec_3DCoord' + EXP_NAME)
    model,optimizer,criterion=VPT360.create_VPT360_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device)
    eval_metrics['orth_dist']=MetricOrthLoss
    eval_metrics['combinatorial_loss']=criterion
    
if model_name == "AMH":
    RESULTS_FOLDER = os.path.join(root_dataset_folder, f'AMH/Results_K{K}_EncDec_3DCoord' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, f'AMH/Models_K{K}_EncDec_3DCoord' + EXP_NAME)
    model,optimizer,criterion=adaptiveMultiHead.create_AMH_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device, full_vec=True, mode=mode)
    eval_metrics['orth_dist']=MetricOrthLoss
    eval_metrics['combinatorial_loss']=criterion
if model_name == "ALSTM":
    RESULTS_FOLDER = os.path.join(root_dataset_folder, f'ALSTM/Results_K{K}_EncDec_3DCoord' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, f'ALSTM/Models_K{K}_EncDec_3DCoord' + EXP_NAME)
    model,optimizer,criterion=AdaptiveLSTM.create_ALSTM_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device, mode=mode)
    eval_metrics['orth_dist']=MetricOrthLoss
if model_name == "ALSTM-E":
    RESULTS_FOLDER = os.path.join(root_dataset_folder, f'ALSTM/Results_K{K}_EncDec_3DCoord' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, f'ALSTM/Models_K{K}_EncDec_3DCoord' + EXP_NAME)
    model,optimizer,criterion=AdaptiveLSTM.create_ALSTM_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device, entropy=False)
    eval_metrics['orth_dist']=MetricOrthLoss
    #eval_metrics['combinatorial_loss']=criterion
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
    model,optimizer,criterion=TRACK_POS.create_pos_only_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device,)
    eval_metrics['orth_dist']=get_orthodromic_distance_euler
elif model_name == 'pos_only_weighted_loss':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_weighted/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_weighted/Models_EncDec_eulerian' + EXP_NAME)
    model,optimizer,criterion=TRACK_POS.create_pos_only_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device, loss="IE")
    eval_metrics['orth_dist']=get_orthodromic_distance_euler
elif model_name == 'pos_only_augmented':
    RESULTS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_augment/Results_EncDec_eulerian' + EXP_NAME)
    MODELS_FOLDER = os.path.join(root_dataset_folder, 'pos_only_augment/Models_EncDec_eulerian' + EXP_NAME)
    model,optimizer,criterion=TRACK_POS.create_pos_only_augmented_model(M_WINDOW=M_WINDOW,H_WINDOW=H_WINDOW,device=device, input_size=3,mode=mode)
    eval_metrics['orth_dist']=get_orthodromic_distance_euler
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

    videos = get_video_ids(VIDEO_DATA_FOLDER)
    users = get_user_ids(VIDEO_DATA_FOLDER)
    users_per_video = get_users_per_video(VIDEO_DATA_FOLDER)

    if dataset_name in ["Fan_NOSSDAV_17","Jin_22", 'PAMI18','MMSys18']:
        split_path=os.path.join(dataset_name,"splits")
        if os.path.exists(os.path.join(split_path,'train_set')):
            train_traces=load_dict_from_csv(os.path.join(split_path,'train_set'),columns=['user','video'])
            test_traces=load_dict_from_csv(os.path.join(split_path,'test_set'),columns=['user','video'])
            user_test_traces=load_dict_from_csv(os.path.join(split_path,'user_test_set'),columns=['user','video'])
            video_test_traces=load_dict_from_csv(os.path.join(split_path,'video_test_set'),columns=['user','video'])
            test_vids=load_dict_from_csv(os.path.join(split_path,'test_vids'),columns=['video'])
        else:
            train_traces,test_traces,video_test_traces,user_test_traces, train_vids, test_vids=split_data_all_users(root_dataset_folder,
                                                                                                                    total_users=users,
                                                                                                                    users_per_video=users_per_video,
                                                                                                                    bins=2,
                                                                                                                    video_test_size=PERC_VIDEOS_TEST,
                                                                                                                    user_test_size=PERC_USERS_TEST)
            #os.makedirs(os.path.join(split_path))
            store_list_as_csv(os.path.join(split_path,'train_set'),['user','video'],train_traces)
            store_list_as_csv(os.path.join(split_path,'test_set'),['user','video'],test_traces)
            store_list_as_csv(os.path.join(split_path,'user_test_set'),['user','video'],user_test_traces)
            store_list_as_csv(os.path.join(split_path,'video_test_set'),['user','video'],video_test_traces)
            with open(os.path.join(split_path,'test_vids'), mode='w', newline='') as file:
                writer = csv.writer(file)
                
                # Write the header
                writer.writerow(['video'])
                
                # Write each item in the data list as a row
                for item in test_vids:
                    writer.writerow([item])
        partitions=partition_in_train_and_test(VIDEO_DATA_FOLDER,
                                               init_window=INIT_WINDOW,
                                               end_window=END_WINDOW,
                                               train_traces=train_traces,
                                               test_traces=test_traces,
                                               user_test_traces=user_test_traces,
                                               video_test_traces=video_test_traces)

    #print(train_traces.shape)
    #print(test_traces.shape)
    #print(partitions)
    # Dictionary that stores the traces per video and user
    all_traces = {}
    for video in videos:
        all_traces[video] = read_sampled_unit_vectors_by_users(VIDEO_DATA_FOLDER, video,users_per_video[video])
    #print(users_per_video[videos[0]])        
    #print(all_traces[videos[0]][users_per_video[videos[0]][0]])
    #print(all_traces['coaster']['user21'].shape)
    #sys.exit()


    all_saliencies = {}
    if model_name not in ['pos_only', 'pos_only_3d_loss', 'no_motion', 'true_saliency', 'content_based_saliency','DVMS','VPT360','AMH','pos_only_augmented',
                          'ALSTM','ALSTM-E','pos_only_weighted_loss']:
        if args.use_true_saliency:
            for video in videos:
                print(f"Loading {video} saliencies")
                all_saliencies[video]=load_true_saliency(TRUE_SALIENCY_FOLDER,video)
        else:
            for video in videos:
                print(f"Loading {video} saliencies")
                all_saliencies[video]=load_saliency(SALIENCY_FOLDER,video)
    IEs={}
    SEs={}
    if model_name in  ['AMH','pos_only_augmented','ALSTM','ALSTM-E','pos_only_weighted_loss']:
        SEs,_,IEs=fetch_entropies(root_folder,dataset_name)
    print(SEs['sport'].shape)
    print(IEs['sport'])
    exit()
    
    

    if model_name in ['TRACK', 'DVMS']:
        metrics = {"orth_dist": MetricOrthLoss}
    if model_name in ['VPT360','AMH','ALSTM','ALSTM-SE','AMH-SE']:
        metrics= eval_metrics
    else:
        metrics=None
    EPOCHS=500
    if mode=='SE':
        model_name=model_name+'-SE'
    if H_WINDOW==25:
        model_save_path=os.path.join('SavedModels',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
    else:
        model_save_path=os.path.join('SavedModels',f'{H_WINDOW/5}sec',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
    print(model_save_path)
    #torch.autograd.set_detect_anomaly(True)
    if TRAIN_MODEL:
        train_data=PositionDataset(partitions['train'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                   all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies,
                                   all_IEs=IEs, all_SEs=SEs)
        train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True, pin_memory=True, num_workers=4)
        test_data=PositionDataset(partitions['test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                  all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies,
                                  all_IEs=IEs, all_SEs=SEs)
        test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=4)
        if model_name in ['pos_only','DVMS','TRACK','VPT360','AMH', 'pos_only_augmented','ALSTM','ALSTM-E',
                          'pos_only_weighted_loss']:
            losses,val_losses=train_model(model=model,train_loader=train_loader,
                                          validation_loader=test_loader,
                                          optimizer=optimizer,criterion=criterion, metric=metrics,epochs=EPOCHS,
                                        device=device,path=model_save_path, tolerance=20, model_name=model_name,alpha=alpha)
            print(f"Final Loss:{losses[-1]:.4f}")
            if not os.path.exists(os.path.join('Losses',dataset_name)):
                os.makedirs(os.path.join('Losses',dataset_name))
            torch.save(losses,os.path.join('Losses',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}"))
    eval_metrics={}
    if model_name in ['pos_only','pos_only_augmented','pos_only_weighted_loss','pos_only_augmented-SE']:
        eval_metrics['orth_dist']=get_orthodromic_distance_euler
    elif model_name == 'DVMS':
        eval_metrics['orth_dist']=DVMS.flat_top_k_orth_dist
    else:
        eval_metrics['orth_dist']=get_orthodromic_distance_cartesian
    if EVALUATE_MODEL:
        test_data=PositionDataset(partitions['test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                  all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies,
                                  all_IEs=IEs, all_SEs=SEs)
        test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)
        saved_models=os.listdir(model_save_path)
        epoch_files=[f for f in saved_models if f.endswith('.pth')]
        if not epoch_files:
            raise FileNotFoundError("No saved models for given model name and dataset.")
        epoch_numbers=[int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest=max(epoch_numbers)
        best_model=os.path.join(model_save_path,f'Epoch_{latest}.pth')
        model_data=torch.load(best_model,weights_only=False)
        model.load_state_dict(model_data['model_state_dict'])
        plot_path=os.path.join("TestPlots",dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
        test_model(model=model,validation_loader=test_loader,criterion=criterion,device=device,
                   metric=eval_metrics,path=plot_path, model_name=model_name, K=K)
    if EVALUATE_OLD_VIDEOS:
        test_data=PositionDataset(partitions['user_test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                  all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies,
                                  all_IEs=IEs, all_SEs=SEs)
        test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)
        saved_models=os.listdir(model_save_path)
        epoch_files=[f for f in saved_models if f.endswith('.pth')]
        if not epoch_files:
            raise FileNotFoundError("No saved models for given model name and dataset.")
        epoch_numbers=[int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest=max(epoch_numbers)
        best_model=os.path.join(model_save_path,f'Epoch_{latest}.pth')
        model_data=torch.load(best_model,weights_only=False)
        model.load_state_dict(model_data['model_state_dict'])
        plot_path=os.path.join("New_User_TestPlots",dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
        test_model(model=model,validation_loader=test_loader,criterion=criterion,device=device,
                   metric=eval_metrics,path=plot_path, model_name=model_name, K=K)
    
    if EVALUATE_OLD_USERS:
        test_data=PositionDataset(partitions['video_test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                  all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies,
                                  all_IEs=IEs, all_SEs=SEs)
        test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)
        saved_models=os.listdir(model_save_path)
        epoch_files=[f for f in saved_models if f.endswith('.pth')]
        if not epoch_files:
            raise FileNotFoundError("No saved models for given model name and dataset.")
        epoch_numbers=[int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest=max(epoch_numbers)
        best_model=os.path.join(model_save_path,f'Epoch_{latest}.pth')
        model_data=torch.load(best_model,weights_only=False)
        model.load_state_dict(model_data['model_state_dict'])
        plot_path=os.path.join("New_Video_TestPlots",dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
        test_model(model=model,validation_loader=test_loader,criterion=criterion,device=device,
                   metric=eval_metrics,path=plot_path, model_name=model_name, K=K)
        
    if EVALUATE_VIDEOS:
        saved_models=os.listdir(model_save_path)
        epoch_files=[f for f in saved_models if f.endswith('.pth')]
        if not epoch_files:
            raise FileNotFoundError("No saved models for given model name and dataset.")
        epoch_numbers=[int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest=max(epoch_numbers)
        best_model=os.path.join(model_save_path,f'Epoch_{latest}.pth')
        model_data=torch.load(best_model,weights_only=False)
        model.load_state_dict(model_data['model_state_dict'])
        test_vids=[vid[0] for vid in test_vids]
        print(*test_vids)
        plot_path=os.path.join("Test_vid_plots",dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
        for video in test_vids:
            test_data=PositionDataset(partitions['test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                      all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies, video_name=video,
                                      all_IEs=IEs, all_SEs=SEs)
            test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)
            test_model(model=model,validation_loader=test_loader,criterion=criterion,device=device,
                       metric=eval_metrics,path=plot_path, model_name=model_name, K=K, vid_name=video)
        
    if EVALUATE_FULL_VIDEOS:
        saved_models=os.listdir(model_save_path)
        epoch_files=[f for f in saved_models if f.endswith('.pth')]
        if not epoch_files:
            raise FileNotFoundError("No saved models for given model name and dataset.")
        epoch_numbers=[int(f.split('_')[1].split('.')[0]) for f in epoch_files]
        latest=max(epoch_numbers)
        best_model=os.path.join(model_save_path,f'Epoch_{latest}.pth')
        model_data=torch.load(best_model,weights_only=False)
        model.load_state_dict(model_data['model_state_dict'])
        test_vids=[vid[0] for vid in test_vids]
        print(*test_vids)
        plot_path=os.path.join("Plots",'Full_video_plots',dataset_name,f"{model_name}_{EXP_NAME}_Epoch{EPOCHS}")
        for video in test_vids:
            user_list=[trace[0] for trace in test_traces if trace[1]==video]
            for user in user_list:
                test_data=PositionDataset(partitions['test'],future_window=H_WINDOW,M_WINDOW=M_WINDOW,
                                      all_traces=all_traces,model_name=model_name,all_saliencies=all_saliencies, video_name=video, user_name=user,
                                      all_IEs=IEs, all_SEs=SEs)
                test_loader=DataLoader(test_data,batch_size=BATCH_SIZE,shuffle=False, pin_memory=True,num_workers=0)   
                test_full_vid(model=model,validation_loader=test_loader,criterion=criterion,device=device,
                        metric=eval_metrics,path=plot_path, model_name=model_name, K=K, vid_name=video, user_name=user)
            print(f"Saved files for {video} with {len(user_list)} users.")

            