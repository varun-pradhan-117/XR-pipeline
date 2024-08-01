import os
import sys
sys.path.insert(0, './')

import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import ast
import argparse
import shutil

from Utils import eulerian_to_cartesian, cartesian_to_eulerian, rotationBetweenVectors, interpolate_quaternions, degrees_to_radian, radian_to_degrees

dataset_folder='D:/Jin_22'
trajec_dir=os.path.join(dataset_folder,'Full_Dataset')
users_folder=os.path.join(dataset_folder,"Version2")
video_img_folder=os.path.join(dataset_folder,'5fps_Video_Images')
video_path=os.path.join(dataset_folder,'Videos')
agg_folder=os.path.join(dataset_folder,'Trajectory_info')


# Resample original videos
def store_frames(video_path,output_folder,fps=None,rescale=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    #print(f"Saving video {os.path.basename(video_path).split('.')[0]}")
    video=cv2.VideoCapture(os.path.join(video_path))
    original_frame_rate=round(video.get(cv2.CAP_PROP_FPS))
    print(original_frame_rate)
    is_div= original_frame_rate%fps==0
    original_frame_width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_duration=1/original_frame_rate
    if fps:
        frame_interval=original_frame_rate//fps
    else:
        frame_interval=1
    frame_count=0
    last_frame=int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    timestamps=[]
    frames=[]
    while video.isOpened():
        ret,frame=video.read()
        if not ret:
            break
        if is_div:
            if frame_count%frame_interval==0 or frame_count+1==last_frame:
                frame=cv2.resize(frame,(original_frame_width//rescale,original_frame_height//rescale))
                frames.append(frame)
                timestamps.append(frame_count*frame_duration)
        else:
            frame=cv2.resize(frame,(original_frame_width//rescale,original_frame_height//rescale))
            frames.append(frame)
            timestamps.append(frame_count*frame_duration)
        frame_count+=1
    if not is_div:
        sampled_frames=[]
        sampled_timestamps=[]
        desired_timestamps=list(np.arange(0,round(timestamps[-1])+0.1,1/fps))
        for ts in desired_timestamps:
            closest_idx=np.abs(timestamps-ts).argmin()
            sampled_frames.append(frames[closest_idx])
            sampled_timestamps.append(timestamps[closest_idx])
        timestamps=sampled_timestamps
    video.release()
    frames=np.array(frames)
    timestamps=np.array(timestamps)
    np.save(f'{output_folder}/sampled_video.npy',frames)
    np.save(f'{output_folder}/timestamps.npy',timestamps)

# Rearrange folder to match standard file structure of Video_Folder -> User_trajectories
def rearrange_folders(base_dir,dest_dir):
    user_folders=sorted(os.listdir(base_dir),key=lambda x:int(x.split(' ')[1][1:-1]))
    for folder in user_folders:
        user_num=folder.split(' ')[1][1:-1]
        user_folder_path=os.path.join(base_dir,folder)
        trajectory_files=os.listdir(os.path.join(user_folder_path))
        for trajectory_file in trajectory_files:
            vid_num=trajectory_file.split('_')[2]
            new_name=f'user-{user_num}_vid-{vid_num}_trajectory.csv'
            vid_folder_path=os.path.join(dest_dir,f'video_{vid_num}')
            if not os.path.exists(vid_folder_path):
                os.makedirs(vid_folder_path)
            
            src_path=os.path.join(user_folder_path,trajectory_file)
            dst_path=os.path.join(vid_folder_path,new_name)
            shutil.copy2(src_path,dst_path)


# Interpolation functions for resampling
def interpolate_position(timestamps, positions, new_timestamps):
    interp_fun = interp1d(
        timestamps, positions, axis=0, kind="linear", fill_value="extrapolate"
    )
    return interp_fun(new_timestamps)


def interpolate_quat(timestamps, quaternions, new_timestamps):
    slerp = R.from_quat(quaternions)
    key_rots = slerp.as_quat()
    interp_fun = interp1d(
        timestamps, key_rots, axis=0, kind="linear", fill_value="extrapolate"
    )
    interpolated_rots = interp_fun(new_timestamps)
    return R.from_quat(interpolated_rots).as_quat()


def interpolate_unit_vector(timestamps, vectors, new_timestamps):
    interp_func = interp1d(
        timestamps, vectors, axis=0, kind="linear", fill_value="extrapolate"
    )
    interpolated_vectors = interp_func(new_timestamps)
    norm = np.linalg.norm(interpolated_vectors, axis=1, keepdims=True)
    normalized_vectors = interpolated_vectors / norm
    return normalized_vectors

# Get timestamps for trajectory resampling
def load_timestamps(video_folder):
    video_timestamps = {}
    for root, dirs, files in os.walk(video_folder):
        for dir in dirs:
            video_number = int(dir.split("_")[1])
            timestamps_path = os.path.join(root, dir, "timestamps.npy")
            if os.path.exists(timestamps_path):
                video_timestamps[video_number] = np.load(timestamps_path)
    return video_timestamps

# Get video number from file name
def get_video_number(file_name):
    vid_number = int(file_name.split("_")[2])
    return vid_number


def process_csv(file_path, new_timestamps):
    # print(file_path)
    data = pd.read_csv(file_path)
    time = data["AdjustedTime"]
    # print(time[0])
    time = time - time[0]
    # print(time[0])
    pose_position = np.array(data["Pose_Position"].apply(ast.literal_eval).tolist())
    pose_rotation = np.array(data["Pose_Rotation"].apply(ast.literal_eval).tolist())
    unit_vector = np.array(data["Unit_Vector"].apply(ast.literal_eval).tolist())
    # print(pose_position)
    # print(pose_rotation[23,:],pose_rotation[24,:],pose_rotation[25,:])
    # print(len(time))
    # print(len(pose_rotation))
    interpolated_position = interpolate_position(time, pose_position, new_timestamps)
    interpolated_rotation = interpolate_quat(time, pose_rotation, new_timestamps)
    interpolated_unit_vector = interpolate_unit_vector(
        time, unit_vector, new_timestamps
    )
    # print(interpolated_position.tolist())
    # print(interpolated_position[1,:])
    """ sampled_data = pd.DataFrame({
        'Timestamp': new_timestamps,
        'Pose_Position': interpolated_position.tolist(),
        'Pose_Rotation': interpolated_rotation.tolist(),
        'Unit_Vector': interpolated_unit_vector.tolist(),
    }) """

    return interpolated_position, interpolated_rotation, interpolated_unit_vector


def save_numpy_arrays(video_output_dir, vid_name, user_data):
    timestamps, positions, rotations, vectors = user_data
    np.save(os.path.join(video_output_dir, f"{vid_name}_timestamps.npy"), timestamps)
    np.save(os.path.join(video_output_dir, f"{vid_name}_positions.npy"), positions)
    np.save(os.path.join(video_output_dir, f"{vid_name}_rotations.npy"), rotations)
    np.save(os.path.join(video_output_dir, f"{vid_name}_unit_vectors.npy"), vectors)

def resample_trajectories(src_dir,dst_dir,video_img_folder):
    tstamps=load_timestamps(video_img_folder)
    videos=os.listdir(src_dir)
    for vid in videos:
        vid_path=os.path.join(src_dir,vid)
        vid_num=vid.split("_")[1]
        timestamps=tstamps[int(vid_num)]
        output_dir=os.path.join(dst_dir,vid)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        csv_files=[f for f in os.listdir(vid_path) if f.endswith('.csv')]
        for f in csv_files:
            traj_path=os.path.join(vid_path,f)
            pos,rots,unit_vecs=process_csv(traj_path,timestamps)
            sampled_data=pd.DataFrame({
                'Timestamp':timestamps,
                'Pose_Position': pos.tolist(),
                'Pose_Rotation': rots.tolist(),
                'Unit_Vector':unit_vecs.tolist()
            })
            dst_path=os.path.join(dst_dir,vid,f)
            sampled_data.to_csv(dst_path,index=False)
            
def get_user_num(filename):
    num=filename.split('_')[0].split('-')[1]
    return int(num)

def store_trajectory_aggregates(src_dir,dst_dir):
    videos=os.listdir(src_dir)
    print(len(videos))
    for video in videos:
        trajs=os.listdir(os.path.join(src_dir,video))
        print(f'Processing {video}')
        op_dir=os.path.join(dst_dir,video)
        if not os.path.exists(op_dir):
            os.makedirs(op_dir)
        poses=[]
        rots=[]
        uvs=[]
        trajs=sorted(trajs,key=get_user_num)
        for traj in trajs:
            path=os.path.join(src_dir,video,traj)
            df=pd.read_csv(path)
            tstamps=np.array(df['Timestamp'])
            pos=np.array(df['Pose_Position'].apply(ast.literal_eval).tolist())
            rot=np.array(df['Pose_Rotation'].apply(ast.literal_eval).tolist())
            uv=np.array(df['Unit_Vector'].apply(ast.literal_eval).tolist())
            poses.append(pos)
            rots.append(rot)
            uvs.append(uv)
        uvs=np.array(uvs)
        poses=np.array(poses)
        rots=np.array(rots)

        save_numpy_arrays(op_dir,video,(tstamps,poses,rots,uvs)) 


NUM_TILES_WIDTH_TRUE_SAL = 256
NUM_TILES_HEIGHT_TRUE_SAL = 256
OUTPUT_TRUE_SALIENCY_FOLDER = os.path.join(dataset_folder,'true_saliency')

def get_video_ids(folder):
    list_of_videos = [o for o in os.listdir(folder) if not o.endswith('.gitkeep')]
    return list_of_videos

def get_sampled_unit_vectors():
    pass

def create_and_store_true_saliency(sampled_dataset):
    if not os.path.exists(OUTPUT_TRUE_SALIENCY_FOLDER):
        os.makedirs(OUTPUT_TRUE_SALIENCY_FOLDER)

    # Returns an array of size (NUM_TILES_HEIGHT_TRUE_SAL, NUM_TILES_WIDTH_TRUE_SAL) with values between 0 and 1 specifying the probability that a tile is watched by the user
    # We built this function to ensure the model and the groundtruth tile-probabilities are built with the same (or similar) function
    def from_position_to_tile_probability_cartesian(pos):
        yaw_grid, pitch_grid = np.meshgrid(np.linspace(0, 1, NUM_TILES_WIDTH_TRUE_SAL, endpoint=False),
                                           np.linspace(0, 1, NUM_TILES_HEIGHT_TRUE_SAL, endpoint=False))
        yaw_grid += 1.0 / (2.0 * NUM_TILES_WIDTH_TRUE_SAL)
        pitch_grid += 1.0 / (2.0 * NUM_TILES_HEIGHT_TRUE_SAL)
        yaw_grid = yaw_grid * 2 * np.pi
        pitch_grid = pitch_grid * np.pi
        x_grid, y_grid, z_grid = eulerian_to_cartesian(theta=yaw_grid, phi=pitch_grid)
        great_circle_distance = np.arccos(np.maximum(np.minimum(x_grid * pos[0] + y_grid * pos[1] + z_grid * pos[2], 1.0), -1.0))
        gaussian_orth = np.exp((-1.0 / (2.0 * np.square(0.1))) * np.square(great_circle_distance))
        return gaussian_orth

    videos = get_video_ids(OUTPUT_FOLDER)
    print()
    for enum_video, video in enumerate(videos):
        print('creating true saliency for video', video, '-', enum_video, '/', len(videos))
        real_saliency_for_video = []
        video_data=np.load(os.path.join(sampled_dataset),video,f'{video}_unit_vectors.npy')
        print(video_data.shape)
        return
        max_num_samples = get_max_num_samples_for_video(video, sampled_dataset)

        for x_i in range(max_num_samples):
            tileprobs_for_video_cartesian = []
            for user in sampled_dataset.keys():
                tileprobs_cartesian = from_position_to_tile_probability_cartesian(sampled_dataset[user][video][x_i, 1:])
                tileprobs_for_video_cartesian.append(tileprobs_cartesian)
            tileprobs_for_video_cartesian = np.array(tileprobs_for_video_cartesian)
            real_saliency_cartesian = np.sum(tileprobs_for_video_cartesian, axis=0) / tileprobs_for_video_cartesian.shape[0]
            real_saliency_for_video.append(real_saliency_cartesian)
        real_saliency_for_video = np.array(real_saliency_for_video)

        true_sal_out_file = os.path.join(OUTPUT_TRUE_SALIENCY_FOLDER, video)
        np.save(true_sal_out_file, real_saliency_for_video)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Process the input parameters to parse the dataset.')
    parser.add_argument('-creat_samp_dat', action="store_true", dest='_create_sampled_dataset', help='Flag that tells if we want to create and store the sampled dataset from the gaze positions.')
    parser.add_argument('-creat_samp_dat_head', action="store_true", dest='_create_sampled_dataset_head', help='Flag that tells if we want to create and store the sampled dataset from the gaze positions.')
    parser.add_argument('-creat_true_sal', action="store_true", dest='_create_true_saliency', help='Flag that tells if we want to create and store the ground truth saliency.')
    parser.add_argument('-creat_cb_sal', action="store_true", dest='_create_cb_saliency', help='Flag that tells if we want to create the content-based saliency maps.')
    parser.add_argument('-compare_traces', action="store_true", dest='_compare_traces', help='Flag that tells if we want to compare the original traces with the sampled traces.')
    parser.add_argument('-plot_3d_traces', action="store_true", dest='_plot_3d_traces', help='Flag that tells if we want to plot the traces in the unit sphere.')

    args = parser.parse_args()