import os
import pandas as pd
import numpy as np
from sklearn import preprocessing
import pickle
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R
import ast


def store_frames(video_path,output_folder,fps=None,rescale=2):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(f"Saving video {os.path.basename(video_path).split('.')[0]}")
    video=cv2.VideoCapture(os.path.join(video_path))
    original_frame_rate=round(video.get(cv2.CAP_PROP_FPS))

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
        # Sample at exact intervals if framerate is divisible by desired fps
        if is_div:
            if frame_count%frame_interval==0 or frame_count+1==last_frame:
                frame=cv2.resize(frame,(original_frame_width//rescale,original_frame_height//rescale))
                frames.append(frame)
                timestamps.append(frame_count*frame_duration)
        # Sample nearest timestep if framerate is not divisible by desired fps
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
        frames=sampled_frames
    video.release()
    frames=np.array(frames)
    timestamps=np.array(timestamps)

    np.save(f'{output_folder}/sampled_video.npy',frames)
    np.save(f'{output_folder}/timestamps.npy',timestamps)
    print(len(timestamps))


def store_instant_vel(data_folder, dataset):
    data_path=os.path.join(data_folder, dataset,'video_data')
    videos=os.listdir(data_path)
    for video in videos:
        unit_vectors=np.load(os.path.join(data_path,video,f'{video}_unit_vectors.npy'))
        ivs=np.diff(unit_vectors,axis=1)
        np.save(os.path.join(data_path,video, f'{video}_instantaneous_velocities.npy'), ivs)