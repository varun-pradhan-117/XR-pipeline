import numpy as np
import cv2
from scipy import ndimage

DEFAULT_GAMMA = 2.4
DEFAULT_LMAX = 300
DEFAULT_LMIN = 0.1

# Luma extracted using ITU-R BT.709
def get_luma(frame_data,format="BGR"):
    r=0.2126
    g=0.7152
    b=0.0722
    if format=="BGR":
        luma=frame_data[:,:,0]*b + frame_data[:,:,1]*g + frame_data[:,:,2]*r
    return luma

def get_luma_array(vid_array):
    r = 0.2126
    g = 0.7152
    b = 0.0722
    
    # Apply the function across all frames
    luma_array = np.einsum('ijkl,l->ijk', vid_array, [b, g, r])
    
    return luma_array


def normalize_frame(frame_data):
    frame_data=frame_data/255
    return frame_data
def normalize_array(vid_array):
    return vid_array/255

def denormalize_array(vid_array):
    return vid_array*255


# Using a simplified version of ITU-R Rec BT.1886
def eotf_1886(frame_data,gamma=DEFAULT_GAMMA, l_min=DEFAULT_LMIN,l_max=DEFAULT_LMAX):
    frame_data=np.maximum(frame_data,0.0)
    frame_data=np.minimum(frame_data,1.0)
    # a will be 1 since the frame is already mapped to luminance outside the function
    frame_data=np.power(frame_data,gamma)
    return (l_max-l_min)*frame_data + l_min

def oetf_PQ(frame_data):
    m = 78.84375
    n = 0.1593017578125
    c1 = 0.8359375
    c2 = 18.8515625
    c3 = 18.6875
    lm1 = np.power(10000.0, n)
    # FIXME: this might return an error if input is negative, see https://stackoverflow.com/q/45384602/
    lm2 = np.power(frame_data, n)
    frame_data = np.power((c1 * lm1 + c2 * lm2) / (lm1 + c3 * lm2), m)
    return frame_data


def get_SI(lum):
    sob_y=ndimage.sobel(lum,axis=1)
    sob_x=ndimage.sobel(lum,axis=0)
    si=np.hypot(sob_x,sob_y)[1:-1,1:-1]
    si=si.astype(np.float32).std()
    return si


def get_TI(frame_data,previous_frame=None):
    if previous_frame is None:
        return 0
    else:
        return (frame_data-previous_frame).std()
    
    
def get_SITI(vid_array):
    SI=[]
    TI=[]
    previous_frame=None
    luma_array=get_luma_array(vid_array)
    norm_luma=normalize_array(luma_array)
    for frame in norm_luma:
        frame_data=eotf_1886(frame)
        frame_data=oetf_PQ(frame_data)
        si=get_SI(frame_data)
        ti=get_TI(frame_data,previous_frame)
        
        SI.append(si)
        TI.append(ti)
        previous_frame=frame_data
    TI[0]=0
    SI=np.array(SI)
    SI=denormalize_array(SI)
    TI=np.array(TI)
    TI=denormalize_array(TI)
    return SI,TI