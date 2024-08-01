import numpy as np
from skimage.transform import resize
def xyz2thetaphi(x,y,z):
    R=np.sqrt(x**2+y**2+z**2)
    theta=np.arctan2(y,x)
    phi=np.arccos(z/R)
    theta=np.where(theta<0,theta+(2*np.pi),theta)
    phi=np.where(phi<0,phi+np.pi,phi)
    return theta, phi, R


def evaluate_fix_map_traj(az,el,resolution,scale=0.025):
    traj=np.zeros(len(az))
    x=np.zeros(len(az))
    y=np.zeros(len(az))
    # resolution should be in width,height
    fix_map=np.zeros((resolution[1],resolution[0]))
    fix_map=resize(fix_map,[np.ceil(resolution[1]*scale),np.ceil(resolution[0]*scale)])
    im_w=fix_map.shape[1]
    im_h=fix_map.shape[0]
    im_theta=np.linspace(0,2*np.pi-2*np.pi/im_w,im_w)
    im_phi=np.linspace(0+np.pi/(2*im_h),np.pi-np.pi/(2*im_h),im_h)
    for i in range(len(az)):
        target_theta=az[i]
        if np.isnan(el[i]):
            target_phi=0
        else:
            target_phi=el[i]
        
        #mindiff_theta=np.min(np.abs(im_theta-target_theta))
        im_col=np.argmin(np.abs(im_theta-target_theta))
        #mindiff_phi=np.min(np.abs(im_phi-target_phi))
        im_row=np.argmin(np.abs(im_phi-target_phi))
        #fix_map[im_row,im_col]=1
        #x[i]=im_row
        #y[i]=im_col
        traj[i]=im_col*im_h+(im_row+1)
        #fix_map=np.zeros_like(fix_map)
        
    return traj,x,y

def lzentropy_AE(rd):
    n=len(rd)
    L=np.zeros(n)
    L[0]=1
    for i in  range(1,n):
        sub=rd[i]
        match=rd[:i]==sub
        if np.all(match==0):
            L[i]=1/np.log2(i+1)
        else:
            k=1
            while k<i+1:
                if i+k+1>n:
                    break
                sub=rd[i:i+k+1]
                for j in range(i):
                    match=rd[j:j+len(sub)]==sub    
                    if np.all(match==1):
                        break
                L[i]=len(sub)/np.log2(i+1)
                if np.all(match==1)==0:
                    k=i
                k=k+1
        
    AE=(1/n*sum(L))**(-1)
    IE=L
    return AE,IE

def lzentropy_IE_rec(rd,L):
    j=len(rd)-1
    k=0
    while k<=j:
        if j+k<len(rd)-1:
            break
        sub=rd[j-k:j+1]
        for t in range(j-len(sub)+1):
            match=rd[t:t+len(sub)]==sub
            if np.all(match==1):
                sub=rd[j-k+1:j]
                break
        if j<len(L):
            L[j]=len(sub)/np.log2(j+1)
        else:
            L.append(len(sub)/np.log2(j+1))
        if not np.all(match==1):
            k=j
        k=k+1
        
    IE=(1/(j+1)*np.sum(L))**(-1)
    return IE,L

def get_entropies(data_x,data_y,data_z,vr,scale=0.025):
    theta,phi,R=xyz2thetaphi(data_x,data_y,data_z)
    phi[np.isnan(phi)]=0
    traj,x,y=evaluate_fix_map_traj(theta,phi,vr,scale)
    
    actual_entropy_traj,t=lzentropy_AE(traj)
    vid_IE=np.zeros([len(traj),1])
    L=[1,0]
    for i in range(1,len(traj)):
        vid_IE[i],L=lzentropy_IE_rec(traj[0:i+1],L)

    return actual_entropy_traj,vid_IE,L