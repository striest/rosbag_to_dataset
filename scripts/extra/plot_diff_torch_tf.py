import matplotlib.pyplot as plt
import os
from os.path import join
# from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
# from torch_mpc.models.gravity_throttle_kbm import GravityThrottleKBM
import torch
from copy import deepcopy
import numpy as np
import yaml
from tqdm import tqdm
import scipy
from rosbag_to_dataset.util.os_util import maybe_mkdirs


path = '/project/learningphysics/parvm/torch_dataset_20_correct_steer/all'
traj_list_fp = '/home/parvm/tartan_extract_ws/src/rosbag_to_dataset/scripts/trajlist_tartan_train.txt'
dest = '/home/parvm/tartan_extract_ws/src/rosbag_to_dataset/scripts/extra/results/diff'
maybe_mkdirs(dest)
with open(traj_list_fp) as f:
    traj_list = [x.strip().split(' ')[-1]+'.pt' for x in f.readlines()]
for traj in tqdm(traj_list):
    odom = torch.load(join(path, traj))['observation']['state']
    x,y,z,q0,q1,q2,q3,vx,vy,vz,wx,wy,wz = odom.moveaxis(-1,0)
    roll,pitch,yaw = np.moveaxis(scipy.spatial.transform.Rotation.from_quat(odom[..., 3:7]).as_rotvec(),-1,0)

    x_diff = np.diff(x)/0.1
    y_diff = np.diff(y)/0.1
    z_diff = np.diff(z)/0.1

    roll_diff = np.diff(roll)/0.1
    pitch_diff = np.diff(pitch)/0.1
    yaw_diff = np.diff(yaw)/0.1

    fig, axs = plt.subplots(4, 4,figsize=(20, 20))

    axs[0][0].plot(x_diff)
    axs[0][0].legend(['x_diff'])
    axs[0][1].plot(y_diff)
    axs[0][1].legend(['y_diff'])
    axs[0][2].plot(z_diff)
    axs[0][2].legend(['z_diff'])
    axs[0][3].plot(yaw)
    axs[0][3].legend(['yaw'])
    axs[1][0].plot(vx)
    axs[1][0].legend(['vx'])
    axs[1][1].plot(vy)
    axs[1][1].legend(['vy'])
    axs[1][2].plot(vz)
    axs[1][2].legend(['vz'])
    axs[1][3].plot(np.arctan2(vy,vx))
    axs[1][3].legend(['tan-1(vy/vx)'])
    
    axs[3][0].plot(wx)
    axs[3][0].legend(['wx'])
    axs[3][1].plot(wy)
    axs[3][1].legend(['wy'])
    axs[3][2].plot(wz)
    axs[3][2].legend(['wz'])

    axs[2][0].plot(roll_diff)
    axs[2][0].legend(['roll_diff'])
    axs[2][1].plot(pitch_diff)
    axs[2][1].legend(['pitch_diff'])
    axs[2][2].plot(yaw_diff)
    axs[2][2].legend(['yaw_diff'])
    fig.tight_layout()

    plt.savefig(join(dest,f'{traj}.png'))
    plt.close()





