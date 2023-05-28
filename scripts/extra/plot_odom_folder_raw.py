import matplotlib.pyplot as plt
import os
from os.path import join
# from torch_mpc.models.steer_setpoint_throttle_kbm import SteerSetpointThrottleKBM
# from torch_mpc.models.gravity_throttle_kbm import GravityThrottleKBM
import torch
from copy import deepcopy
import numpy as np
import yaml

import scipy
from rosbag_to_dataset.util.os_util import maybe_mkdirs


path = '/project/learningphysics/tartandrive_trajs'
traj_list_fp = '/home/parvm/tartan_extract_ws/src/rosbag_to_dataset/scripts/trajlist_tartan_train.txt'
dest = '/home/parvm/tartan_extract_ws/src/rosbag_to_dataset/scripts/extra/odom_folder_plots'
maybe_mkdirs(dest)
with open(traj_list_fp) as f:
    traj_list = [x.strip().split(' ')[-1] for x in f.readlines()]
for traj in traj_list[:30]:
    odom = np.load(join(path, traj,'odom','odometry.npy'))
    x,y,z,q0,q1,q2,q3,vx,vy,vz,wx,wy,wz = np.moveaxis(odom,-1,0)

    roll,pitch,yaw = np.moveaxis(scipy.spatial.transform.Rotation.from_quat(odom[..., 3:7]).as_rotvec(),-1,0)


    fig, axs = plt.subplots(4, 4,figsize=(20, 20))

    axs[0][0].plot(x)
    axs[0][0].legend(['x'])
    axs[0][1].plot(y)
    axs[0][1].legend(['y'])
    axs[0][2].plot(z)
    axs[0][2].legend(['z'])
    axs[0][3].plot(x,y)
    axs[0][3].plot(x[0],y[0],'o')
    axs[0][3].legend(['traj','start'])
    axs[1][0].plot(vx)
    axs[1][0].legend(['vx'])
    axs[1][1].plot(vy)
    axs[1][1].legend(['vy'])
    axs[1][2].plot(vz)
    axs[1][2].legend(['vz'])
    axs[2][0].plot(wx)
    axs[2][0].legend(['wx'])
    axs[2][1].plot(wy)
    axs[2][1].legend(['wy'])
    axs[2][2].plot(wz)
    axs[2][2].legend(['wz'])
    axs[3][0].plot(roll)
    axs[3][0].legend(['roll'])
    axs[3][1].plot(pitch)
    axs[3][1].legend(['pitch'])
    axs[3][2].plot(yaw)
    axs[3][2].legend(['yaw'])
    fig.tight_layout()

    plt.savefig(join(dest,f'{traj}.png'))
    plt.close()





