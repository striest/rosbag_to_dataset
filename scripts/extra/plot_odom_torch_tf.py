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


# path = '/project/learningphysics/parvm/torch_dataset_20_correct_steer/all'
traj_list = '/home/parvm/ogpkg/src/wheeledsim_rl/scripts/world_models/data_split/perceptron/traj_2022_all.txt'
dest = '/home/parvm/tartan_extract_ws/src/rosbag_to_dataset/scripts/extra/results/2022'
maybe_mkdirs(dest)
# with open(traj_list_fp) as f:
#     traj_list = [x.strip().split(' ')[-1]+'.pt' for x in f.readlines()]

with open(traj_list) as f:
        fps = [x.strip() for x in f.readlines()]
        fps.sort()
for traj in tqdm(fps):
    traj_name = traj.split('/')[-1].split('.')[0]
    odom = torch.load(traj)['observation']['state']
    x,y,z,q0,q1,q2,q3,vx,vy,vz,wx,wy,wz = odom.moveaxis(-1,0)
    roll,pitch,yaw = np.moveaxis(scipy.spatial.transform.Rotation.from_quat(odom[..., 3:7]).as_rotvec(),-1,0)
    # if np.abs(pitch).max() < 0.3:
    #     continue 

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

    plt.savefig(join(dest,f'{traj_name}.png'))
    plt.close()





