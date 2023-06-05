import torch
from os.path import join
from os import listdir
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import scipy
root_fp ='/project/learningphysics/parvm/torch_2021/vel_split/vel/between_3.0_5.0'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    traj = torch.load(traj_fp)
    if traj['next_observation']['new_state'].shape[-1] == 16:
        continue

    obs_steer = traj['observation']['steer_angle'] * (30./415.) * (-torch.pi/180.)
    traj['observation']['new_state'] = torch.cat([traj['observation']['new_state'],obs_steer],dim=-1)

    next_obs_steer = traj['next_observation']['steer_angle'] * (30./415.) * (-torch.pi/180.)
    traj['next_observation']['new_state'] = torch.cat([traj['next_observation']['new_state'],next_obs_steer],dim=-1)
    torch.save(traj,traj_fp)