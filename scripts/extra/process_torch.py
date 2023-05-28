import torch
from os.path import join
from os import listdir
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import scipy
root_fp ='/home/offroad/parv_code/data/torch_dataset/torch_dataset_20_correct_steer/eval'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    traj = torch.load(traj_fp)

    # # idx = np.arange(traj['observation']['state'].shape[-1])
    # idx[3] = 4
    # idx[4] = 3
    traj['observation']['state'][...,[4,11]] *= -1
    traj['next_observation']['state'][...,[4,11]] *= -1
    
    rot_matrix = scipy.spatial.transform.Rotation.from_quat(traj['observation']['state'][..., 3:7].view(-1,4)).as_matrix()
    rot_6 = rot_matrix[...,[0,1]].reshape((*traj['observation']['state'].shape[:-1],6))
    new_state = np.concatenate([traj['observation']['state'][...,:3],rot_6,traj['observation']['state'][...,7:]],axis=-1)

    next_rot_matrix = scipy.spatial.transform.Rotation.from_quat(traj['next_observation']['state'][..., 3:7].view(-1,4)).as_matrix()
    next_rot_6 = next_rot_matrix[...,[0,1]].reshape((*traj['next_observation']['state'].shape[:-1],6))
    next_new_state = np.concatenate([traj['next_observation']['state'][...,:3],next_rot_6,traj['next_observation']['state'][...,7:]],axis=-1)

    traj['observation']['new_state'] =  torch.from_numpy(new_state).to(torch.float32)
    traj['next_observation']['new_state'] = torch.from_numpy(next_new_state).to(torch.float32)
    torch.save(traj,traj_fp)