import torch
from os.path import join
from os import listdir
from tqdm import tqdm
from copy import deepcopy
import numpy as np
import scipy
root_fp ='/project/learningphysics/parvm/torch_2022/remaining_vel_split/vel/less_than_3.0'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    traj = torch.load(traj_fp)

    traj['terminal'][-1] = True
    torch.save(traj,traj_fp)