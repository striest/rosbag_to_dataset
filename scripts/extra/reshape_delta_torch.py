import torch
from os.path import join
from os import listdir
from tqdm import tqdm
from copy import deepcopy

root_fp ='/project/learningphysics/parvm/torch_dataset_20_correct_steer/train'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    traj = torch.load(traj_fp)
    traj['observation']['steer_angle'] =  deepcopy(traj['observation']['delta'])
    traj['next_observation']['steer_angle'] = deepcopy(traj['next_observation']['delta'])
    traj['observation'].pop('delta')
    traj['next_observation'].pop('delta')
    torch.save(traj,traj_fp)