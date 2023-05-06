import torch
from os.path import join
from os import listdir
from tqdm import tqdm

root_fp ='/project/learningphysics/parvm/torch_dataset_20/eval'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    traj = torch.load(traj_fp)
    traj['observation']['delta'] =  traj['observation']['delta'].reshape((-1,1))
    traj['next_observation']['delta'] = traj['next_observation']['delta'].reshape((-1,1))
    torch.save(traj,traj_fp)