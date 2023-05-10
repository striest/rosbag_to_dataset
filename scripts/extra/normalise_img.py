import torch 
from os.path import join
from os import listdir 

import logging

from tqdm import tqdm 

paths = ['/project/learningphysics/parvm/torch_dataset_20/train','/project/learningphysics/parvm/torch_dataset_20/eval']

for path in paths:
    for traj in tqdm(listdir(path)):
        torch_fp = join(path,traj)
        torch_traj = torch.load(torch_fp)
        torch_traj['observation']['image_rgb'] = torch_traj['observation']['image_rgb'] / 255.
        torch_traj['next_observation']['image_rgb'] = torch_traj['next_observation']['image_rgb'] / 255.
        torch.save(torch_traj,torch_fp)
