import torch
from os.path import join
from os import listdir
from tqdm import tqdm
import numpy as np 

root_fp ='/project/learningphysics/tartandrive_trajs'
traj_file = '/home/parvm/physics_atv_ws/src/learning/rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/data_partition/new/eval.txt'
traj_file = open(traj_file)
traj_list = traj_file.read()
traj_name_list = traj_list.split(', ')

all_traj_names = listdir(root_fp)

for traj_name in tqdm(all_traj_names):
    if traj_name not in traj_name_list:
        continue
    traj_fp = join(root_fp,traj_name)
    if 'delta' not in listdir(traj_fp):
        print(f'{traj_name} does not havea delta folder skipping')
        continue
    delta_fp = join(traj_fp,'delta/float.npy')
    delta = np.load(delta_fp)
    delta = delta.reshape((-1,1))
    np.save(delta_fp,delta)