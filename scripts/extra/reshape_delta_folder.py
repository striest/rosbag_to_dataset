import torch
from os.path import join
from os import listdir
from tqdm import tqdm

root_fp ='/project/learningphysics/tartandrive_trajs'
traj_names = listdir(root_fp)

for traj_name in tqdm(traj_names):
    traj_fp = join(root_fp,traj_name)
    if 'delta' not in listdir(traj_fp):
        print(f'{traj_name} does not havea delta folder skipping')
        continue
    delta_fp = join(traj_fp,'delta/float.npy')
    delta = np.load(delta_fp)
    delta = delta.reshape((-1,1))
    np.save(delta_fp,delta)