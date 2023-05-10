import torch 
from os.path import join
from os import listdir 

import logging

from tqdm import tqdm 

logging.basicConfig(filename='diff.log',level=logging.DEBUG)



old_fp = '/project/learningphysics/parvm/torch_dataset' 
new_fp = '/project/learningphysics/parvm/torch_dataset_20' 

for k in ['train','eval']:
    old_folder_fp = join(old_fp,k)
    new_folder_fp = join(new_fp,k)

    old_trajs = listdir(old_folder_fp)
    new_trajs = listdir(new_folder_fp)

    old_trajs.sort()
    new_trajs.sort()

    for i,old_traj in tqdm(enumerate(old_trajs)): 
        new_traj = new_trajs[i]
        if old_traj != new_traj:
            logging.warning(f'{i} {old_traj} {new_traj} not equal !! CHECK')
        else:
            old_torch = torch.load(join(old_folder_fp,old_traj))
            new_torch = torch.load(join(new_folder_fp,new_traj))
            for modality in old_torch['observation'].keys():
                if modality != 'imu':
                    if old_torch['observation'][modality].shape != new_torch['observation'][modality].shape:
                        logging.warning(f'{modality} {old_traj} {new_traj} shape not equal !! CHECK')
                        continue
                    if torch.any(old_torch['observation'][modality] != new_torch['observation'][modality]):
                        logging.debug(f'{modality} {old_traj} {new_traj} observation not equal !! CHECK')
                        # import pdb;pdb.set_trace()
                    if torch.any(old_torch['next_observation'][modality] != new_torch['next_observation'][modality]):
                        logging.debug(f'{modality} {old_traj} {new_traj} next_observation not equal !! CHECK')
            for key in old_torch.keys():
                if key not in ['observation','next_observation']:
                    if old_torch[key].shape != new_torch[key].shape:
                        logging.warning(f'{key} {old_traj} {new_traj} shape not equal !! CHECK')
                        continue
                    if torch.any(old_torch[key] != new_torch[key]):
                        logging.debug(f'{key} {old_traj} {new_traj} key not equal !! CHECK')
                    