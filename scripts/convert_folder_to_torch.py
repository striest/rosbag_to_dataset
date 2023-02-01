import rospy
import torch
import rosbag
import yaml
import numpy as np
import argparse
from os import listdir
from os.path import join,isdir
import cv2
from PIL import Image
from numpy import asarray
from tqdm import tqdm
from rosbag_to_dataset.util.os_util import maybe_mkdirs
from copy import deepcopy

from datetime import datetime


class ConvertToTorchTraj:

    def __init__(self,config):
        self.config = config
        self.observation = list(self.config['observation'].keys())
        self.action = list(self.config['action'].keys())
        self.remap = {'odom':'state','delta':'delta','imu':'imu', 'cmd':'cmd','rgb_map': 'rgbmap', 'height_map':'heightmap','image_left_color':'image_rgb'}
        self.strides = dict([(x,1) for x in self.action + self.observation])
        if 'imu' in self.strides.keys():
            self.strides['imu'] = self.config['observation']['imu']['stride']
        self.dt = self.config['dt']
        self.res = dict([(k,x['res']) for k,x in self.config['observation'].items() if 'res' in x.keys()])
        self.reset()         
        
    def reset(self):
        self.queue = dict([(self.remap[x],None) for x in self.action + self.observation])
    
    def resize_image(self,x,output_res,key):
        if 'rgb' in key:
            x = x.astype(np.uint8)
        else:
            x = x.astype(np.float32)
        if key == 'height_map':
            if 'num_channels' in self.config['observation']['height_map'].keys():
                num_channels = self.config['observation']['height_map']['num_channels']
                x = x[...,num_channels]
            if 'empty_value' in self.config['observation'][key].keys():
                mask = np.isclose(abs(x), self.config["observation"]['height_map']['empty_value'])
                if self.config['observation'][key]['fill_value'] is not None:
                    fill_value = self.config['observation'][key]['fill_value']
                else:
                    fill_value = np.percentile(x[~mask], 99)
                x[mask] = fill_value
                mask = np.any(mask,axis=-1,keepdims=True).astype(np.float32)
                mask_normal = cv2.resize(mask, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
                mask_int = cv2.resize(mask, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_NEAREST)
                assert(len(mask_normal.shape) == 2)
                
                mask_normal = np.expand_dims(mask_normal,axis = -1)
                mask_int = np.expand_dims(mask_int,axis = -1)

                x = cv2.resize(x, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)
                x = np.concatenate((x,mask_normal,mask_int),axis=-1)
            return x
        else:
            return cv2.resize(x, dsize=(output_res[0],output_res[1]), interpolation=cv2.INTER_AREA)


    def load_maps_img(self,fp,last = 'npy',key = 'height_map'):
        all_frames = [join(fp,x) for x in listdir(fp) if x.endswith(last)]
        all_frames.sort()

        if last == 'npy':
            frame_list = [np.expand_dims(self.resize_image(np.load(x),self.res[key],key),axis=0) for x in all_frames]
        elif last =='jpg' or last =='png':
            frame_list = [np.expand_dims(self.resize_image(asarray(Image.open(x)),self.res[key],key),axis=0) for x in all_frames]

        if len(frame_list) > 0:
            final_queue = frame_list[0]

        for i in frame_list[1:]:
            final_queue = np.concatenate((final_queue,i),axis=0)

        self.queue[self.remap[key]] = final_queue

    def load_queue(self, fp):
        self.reset()
        for x in self.observation+self.action:
            traj = join(fp,x)
            flag = False
            if x == 'odom':
                fname = 'odometry.npy'
            elif x =='imu':
                fname = 'imu.npy'
            elif x == 'cmd':
                fname = 'twist.npy'
            elif x == 'delta':
                fname = 'float.npy'
            else:
                flag = True
                if 'map' in x:
                    last = 'npy'
                else:
                    last = 'png'
                self.load_maps_img(traj,last = last,key = x)
            
            if not flag:
                self.queue[self.remap[x]] = np.load(join(traj,fname))


    def convert_queue(self):
        """
        Actually convert the queue into numpy.
        """
        print('converting...')
        out = {
            'observation':{},
            'action':{},
        }
        for topic in self.observation:
            data = self.queue[self.remap[topic]] # 0 is time


            if self.strides[topic] > 1:
                data = data.reshape(-1, self.strides[topic], *data.shape[1:])

            out['observation'][self.remap[topic]] = data

        for topic in self.action:
            data = self.queue[self.remap[topic]]
            out['action'][topic] = data

        if len(self.action) > 0:
            out['action'] = np.concatenate([v for v in out['action'].values()], axis=1)

        return out

    def traj_to_torch(self, traj):
        torch_traj = {}

        #-1 to account for next_observation
        trajlen = min(traj['action'].shape[0], min([traj['observation'][k].shape[0] for k in traj['observation'].keys()])) - 1

        #Nan check
        max_nan_idx=-1
        for t in range(trajlen):
            obs_nan = any([not np.any(np.isfinite(traj['observation'][k][t])) for k in traj['observation'].keys()])
            act_nan = not np.any(np.isfinite(traj['action'][t]))

            if obs_nan or act_nan:
                max_nan_idx = t
        
        start_idx = max_nan_idx + 1
        torch_traj['action'] = torch.tensor(traj['action'][start_idx:trajlen]).float()
        torch_traj['reward'] = torch.zeros(trajlen - start_idx)
        torch_traj['terminal'] = torch.zeros(trajlen - start_idx).bool()
        torch_traj['terminal'][-1] = True
        torch_traj['observation'] = {k:torch.tensor(v[start_idx:trajlen]).float() for k,v in traj['observation'].items()}
        torch_traj['next_observation'] = {k:torch.tensor(v[start_idx+1:trajlen+1]).float() for k,v in traj['observation'].items()}

        return torch_traj

    def preprocess_pose(self, traj, zero_init=True):
        """
        Do a sliding window to smooth it out a bit
        """
        N = 2
        T = traj['observation']['state'].shape[0]
        pad_states = torch.cat([traj['observation']['state'][[0]]] * N + [traj['observation']['state']] + [traj['observation']['state'][[-1]]] * N)
        smooth_states = torch.stack([pad_states[i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:, :3]
        pad_next_states = torch.cat([traj['next_observation']['state'][[0]]] * N + [traj['next_observation']['state']] + [traj['next_observation']['state'][[-1]]] * N)
        smooth_next_states = torch.stack([pad_next_states[i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:, :3]
        traj['observation']['state'][:, :3] = smooth_states
        traj['next_observation']['state'][:, :3] = smooth_next_states

        if zero_init:
            init_state = traj['observation']['state'][0, :3]
            traj['next_observation']['state'][:, :3] = traj['next_observation']['state'][:, :3] - init_state
            traj['observation']['state'][:, :3] = traj['observation']['state'][:, :3] - init_state

        return traj
    
    def preprocess_observations(self, traj, fill_value=0.):
        """
        NOTE: These are temporary fixes to get the models to run.
        we are
            1. Just looking at the high value of the map (map should listen to both)
            2. Replacing nans with a fill value (we should add a mask channel)
        """
        for k in traj['observation'].keys():
            if k not in ['state','imu','delta']:
                # import pdb;pdb.set_trace()
                if 'rgb' in k:
                    og_key = [key for key,x in self.remap.items() if x == k][0]
                    if self.config['observation'][og_key]['normalize']:
                        traj['observation'][k] = traj['observation'][k]/255.
                        traj['next_observation'][k] = traj['next_observation'][k]/255.

                traj['observation'][k] = traj['observation'][k].moveaxis(-1, -3).moveaxis(-1, -2)
                traj['next_observation'][k] = traj['next_observation'][k].moveaxis(-1, -3).moveaxis(-1, -2)
        
        return traj
    
    def convert_to_torch(self,cvt,source_fp,save_fp):
        cvt.load_queue(source_fp)
        torch_traj = cvt.convert_queue()
        torch_traj = cvt.traj_to_torch(torch_traj)
        torch_traj = cvt.preprocess_pose(torch_traj)
        torch_traj = cvt.preprocess_observations(torch_traj)
        torch_traj['dt'] = torch.ones(torch_traj['action'].shape[0]) * self.dt
        torch.save(torch_traj,save_fp)


if __name__ =='__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--source_fp', type=str, required=True, help='Path to the source directory')
    parser.add_argument('--save_fp', type=str, required=True, help='Path to the destination trajectory')
    config = yaml.load(open('folder_to_traj.yaml', 'r'), Loader=yaml.FullLoader)
    args = parser.parse_args()

    root_source_fp = args.source_fp
    root_save_fp = args.save_fp

    now = datetime.now()
    program_time = f'{now.strftime("%m-%d-%Y,%H-%M-%S")}'

    root_save_fp = join(root_save_fp,program_time)

    maybe_mkdirs(root_save_fp, force=False)
    
    cvt = ConvertToTorchTraj(config)

    traj_name = [x for x in listdir(root_source_fp) if isdir(join(root_source_fp,x))]
    traj_name.sort()

    for x in tqdm(traj_name):
        source_fp = join(root_source_fp,x)
        save_fp = join(root_save_fp,f'{x}.pt')
        cvt.convert_to_torch(cvt,source_fp,save_fp)
        # import pdb;pdb.set_trace()