from ast import arg
import queue
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from rosbag_to_dataset.post_processing import *
from rosbag_to_dataset.post_processing.dataloader.TartanDriveDataset import DatasetBase
from wheeledsim_rl.util.util import dict_to, DummyEnv
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
import copy
import torch
import random

import numpy as np

import matplotlib.pyplot as plt
import argparse
import os
import yaml
import time
from datetime import datetime


from torch import nn, optim

from tqdm import tqdm

import argparse
from datetime import datetime
import os

import multiprocessing as mp
from multiprocessing import Pool
import multiprocessing.pool

import threading
import concurrent.futures


class DataLoaderUtil:

	def __init__(self,frame_list_file_fp , dataroot_fp, config, observation = None, frame_skip = 0, frame_stride= 5, batch_size = 1, shuffle = True, num_workers = 0, persistent_workers = True, init_dataset = True):
		self.frame_list_file_fp = frame_list_file_fp
		self.dataroot_fp= dataroot_fp
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.persistent_workers = persistent_workers
		
		self.dt = config['dt']
		self.N_per_step= config['N_per_step']
		self.imu_freq = self.N_per_step
		self.modality_len = config['modality_len']
		self.modalities_len = []
		
		self.remap= {'img0':'image_left','img1':'image_right','disp0':'disp0','imgc':'image_rgb','odom':'state','imu':'imu','heightmap':'heightmap','rgbmap':'rgbmap'}
		if observation != None:
			self.observation = observation
		else:
			self.observation = []
			for k,v in config['modalities'].items():
				for key, value in self.remap.items():
					if v['topic'] == value:
						self.observation.append(key)
						break
				if v['topic'] == 'heightmap':
					self.min_height = v['clipping'][0]
					self.max_height = v['clipping'][1]
					self.heightmap_num_channels = v['num_channels']
		self.action = ['cmd']
		self.observation.append('odom') #since odom is not added as modality
		self.remapped_obs = [self.remap[i] for i in self.observation]
		self.datatypes = ""
		for i in range(len(self.observation)):
			if i == 0:
				self.datatypes = f"{self.observation[i]}"
			else:
				self.datatypes = self.datatypes+f",{self.observation[i]}"
			if self.observation[i] != 'imu':
				self.modalities_len.append(self.modality_len)
			else:
				self.modalities_len.append(self.modality_len*self.N_per_step)

		if self.datatypes != "":
			self.datatypes=self.datatypes+","
		self.datatypes=self.datatypes+"cmd"
		self.modalities_len.extend([self.modality_len]) #for cmd

		self.config = config
		self.num_workers = num_workers
		if init_dataset:
			self.dataset = DatasetBase(self.frame_list_file_fp, \
									dataroot= self.dataroot_fp, \
									datatypes = self.datatypes, \
									modalitylens = self.modalities_len, \
									transform=None, \
									imu_freq = self.N_per_step, \
									frame_skip = frame_skip, frame_stride=frame_stride, config = self.config , remap = self.remap)


	def convert_queue(self, queue):
		"""
		Actually convert the queue into numpy.
		"""
		out = {
			'observation':{},
			'action':{},
		}

		for topic in self.remapped_obs:
			data = queue[topic]
			if topic == 'state' and self.config['state']['velocity'] == False:
				data = data[...,:7]
			out['observation'][topic] = data
		
		for topic in self.action:
			data = queue[topic]
			out['action'][topic] = data

		if len(self.action) > 0:
			out['action'] = np.concatenate([v for v in out['action'].values()], axis=1)
		
		return out

	def traj_to_torch(self, traj):
		torch_traj = {}

		#-1 to account for next_observation
		# trajlen = min(traj['action'].shape[0], min([traj['observation'][k].shape[0] for k in traj['observation'].keys()])) - 1

		trajlen = min(traj['action'].shape[1], min([traj['observation'][k].shape[1] for k in traj['observation'].keys()]))

		torch_traj['action'] = [0]*traj['action'].shape[0]
		torch_traj['reward'] = [0]*traj['action'].shape[0]
		torch_traj['terminal'] = [0]*traj['action'].shape[0]
		torch_traj['observation'] = [0]*traj['action'].shape[0]
		torch_traj['next_observation'] = [0]*traj['action'].shape[0]
		for i in range(traj['action'].shape[0]):
			max_nan_idx=-1
			for t in range(trajlen):
				temp = [not np.isfinite(traj['observation'][k][i][t]).any() for k in traj['observation'].keys()]
				obs_nan = any(temp)
				act_nan = not np.any(np.isfinite(traj['action'][i][t]))

				if obs_nan or act_nan:
					max_nan_idx = t
			start_idx = max_nan_idx + 1
			torch_traj['action'][i] = torch.tensor(traj['action'][i][start_idx:trajlen]).float()
			torch_traj['reward'][i] = torch.zeros(trajlen - start_idx)
			torch_traj['terminal'][i] = torch.zeros(trajlen - start_idx).bool()
			torch_traj['terminal'][i][-1] = True

			temp = {}
			for k,v in traj['observation'].items():
				if k != 'imu':
					temp[k] = torch.tensor(v[i][start_idx:trajlen]).float()
				else:
					strides = self.imu_freq
					data = torch.tensor(v[i][start_idx*strides:trajlen*strides]).float()
					if strides > 1:
						data = data.reshape(-1, strides, *data.shape[1:])
					temp[k] = data
			torch_traj['observation'][i] = temp

			temp = {}
			for k,v in traj['observation'].items():
				if k != 'imu':
					temp[k] = torch.tensor(v[i][start_idx+1:trajlen+1]).float()
					temp[k] = torch.cat((temp[k],temp[k][[-1]]),dim=0)
				else:
					strides = self.imu_freq
					data = torch.tensor(v[i][start_idx+1*strides:trajlen*strides]).float()
					pad_t = strides - (data.shape[0] % strides) #pad the final state with the missing frames as the previous state
					data = torch.cat([data, torch.stack([data[-1]] * pad_t, axis=0)], axis=0)
					if strides > 1:
						data = data.reshape(-1,strides, *data.shape[1:])
					temp[k] = data
			torch_traj['next_observation'][i] = temp

		torch_traj['action'] = torch.stack(torch_traj['action'])
		torch_traj['reward'] = torch.stack(torch_traj['reward'])
		torch_traj['terminal'] = torch.stack(torch_traj['terminal'])
		torch_traj['observation'] = {key: torch.stack([i[key] for i in torch_traj['observation']]) for key in torch_traj['observation'][0].keys()}
		torch_traj['next_observation'] = {key: torch.stack([i[key] for i in torch_traj['next_observation']]) for key in torch_traj['next_observation'][0].keys()}
		return torch_traj

	def preprocess_pose(self, traj, zero_init=True):
		"""
		Do a sliding window to smooth it out a bit
		"""
		N = 2
		T = traj['observation']['state'].shape[1]
		pad_states = torch.cat([traj['observation']['state'][:,[0]]] * N + [traj['observation']['state'][:]] + [traj['observation']['state'][:,[-1]]] * N,dim = 1)    
		smooth_states = torch.stack([pad_states[:,i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:,:,:3]
		pad_next_states = torch.cat([traj['next_observation']['state'][:,[0]]] * N + [traj['next_observation']['state'][:]] + [traj['next_observation']['state'][:,[-1]]] * N,dim = 1)
		temp = [pad_next_states[:,i:T+i] for i in range(N*2+1)]
		smooth_next_states = torch.stack(temp, dim=-1).mean(dim=-1)[:,:,:3]
		traj['observation']['state'][:,:, :3] = smooth_states
		traj['next_observation']['state'][:,:, :3] = smooth_next_states

		if zero_init:
			init_state = traj['observation']['state'][:,[0],:3]
			traj['next_observation']['state'][:,:,:3] = traj['next_observation']['state'][:,:,:3] - init_state
			traj['observation']['state'][:,:, :3] = traj['observation']['state'][:,:, :3]- init_state
		return traj

	def preprocess_observations(self, res, fill_value=0.):
		"""
		NOTE: These are temporary fixes to get the models to run.
		we are
			1. Just looking at the high value of the map (map should listen to both)
			2. Replacing nans with a fill value (we should add a mask channel)
		"""
		for k in res['observation'].keys():
			if k not in ['state','imu']:
				# map_data = res['observation'][k]
				# map_data[~torch.isfinite(map_data)] = fill_value
				# if k in ['heightmap']:
				# 	res['observation'][k] = res['observation'][k][...,:self.heightmap_num_channels]
				# 	res['observation'][k] = torch.clamp(res['observation'][k],self.min_height,self.max_height)
				res['observation'][k] = res['observation'][k].moveaxis(-1, -3)
				# map_data = res['next_observation'][k]
				# map_data[~torch.isfinite(map_data)] = fill_value
				# if k in ['heightmap']:
				# 	res['next_observation'][k] = res['next_observation'][k][...,:self.heightmap_num_channels]
				# 	res['next_observation'][k] = torch.clamp(res['next_observation'][k],self.min_height,self.max_height)
				res['next_observation'][k] = res['next_observation'][k].moveaxis(-1, -3)
		return res
	
	def calc_num_batches(self, capacity):
		if capacity == None:
			return None
		num_batches = int(capacity / (self.batch_size * self.modality_len))
		if self.num_workers!=0 and (num_batches % self.num_workers != 0):
			num_batches = num_batches + self.num_workers - (num_batches%self.num_workers)
		return num_batches
	
	def convert_obs_to_traj(self, num_batches = None):
		if self.num_workers > 0 :
			self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, num_workers = self.num_workers , persistent_workers = self.persistent_workers , shuffle=self.shuffle)
		else:
			self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
		traj_list = []
		temp_dataloader = iter(self.dataloader)
		if num_batches != None:
			batches_len = min(num_batches,len(temp_dataloader))
		else:
			batches_len = len(temp_dataloader)
		for _ in tqdm(range(batches_len)):
			data = next(temp_dataloader)
			for i in self.observation:
				data[self.remap[i]] = data[i]
				if self.remap[i] != i:
					data.pop(i)
			try:
				traj = self.convert_queue(data)
				torch_traj = self.traj_to_torch(traj)
				torch_traj = self.preprocess_pose(torch_traj)
				torch_traj = self.preprocess_observations(torch_traj)
				torch_traj['dt'] = torch.ones(torch_traj['action'].shape[0:2]) * self.dt
			except Exception as e:
				print(e)
				import pdb;pdb.set_trace()
			for i in range(torch_traj['action'].shape[0]):  #TODO try to find a way to not use temp_torch_traj to save time
				temp_torch_traj = {'observation':{},'next_observation':{}}
				for k in torch_traj.keys():
					if(type(torch_traj[k]) is dict):
						for k1 in torch_traj[k].keys():
							temp_torch_traj[k][k1] = torch.squeeze(torch_traj[k][k1][i],dim=0)
					else:
						temp_torch_traj[k] = torch.squeeze(torch_traj[k][i],dim=0)
				
				traj_list.append(temp_torch_traj)
		if self.num_workers > 0 :
			temp_dataloader._shutdown_workers()
		return traj_list

def BackgroundLoader(dataloader,num_batches=None):
	try:
		traj_list =  dataloader.convert_obs_to_traj(num_batches=num_batches)

		print(f"Loaded {len(traj_list)} trajs ")
		return traj_list
	except Exception as e:
		print(e)
		raise Exception

if __name__ == '__main__':
	print(f"Helloo")
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_fp', type=str, required=True, help='The yaml file for the experiment')
	args = parser.parse_args()
	

	config = yaml.load(open(args.config_fp, 'r'), Loader=yaml.FullLoader)

	DataLoaderObj = DataLoaderUtil(config["train_framelistfile"] , config["train_fp"], config, batch_size = config["loader"]['train']['batch_size'], shuffle = config["loader"]['train']['shuffle'], num_workers = config["loader"]['train']['num_workers'], persistent_workers = config["loader"]['train']['persistent_workers'])

	future = concurrent.futures.ThreadPoolExecutor().submit(BackgroundLoader, DataLoaderObj,4)
	env_traj = future.result()[0]
	env = DummyEnv(env_traj)
	train_buf = NStepDictReplayBuffer(env, capacity=10)
	future = concurrent.futures.ThreadPoolExecutor().submit(BackgroundLoader, DataLoaderObj,4)
	train_traj = future.result()