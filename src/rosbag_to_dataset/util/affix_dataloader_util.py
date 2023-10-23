from ast import arg
import queue
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from rosbag_to_dataset.post_processing import *
# from rosbag_to_dataset.post_processing.dataloader.TartanDriveDataset import DatasetBase
from torch_mpc.sysid.affix_sysid.AffixKBMDataset import AffixDatasetBase
from wheeledsim_rl.util.util import dict_to, DummyEnv
from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer

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


class AffixDataLoaderUtil:

	def __init__(self,frame_list_file_fp , dataroot_fp, \
				config, \
			datatypes = "prev_cmd,prev_odom,prev_delta,prev_params,cur_cmd,cur_odom,cur_delta,cur_params", \
			frame_skip = 0, frame_stride= 1, batch_size = 1, shuffle = True,\
			num_workers = 0, persistent_workers = True):
		self.frame_list_file_fp = frame_list_file_fp
		self.dataroot_fp= dataroot_fp
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.persistent_workers = persistent_workers
		
		self.dt = config['dt']
		self.datatypes = datatypes
		self.config = config
		self.num_workers = num_workers

		self.dataset = AffixDatasetBase(self.frame_list_file_fp, \
								dataroot= self.dataroot_fp, \
								datatypes = self.datatypes, \
								transform=None, \
								frame_skip = frame_skip, frame_stride=frame_stride, config = self.config)

	def preprocess_pose(self, traj, zero_init=True):
		"""
		Do a sliding window to smooth it out a bit
		"""
		N = 2
		for key in traj.keys() :
			k = 'odom'
			if k not in traj[key].keys():
				print("ODOM not found hence passing the init pose")
				continue

			T = traj[key][k].shape[1]
			pad_states = torch.cat([traj[key][k][...,[0],:]] * N + [traj[key][k]] + [traj[key][k][...,[-1],:]] * N,dim = -2)    
			smooth_states = torch.stack([pad_states[...,i:T+i,:] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[...,3]
			traj[key][k][...,3] = smooth_states
			if zero_init:
				init_state = traj[key][k][...,[0],:3]
				traj[key][k][...,:3] = traj[key][k][...,:3]- init_state
		return traj
	
	def calc_num_batches(self, capacity):
		if capacity == None:
			return None
		num_batches = int(capacity / (self.batch_size))
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
			traj = next(temp_dataloader)
			try:
				traj = self.preprocess_pose(traj)
			except Exception as e:
				print(e)
				import pdb;pdb.set_trace()
			for i in range(traj['prev']['odom'].shape[0]):  #TODO try to find a way to not use temp_traj to save time
				temp_traj = {'prev':{},'cur':{}}
				for k in traj.keys():
					for k1 in traj[k].keys():
						temp_traj[k][k1] = traj[k][k1][i]
						

				
				traj_list.append(temp_traj)
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

	print(f"Helloo")
	DataLoaderObj = AffixDataLoaderUtil(config["train_framelistfile"] , config["train_fp"], config, datatypes="odom,cmd,delta,params", batch_size = config["loader"]['train']['batch_size'], shuffle = config["loader"]['train']['shuffle'], num_workers = config["loader"]['train']['num_workers'], persistent_workers = config["loader"]['train']['persistent_workers'])
	future = concurrent.futures.ThreadPoolExecutor().submit(BackgroundLoader, DataLoaderObj,4)
	train_traj = future.result()
	for traj in train_traj:
		for k in traj.keys():
			print(f"{k} - {traj[k].shape}")
		break