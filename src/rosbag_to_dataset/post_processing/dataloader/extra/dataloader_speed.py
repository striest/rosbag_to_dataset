from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from TartanDriveDataset import DatasetBase
import torch

import numpy as np

import matplotlib.pyplot as plt
import argparse
import os
import yaml
import time

from torch import nn, optim

from tqdm import tqdm

import torch.distributed as dist
import argparse
from datetime import datetime
import os
import torch.multiprocessing as mp


if __name__ == '__main__':
	
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_fp', type=str, required=True, help='The yaml file for the experiment')
	args = parser.parse_args()
	

	config = yaml.load(open(args.config_fp, 'r'), Loader=yaml.FullLoader)
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = config["model_device"]+","+config["device_ids"]
	# os.environ["CUDA_VISIBLE_DEVICES"] ="0,1"
	

	observation = ['img0','heightmap','rgbmap','img1','imgc','disp0','odom','imu']
	observation = ['imgc','heightmap','rgbmap','odom','imu']
	remap= {'img0':'image_left','img1':'image_right','disp0':'disp0','imgc':'image_rgb','odom':'state','imu':'imu','heightmap':'heightmap','rgbmap':'rgbmap'}
	action = ['cmd']
	remapped_obs = [remap[i] for i in observation]
	datatypes = "imgc,heightmap,rgbmap,cmd,odom,imu" #observation + cmd
	dt = 0.1
	N_per_step= 10
	modality = 30
	modality_len = [modality]*(len(datatypes.split(","))-1)+[modality*N_per_step]
	print(modality_len)
	dataset_train = DatasetBase(config['train_framelistfile'], \
							dataroot= config['train_fp'], \
							datatypes = datatypes, \
							modalitylens = modality_len, \
							transform=None, \
							imu_freq = N_per_step, \
							frame_skip = 0, frame_stride=5, config = config , remap = remap)

	worker_list=[24,32]
	batch_size_list = [256]
	for k in tqdm(range(len(batch_size_list))):
		for i in tqdm(range(len(worker_list))):
			hst = time.time()
			train_dataloader = DataLoader(dataset_train, batch_size=batch_size_list[k],num_workers=worker_list[i],persistent_workers=True)
			train_dataloader = iter(train_dataloader)
			het = time.time() - hst
			lst = time.time()
			for j in tqdm(range(3)):
				data = next(train_dataloader)
			let = time.time()-lst
			logfile =open("log.txt",mode='a')
			print(f"{batch_size_list[k]} {worker_list[i]} Time for 1 traj = {let/(3*batch_size_list[k])}s . Init Time = {het}s\n",file=logfile)
			logfile.close()

				
				
