import numpy as np
import argparse
from os.path import join, isdir
import os
from tqdm import tqdm 
from os import listdir


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	traj_fp = "/project/learningphysics/torch_dataset/20210905/atv_20210905_train"
	traj_folders = listdir(traj_fp)
	traj_folders = [x[:-3] for x in traj_folders]
	traj_folders.sort()
	f = open("/home/parvm/physics_atv_ws/src/learning/rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/data_partition/new/train.txt",'w')
	for x in traj_folders:
		f.write(f"{x}, ")
	f.close()
