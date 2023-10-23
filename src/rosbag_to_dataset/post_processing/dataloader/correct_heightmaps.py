import numpy as np
import argparse
from os.path import join, isdir
import os
from tqdm import tqdm 

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--root_fp', type=str, required=True, help='The root folder for all trajs')
	args = parser.parse_args()

	traj_list = [join(args.root_fp, x, 'height_map') for x in os.listdir(args.root_fp) if isdir(join(args.root_fp, x))]

	for traj in traj_list:
		frames = [join(traj, x) for x in os.listdir(traj) if x[-3:] == 'npy']
		# print(frames)
		for i in tqdm(frames):
			hmap = np.load(i)
			hmap[~np.isfinite(hmap)] = 0.
			np.save(i,hmap)
