# this image correction is used for correcting the exposure for the warthog data
# Note that the original image_left and image_right folder will be renamed !!

import numpy as np
import cv2
import argparse
from os import listdir, system
from os.path import isdir, isfile, join, split
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir
import time

# python3 image_correction.py --root /project/learningphysics/arl_20220922_traj --bag_list trajlist_arl.txt --save_to /project/learningphysics/arl_20220922_preview
# python scripts/image_correction.py --root /cairo/arl_bag_files/sara_traj --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/arl_20220922_preview

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the trajectories data')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='List of bagfiles and output folders')

    args = parser.parse_args()

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

        # find unique values of the folder
        outfolderlist = [bb[1] for bb in bagfilelist]
        outfolders = set(outfolderlist)
        print('Find {} trajectories'.format(len(outfolders)))
    else:
        print('Find no trajectory in the bag list file {}'.format(args.bag_list))
        exit()

    for outfolder in outfolders:
        print('--- {} ---'.format(outfolder))
        trajdir = args.root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue

        # mv the raw folder
        heightmapfolder = trajdir + '/height_map_correction_awb_gen'

        heightmaplist = listdir(heightmapfolder)
        heightmaplist = [heightmapfolder + '/' + img for img in heightmaplist if img.endswith('.npy')]
        heightmaplist.sort()

        datanum = len(heightmaplist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            heightmap = np.load(heightmaplist[k])
            mask = heightmap[:, :, 0] < 1000
            heightmap[mask,:3] = heightmap[mask,:3] + 0.36

            np.save(heightmaplist[k], heightmap)
            if k % 100 == 0:
                print("  process {}...".format(k))

if __name__ == '__main__':
    main()
