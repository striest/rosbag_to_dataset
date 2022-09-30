import cv2
import numpy as np
import argparse
from os.path import isfile
from os import listdir

FLOATMAX = 1000000.0

def get_vis_heightmap(heightmap, scale=0.5, hmin=-1, hmax=4):
    mask = heightmap[:,:,0]==FLOATMAX
    disp1 = np.clip((heightmap[:, :, 0] - hmin)*100, 0, 255).astype(np.uint8)
    disp2 = np.clip((heightmap[:, :, 1] - hmin)*100, 0, 255).astype(np.uint8)
    disp3 = np.clip((heightmap[:, :, 2] - hmin)*100, 0, 255).astype(np.uint8)
    disp4 = np.clip(heightmap[:, :, 3]*1000, 0, 255).astype(np.uint8)
    disp1[mask] = 0
    disp2[mask] = 0
    disp3[mask] = 0
    disp4[mask] = 0
    disp_1 = np.concatenate((cv2.flip(disp1, -1), cv2.flip(disp2, -1)) , axis=1)
    disp_2 = np.concatenate((cv2.flip(disp3, -1), cv2.flip(disp4, -1)) , axis=1)
    disp = np.concatenate((disp_1, disp_2) , axis=0)
    disp = cv2.resize(disp, (0, 0), fx=scale, fy=scale)
    disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    return disp_color

def get_vis_rgbmap(rgbmap):
    return cv2.flip(rgbmap, -1)


if __name__ == '__main__':
    '''
    bag_list is a text file with the following content: 
      <Full path of the bagfile0> <Output folder name0>
      <Full path of the bagfile1> <Output folder name1>
      <Full path of the bagfile2> <Output folder name2>
      ...
    Turn the npy files of the RGB and height maps into png files for visualization
    Only the second part of the above text file is used
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--bag_list', type=str, default="", help='Path to the bag file to get data from')
    parser.add_argument('--heightmap_folder', type=str, default="height_map", help='Path to the heightmap folder')
    parser.add_argument('--rgbmap_folder', type=str, default="rgb_map", help='Path to the rgbmap folder')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')

    args = parser.parse_args()

    print('setting up...')

    # the input bagfiles
    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]
    else:
        print("No input bagfiles specified..")
        exit()

    for _, outfolder in bagfilelist:
        heightfolder = args.save_to + '/' + outfolder + '/' + args.heightmap_folder
        heightfiles = listdir(heightfolder)
        heightfiles = [ff for ff in heightfiles if ff.endswith('.npy')]
        heightfiles.sort()
        print('height map folder {}, {} files'.format(heightfolder, len(heightfiles)))
        for hh in heightfiles:
            hhnp = np.load(heightfolder + '/' + hh)
            hhvis = get_vis_heightmap(hhnp)
            cv2.imwrite(heightfolder + '/' + hh.replace('.npy', '.png'), hhvis)


        rgbfolder = args.save_to + '/' + outfolder + '/' + args.rgbmap_folder
        rgbfiles = listdir(rgbfolder)
        rgbfiles = [ff for ff in rgbfiles if ff.endswith('.npy')]
        rgbfiles.sort()        
        print('rgb map folder {}, {} files'.format(rgbfolder, len(rgbfiles)))
        for rr in rgbfiles:
            rrnp = np.load(rgbfolder + '/' + rr)
            rrvis = get_vis_rgbmap(rrnp)
            cv2.imwrite(rgbfolder + '/' + rr.replace('.npy', '.png'), rrvis)
