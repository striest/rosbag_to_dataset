# this image correction is used for correcting the exposure for the warthog data
# Note that the original image_left and image_right folder will be renamed !!

import argparse
from os.path import isdir, isfile, join, split
from os import system

# python3 image_correction.py --root /project/learningphysics/arl_20220922_traj --bag_list trajlist_arl.txt --save_to /project/learningphysics/arl_20220922_preview
# python scripts/image_correction.py --root /cairo/arl_bag_files/sara_traj --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/arl_20220922_preview
replacelist = [
    ['height_map_correction_awb_gen', 'height_map'],
    ['rgb_map_correction_awb_gen', 'rgb_map'],
    ['points_left_correction_awb_gen', 'points_left'],
    ['depth_left_correction_awb_gen', 'depth_left'], 
    ['tartanvo_odom_correction_awb_gen', 'tartanvo_odom'],
]
switchlist = [
    ['image_right_correction_awb_gen', 'image_right'],
    ['image_left_correction_awb_gen', 'image_left'], 
]
removelist = [
    'image_right_correction_gen',
    'image_left_correction_gen',
    'image_right_correction_sep',
    'image_right_correction_awb_sep',
    'image_left_correction_sep',
    'image_left_correction_awb_sep',
    'image_right_correction_awb',
    'image_left_correction_awb',
    'image_right_correction',
    'image_left_correction'
]
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

        for [source_folder, target_folder] in replacelist:
            source_dir = join(trajdir, source_folder)
            target_dir = join(trajdir, target_folder)

            cmd = 'rm -rf ' + target_dir
            print(cmd)
            system(cmd)
            cmd = 'mv ' + source_dir + ' ' + target_dir
            print(cmd)
            system(cmd)
        
        for [source_folder, target_folder] in switchlist:
            source_dir = join(trajdir, source_folder)
            target_dir = join(trajdir, target_folder)

            cmd = 'mv ' + target_dir + ' ' + target_dir + '_raw'
            print(cmd)
            system(cmd)
            cmd = 'mv ' + source_dir + ' ' + target_dir
            print(cmd)
            system(cmd)

        for target_folder in removelist:
            target_dir = join(trajdir, target_folder)
            cmd = 'rm -rf ' + target_dir
            print(cmd)
            system(cmd)

if __name__ == '__main__':
    main()
