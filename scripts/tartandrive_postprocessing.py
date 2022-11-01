import argparse
from ast import Store
from os.path import isfile, isdir

from rosbag_to_dataset.post_processing.tartanvo.imgs2odom import TartanVOInference
from rosbag_to_dataset.post_processing.tartanvo.imgs2pointcloud import StereoInference
from rosbag_to_dataset.post_processing.mapping.LocalMappingRegister import LocalMappingRegisterNode

# python scripts/tartandrive_postprocessing.py --root /home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output --bag_list scripts/trajlist_local.txt
if __name__ == '__main__':
    '''
    bag_list is a text file with the following content: 
      <Full path of the bagfile0> <Output folder name0>
      <Full path of the bagfile1> <Output folder name1>
      <Full path of the bagfile2> <Output folder name2>
      ...
    The extracted data will be stored in <save_to>/<Output folder name>

    if no bag_list is specified, the code will look at bag_fp and process the single bagfile. The output folder is specified by the save_to param
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the bag file to get data from')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='Path to the bag file to get data from')
    parser.add_argument('--vo', action='store_true', help='do vo')
    parser.add_argument('--stereo', action='store_true', help='do stereo')
    parser.add_argument('--mapping', action='store_true', help='do mapping')

    args = parser.parse_args()

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

    for bagfile, outfolder in bagfilelist:
        print('--- {} ---'.format(outfolder))
        trajdir = args.root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue

        if not isdir(trajdir + '/image_left') or \
             not isdir(trajdir + '/image_right') or \
             not isdir(trajdir + '/image_left_color'):
             print('!!! Missing data folders image_left, image_right or image_left_color')

        if args.vo:
            vonode = TartanVOInference()
            vonode.process(traj_root_folder=trajdir, vo_output_folder = 'tartanvo_odom')

        if args.stereo:
            stereonode = StereoInference()
            stereonode.process(traj_root_folder=trajdir, 
                        depth_output_folder='depth_left', points_output_folder='points_left')

        if args.mapping:
            mappingnode = LocalMappingRegisterNode()
            mappingnode.process(trajdir, heightmap_output_folder='height_map', rgbmap_output_folder='rgb_map')
