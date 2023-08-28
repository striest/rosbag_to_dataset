import argparse
from ast import Store
from os.path import isfile, isdir

from rosbag_to_dataset.post_processing.tartanvo.imgs2odom import TartanVOInference
from rosbag_to_dataset.post_processing.tartanvo.imgs2pointcloud import StereoInference
from rosbag_to_dataset.post_processing.mapping.LocalMappingRegister import LocalMappingRegisterNode

# python scripts/tartandrive_postprocessing.py --root /home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output --bag_list scripts/trajlist_local.txt
# python3 scripts/tartandrive_postprocessing.py --root /project/learningphysics/tartandrive_trajs --bag_list scripts/trajlist_test.txt --save_to /project/learningphysics/tartandrive_trajs --stereo --vo --mapping
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
    parser.add_argument('--folder_suffix', type=str, default="", help='the intput and output folder')
    parser.add_argument('--platform', type=str, default="yamaha", help='the robot platform used to decide the sensor/body transformation')

    args = parser.parse_args()

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

        # find unique values of the folder
        outfolderlist = [bb[1] for bb in bagfilelist]
        outfolders = outfolderlist #set(outfolderlist)
        print('Find {} trajectories'.format(len(outfolders)))

    else:
        print('Find no trajectory in the bag list file {}'.format(args.bag_list))
        exit()

    folder_suffix = args.folder_suffix
    if folder_suffix != "":
        folder_suffix = '_' + folder_suffix

    for outfolder in outfolders:
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
            vonode.process(traj_root_folder=trajdir, vo_output_folder = 'tartanvo_odom'+folder_suffix,
                           left_input_folder='image_left'+folder_suffix, right_input_folder='image_right'+folder_suffix)

        if args.stereo:
            stereonode = StereoInference()
            stereonode.process(traj_root_folder=trajdir, 
                        left_input_folder='image_left'+folder_suffix, right_input_folder='image_right'+folder_suffix,
                        depth_output_folder='depth_left'+folder_suffix, points_output_folder='points_left'+folder_suffix)

        if args.mapping:
            mappingnode = LocalMappingRegisterNode(args.platform)
            mappingnode.process(trajdir, 
                                points_folder = 'points_left'+folder_suffix, vo_folder = 'tartanvo_odom'+folder_suffix, odom_folder = 'odom',  
                                heightmap_output_folder='height_map'+folder_suffix, rgbmap_output_folder='rgb_map'+folder_suffix)
