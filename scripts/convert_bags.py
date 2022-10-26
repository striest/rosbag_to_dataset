import argparse
import numpy as np
import torch
import rosbag
import os

from rosbag_to_dataset.converter.converter_tofiles import ConverterToFiles
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir

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

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_list', type=str, default="", help='Path to the text file that lists out all the bag files to get data from. If bag_list is specified, bag_fp will be ignored.')
    parser.add_argument('--bag_fp', type=str, help='Path to a specific bag file to get data from. If bag_list is not specified, bag_fp will be used.')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--del_exist', action='store_true', default=False, help='Delete existing trajectory folder if exsits')

    args = parser.parse_args()

    print('Setting up...')
    config_parser = ConfigParser()
    converters, out_folders, rates, dt, maintopic = config_parser.parse_from_fp(args.config_spec)

    ## Choose whether to process list of bag files or single bag
    if args.bag_list != "" and os.path.isfile(args.bag_list):  # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bag_file_list = [line.strip().split(' ') for line in lines]
    elif(os.isfile(args.bag_fp)):  # process one file 
        bag_file_list = [[args.bag_fp, ""]]
    else:
        print("No input bagfiles specified.")
        exit()

    maybe_mkdir(args.save_to)

    ## Process bags
    for bagfile, out_folder in bag_file_list:
        print("-----")
        print(f'Reading bagfile {bagfile}')
        bag = rosbag.Bag(bagfile)
        print('Bagfile loaded')

        # create output folders
        traj_out_folder = os.path.join(args.save_to, out_folder)
        if args.del_exist:
            maybe_rmdir(traj_out_folder)
        maybe_mkdir(traj_out_folder)

        for k, folder in out_folders.items():
            maybe_mkdir(os.path.join(traj_out_folder, folder))

        converter = ConverterToFiles(traj_out_folder, dt, converters, out_folders, rates)
        success = dataset = converter.convert_bag(bag, main_topic=maintopic)

        if not success: 
            print(f'[FAILED] Converting bagfile {bagfile} failed.')
            maybe_rmdir(traj_out_folder)