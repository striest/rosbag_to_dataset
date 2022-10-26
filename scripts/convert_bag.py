import argparse
import numpy as np
import torch
import rosbag
import os

from rosbag_to_dataset.converter.converter import Converter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import maybe_mkdir, str2bool

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--save_as', type=str, required=True, help='Name of the file to save as')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the rosbag time')
    parser.add_argument('--torch', type=str2bool, required=False, default=True, help='Whether to return the dataset as torch or numpy')
    parser.add_argument('--zero_pose_init', type=str2bool, required=False, default=True, help='Whether to initialize all trajs at 0')

    args = parser.parse_args()

    print('setting up...')
    import pdb;pdb.set_trace()
    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)

    bag = rosbag.Bag(args.bag_fp)

    converter = Converter(spec, converters, remap, rates, args.use_stamps)

    dataset = converter.convert_bag(bag, as_torch=args.torch, zero_pose_init=args.zero_pose_init)

    for k in dataset['observation'].keys():
        print('{}:\n\t{}'.format(k, dataset['observation'][k].shape))

    try:
        print('action:\n\t{}'.format(dataset['action'].shape))
    except:
        print('No actions')

    fp = os.path.join(args.save_to, args.save_as)
    if args.torch:
        torch.save(dataset, fp + '.pt')
    else:
        np.savez(fp, **dataset)
