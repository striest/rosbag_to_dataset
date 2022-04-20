import argparse
import rosbag

from rosbag_to_dataset.converter.converter_tofiles import ConverterToFiles
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import maybe_mkdir

# python scripts/tartandrive_dataset.py --bag_fp /cairo/arl_bag_files/tartandrive/20210903_298.bag --config_spec specs/debug_offline.yaml --save_to test_output/20210903_298
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    converters, outfolders, rates, dt, maintopic = config_parser.parse_from_fp(args.config_spec)
    bag = rosbag.Bag(args.bag_fp)

    # create folders
    maybe_mkdir(args.save_to)
    for k, folder in outfolders.items():
        maybe_mkdir(args.save_to+'/'+folder)

    converter = ConverterToFiles(args.save_to, dt, converters, outfolders, rates)
    dataset = converter.convert_bag(bag, main_topic=maintopic)

