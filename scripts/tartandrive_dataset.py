import argparse
import rosbag

from rosbag_to_dataset.converter.converter_tofiles import ConverterToFiles
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir
from os.path import isfile

# python scripts/tartandrive_dataset.py --bag_fp /cairo/arl_bag_files/tartandrive/20210903_298.bag --config_spec specs/sample_tartandrive.yaml --save_to test_output/20210903_298
# python scripts/tartandrive_dataset.py --config_spec specs/sample_tartandrive.yaml --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/tartandrive_extract

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
    parser.add_argument('--bag_list', type=str, default="", help='Path to the bag file to get data from')
    parser.add_argument('--bag_fp', type=str, default="", help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--del_exist', action='store_true', default=False, help='Delete existing trajectory folder if exsits')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    converters, outfolders, rates, dt, maintopic = config_parser.parse_from_fp(args.config_spec)

    # the input bagfiles
    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]
    elif(isfile(args.bag_fp)): # process one file 
        bagfilelist = [[args.bag_fp, ""]]
    else:
        print("No input bagfiles specified..")
        exit()

    maybe_mkdir(args.save_to)
    for bagfile, outfolder in bagfilelist:
        # bagfile, outfolder = line.strip().split(' ')
        print('Reading bagfile {}'.format(bagfile))
        bag = rosbag.Bag(bagfile)
        print('Bagfile loaded')

        # create foldersoutfolders
        trajoutfolder = args.save_to+'/'+outfolder
        if args.del_exist:
            maybe_rmdir(trajoutfolder)
        maybe_mkdir(trajoutfolder)
        for k, folder in outfolders.items():
            maybe_mkdir(trajoutfolder+'/'+folder)

        converter = ConverterToFiles(trajoutfolder, dt, converters, outfolders, rates)
        suc = dataset = converter.convert_bag(bag, main_topic=maintopic)

        if not suc: 
            print('Convert bagfile {} failure..'.format(bagfile))
            maybe_rmdir(trajoutfolder)

