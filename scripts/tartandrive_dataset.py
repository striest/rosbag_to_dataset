import argparse
import rosbag

import numpy as np
from rosbag_to_dataset.converter.converter_tofiles import ConverterToFiles
from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir, rm_file
from os.path import isfile, join, isdir
import time
# python scripts/tartandrive_dataset.py --bag_fp /cairo/arl_bag_files/tartandrive/20210903_298.bag --config_spec specs/sample_tartandrive.yaml --save_to test_output/20210903_298
# python scripts/tartandrive_dataset.py --config_spec specs/sample_tartandrive.yaml --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/tartandrive_extract
# python3 scripts/tartandrive_dataset.py --config_spec specs/sample_tartandrive_add.yaml --bag_fp /project/learningphysics/dataset/20210812/trial3.bag --save_to /project/learningphysics/wenshanw/traj_test/20210812_trial3 --preload_timestamp_folder image_left_color
# python3 scripts/tartandrive_dataset.py --config_spec specs/sample_tartandrive_add.yaml --bag_list scripts/trajlist.txt --save_to /project/learningphysics/tartandrive_trajs --preload_timestamp_folder image_left_color

class FileLogger():
    def __init__(self, filename, overwrite=False):
        if isfile(filename):
            if overwrite:
                print('Overwrite existing file {}'.format(filename))
            else:
                timestr = time.strftime('%m%d_%H%M%S',time.localtime())
                filename = filename+'_'+timestr
        self.f = open(filename, 'w')

    def log(self, logstr):
        print(logstr)
        self.f.write(logstr)

    def logline(self, logstr):
        print(logstr)
        self.f.write(logstr+'\n')

    def close(self,):
        self.f.close()

def mergebags(baglist, outputbag):
    '''
    Combine a list of bags that are temporally continuous
    '''
    baglist.sort()
    outbag = rosbag.Bag(outputbag, 'w')
    try:
        for bagfile in baglist:
            print("Merge {} to {}..".format(bagfile, outputbag))
            bag = rosbag.Bag(bagfile)
            for topic, msg, t in bag.read_messages():
                outbag.write(topic, msg, t)
    finally:
        outbag.close()
    outbag.close()

def load_timestamp(topicfolder):
    timestampfile = join(topicfolder, 'timestamps.txt')
    if not isfile(timestampfile):
        return None
    return np.loadtxt(timestampfile)

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
    parser.add_argument('--preload_timestamp_folder', type=str, default="", help='Use existing timestamps, this is used in adding new modalities')

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

    # we have many time-split bags, combine those bags while extracting the data
    bagfilelist_combine_split = []
    sublist = []
    lastfolder = bagfilelist[0][1]
    for bagfile, outfolder in bagfilelist:
        if outfolder == lastfolder:
            sublist.append(bagfile)
        else:
            bagfilelist_combine_split.append([sublist, lastfolder])
            sublist = [bagfile]
            lastfolder = outfolder
    if len(sublist)>0:
        bagfilelist_combine_split.append([sublist, lastfolder])

    print("Find {} bagfiles, {} trajectories".format(len(bagfilelist), len(bagfilelist_combine_split)))

    maybe_mkdir(args.save_to)
    for bagfiles, outfolder in bagfilelist_combine_split:
        # create folders
        trajoutfolder = join(args.save_to, outfolder)
        if args.del_exist:
            maybe_rmdir(trajoutfolder, force=True)
        maybe_mkdir(trajoutfolder)
        for k, folder in outfolders.items():
            maybe_mkdir(trajoutfolder+'/'+folder)

        # load existing timestamps in the incremental extraction mode
        timestamps = None
        if args.preload_timestamp_folder != "":
            maintopicfolder = join(trajoutfolder, args.preload_timestamp_folder)
            if not isdir(maintopicfolder):
                print("Cannot find folder for the timestamp file {}".format(maintopicfolder))
                continue
            # assert isdir(maintopicfolder), "Cannot find folder for the timestamp file {}".format(maintopicfolder)
            timestamps = load_timestamp(maintopicfolder)


        # merge the bagfiles
        if len(bagfiles) > 1:
            bagfile = trajoutfolder + '/merge_temp.bag'
            mergebags(bagfiles, bagfile)
        else:
            bagfile = bagfiles[0]

        logfilepath = trajoutfolder + '/data_extraction.log'
        logfile = FileLogger(logfilepath,overwrite=False)
        logfile.logline("Process bagfile {}".format(bagfiles[0]))

        bag = rosbag.Bag(bagfile)
        converter = ConverterToFiles(trajoutfolder, dt, converters, outfolders, rates)
        suc = converter.convert_bag(bag, main_topic=maintopic, logfile=logfile, preload_timestamps=timestamps)
        if bagfile.endswith('merge_temp.bag'): # remove the temp file
            rm_file(bagfile)

        if not suc: 
            logfile.logline('Convert bagfile {} failure..'.format(bagfile))
            logfile.close()

            if args.preload_timestamp_folder == "":
                maybe_rmdir(trajoutfolder, force=True)
        else:
            logfile.close()
