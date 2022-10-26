#! /usr/bin/python
from __future__ import print_function

import rosbag
import numpy as np
import argparse
import os
np.set_printoptions(suppress=True, threshold=10000,precision=4)

from rosbag_to_dataset.config_parser.config_parser import ConfigParser

class ConverterToFiles:
    """
    Class that converts a data in a rosbag to numpy arrays and saves to files using the format specified by the config spec. Because the recording time and the msg timestamp can be very different (in the case of re-recording the bags), we only look at the timestamps in the messages. To convert a file, we: 
    1. find the starting time
    2. go through the bagfile the first time, find and store all the timestamps for each topic
    3. align the timestamps
    4. cut the sequence, fill the missing frames
    5. go through the bagfile the second time, save files to hard drive 
    """
    def __init__(self, output_dir, dt, converters, out_folders, rates):
        """
        Args:
            output_dir: 
                String, directory where the data for a specific rosbag trajectory will be saved
            dt:
                Float, sampling period (1/sampling frequency) of the main topic, specified by user in the config_spec YAML file, or passed to this class by ConfigParser output
            converters:
                OrderedDict, output of ConfigParser specifying the class converter class for wach of the topics in the bag, as specified in the config_spec YAML file.
            out_folders:
                Dictionary, specifies the name of the folders for each of the topic names in the dataset.
            rates:
                Dictionary, specifies the sampling period (1/sampling frequency) for each topic (e.g. IMU data may be sampled 10x faster than camera data, which is usually the main topic).
        """
        self.output_dir = output_dir
        self.dt = dt
        self.converters = converters
        self.out_folders = out_folders
        self.rates = rates

    def reset_queue(self):
        self.queue = {}
        self.timesteps = {}
        self.bag_timestamps = {}
        self.topics = self.converters.keys()

        for k,v in self.converters.items():
            self.queue[k] = []
            self.timesteps[k] = [] 
            self.bag_timestamps[k] = [] 
        
    def find_first_msg_time(self, maintopic):
        self.start_timestamps = {tt: self.bag_timestamps[tt][0] for tt in self.topics}
        starttime_list = [self.bag_timestamps[tt][0] for tt in self.topics]
        max_starttime = max(starttime_list)
        main_topic_idx = np.searchsorted(self.bag_timestamps[maintopic], max_starttime)
        starttime = self.bag_timestamps[maintopic][main_topic_idx]

        for topic in self.topics: # find the closest time with startttime for each topic
            idx = np.searchsorted(self.bag_timestamps[topic], starttime) 
            diff1 = abs(starttime - self.bag_timestamps[topic][idx-1]) if idx>0 else 1000000
            diff2 = abs(starttime - self.bag_timestamps[topic][idx]) if idx<len(self.bag_timestamps[topic]) else 100000
            self.start_timestamps[topic] = self.bag_timestamps[topic][idx-1] if diff1<diff2 else self.bag_timestamps[topic][idx]
        return starttime

    def extract_timestamps_from_bag(self, bag):
        print('reading timestamps...')
        last_time = {k:0 for k in self.topics}
        for topic, msg, t in bag.read_messages(topics=list(self.topics)):
            has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
            has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

            # Use the timestamp if its valid. Otherwise default to rosbag time.
            if has_stamp or has_info:
                stamp = msg.header.stamp if has_stamp else msg.info.header.stamp
            else: 
                stamp = t
            stamp_sec = stamp.to_sec()
            self.bag_timestamps[topic].append(stamp_sec)
            # assert stamp_sec > last_time[topic], "Topic {} timestamp has gone back!".format(topic)
            # if stamp_sec <= last_time[topic]:
            #     import ipdb;ipdb.set_trace()
            last_time[topic] = stamp_sec

        for tt in self.topics:
            print('  {} \t {} \t {} - {}'.format( tt, len(self.bag_timestamps[tt]), self.bag_timestamps[tt][0], self.bag_timestamps[tt][-1]))

        return stamp_sec

    def convert_bag(self, bag, main_topic='/multisense/left/image_rect_color'):
        """
        Convert a bag into numpy arrays and save to files. 
        Assume the timestamps in the bagfile for each topic are ordered
        main_topic: use the time of the first frame of main_topic as the starting time
        """
        print('extracting messages...')
        self.reset_queue()

        bagtopics = bag.get_type_and_topic_info()[1].keys() # topics in the bagfile
        for k in self.topics:
            if k not in bagtopics:
                print("Could not find topic {} from envspec in the list of topics for this bag.".format(k))
                return False
            # assert k in bagtopics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

        self.extract_timestamps_from_bag(bag)
        starttime = self.find_first_msg_time(main_topic)
        endtime = min([self.bag_timestamps[tt][-1] for tt in self.topics])
        print("  starting time {}, ending time {}, duration {}".format(starttime, endtime, endtime-starttime))
        if starttime + self.rates[main_topic] >= endtime:
            print("Could not find enough overlap between topics!")
            return False

        # # each topic samples from its own starting timestamp, to avoid the oscilation problem
        # self.timesteps = {k:np.arange(self.start_timestamps[k], endtime, self.rates[k]) for k in self.topics} 
        # each topic samples from a fixed starting timestamp 
        self.timesteps = {k:np.arange(starttime, endtime, self.rates[k]) for k in self.topics}
        self.matched_idxs = {k:np.zeros_like(self.timesteps[k], dtype=np.int32) for k in self.topics}

        # find the topics with closest timestamps for the desired rates
        for topic in self.topics: 
            bagtimestamplist = self.bag_timestamps[topic]
            difflist = []
            for k, timestep in enumerate(self.timesteps[topic]):
                idx = np.searchsorted(bagtimestamplist, timestep)
                diff1 = abs(timestep - bagtimestamplist[idx-1]) if idx>0 else 1000000
                diff2 = abs(timestep - bagtimestamplist[idx]) if idx<len(bagtimestamplist) else 100000
                self.matched_idxs[topic][k] = idx-1 if diff1<diff2 else idx
                difflist.append(min(diff1, diff2))
            # sort out the conflicts
            idx = 0
            while idx<len(self.timesteps[topic]): # go through the timesteps list again
                sel_bag_idx = self.matched_idxs[topic][idx]
                minidx = idx
                mindiff = difflist[idx]
                while (idx<len(self.timesteps[topic])-1) and self.matched_idxs[topic][idx+1]==sel_bag_idx: # find out conflicting time steps 
                    diff = difflist[idx+1]
                    if diff<mindiff: # next frame is more match than the current frame
                        mindiff = diff
                        self.matched_idxs[topic][minidx] = -1
                        minidx = idx + 1
                    else:
                        self.matched_idxs[topic][idx+1] = -1
                    idx = idx + 1
                idx = idx + 1
            # update the timesteps to the actual time from the bag
            for k, timestep in enumerate(self.timesteps[topic]):
                if self.matched_idxs[topic][k] >= 0:
                    self.timesteps[topic][k] = bagtimestamplist[self.matched_idxs[topic][k]]
                else:
                    self.timesteps[topic][k] = -1

        # import ipdb;ipdb.set_trace()
        self.preprocess_queue()
        self.convert_queue(bag)
        return True

    def preprocess_queue(self, ):
        """
        Do some smart things to fill in missing data if necessary.
        """
        
        # import matplotlib.pyplot as plt
        # for k in self.topics:
        #     data = [0. if x is None else 1. for x in self.timesteps[k]]
        #     plt.plot(data, label=k)
        # plt.legend()
        # plt.show()
        
        #Start the dataset at the point where all topics become available
        print('preprocessing...')
        data_exists = {}
        strides = {}
        start_idxs = {}
        end_idxs = {}
        # import pdb; pdb.set_trace()
        for k in self.topics:
            data_exists[k] = [x>=0 for x in self.timesteps[k]]
            strides[k] = int(self.dt/self.rates[k])
            start_idxs[k] = (data_exists[k].index(True) -1 )// strides[k] +1 # cut off the imcomplete frames 
            end_idxs[k] = len(data_exists[k]) //strides[k]

        #This trick doesn't work with differing dts
        #thankfully, index gives first occurrence of value.
        start_idx = max(start_idxs.values())
        end_idx = min(end_idxs.values())

        #import pdb;pdb.set_trace()

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.

        for k in self.topics:
            start_frame = start_idxs[k] * strides[k]
            end_frame = end_idxs[k] * strides[k]
            last_avail = start_frame
            for tt in range(start_frame, start_idx * strides[k]):
                if data_exists[k][tt]:
                    last_avail = tt

            for t in range(start_idx* strides[k], end_frame):
                if data_exists[k][t]:
                    last_avail = t
                else:
                    self.timesteps[k][t] = self.timesteps[k][last_avail]
                    print('   -- {} filled missing frame {} - {}'.format(k, t, self.timesteps[k][t]))
            # import ipdb;ipdb.set_trace()
            data_exists[k] = [x>=0 for x in self.timesteps[k]]
            assert np.array(data_exists[k][start_frame: end_frame]).sum()==end_frame-start_frame, "Error in preprocessing queue! "
        
            print("  frames {}, \t start time {}, end time {}, topic {}".format((end_idx-start_idx)*strides[k], 
                                     self.timesteps[k][start_idx*strides[k]], 
                                     self.timesteps[k][end_idx*strides[k]-1], k))
        
        self.timesteps = {k:v[start_idx*strides[k]: end_idx*strides[k]] for k,v in self.timesteps.items()}

    def convert_queue(self, bag):
        """
        Actually convert the queue into files.
        """
        print('converting...')
        # out = {
        #     'observation':{},
        #     'action':{},
        # }
        topic_curr_idx = {k:0 for k in self.topics}

        for topic, msg, t in bag.read_messages(topics=list(self.topics)):
            tidx = topic_curr_idx[topic]
            if tidx >= len(self.timesteps[topic]):
                continue

            has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
            has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

            # Use the timestamp if its valid. Otherwise default to rosbag time.
            if has_stamp or has_info:
                stamp = msg.header.stamp if has_stamp else msg.info.header.stamp
            else: 
                stamp = t

            while tidx < len(self.timesteps[topic]) and abs(stamp.to_sec() - self.timesteps[topic][tidx])<10e-5: 
                filename = self.output_dir + '/' + self.out_folders[topic] + '/' + str(tidx).zfill(6) 
                self.queue[topic].append(self.converters[topic].save_file_one_msg(msg, filename) )
                topic_curr_idx[topic] = topic_curr_idx[topic] + 1
                tidx = topic_curr_idx[topic]

        for topic in self.topics:
            assert len(self.queue[topic]) == self.timesteps[topic].shape[0], 'convert_queue error: queuelen {} != timestep_len {}'.format(len(self.queue[topic]), self.timesteps[topic].shape[0])
            filefolder = self.output_dir + '/' + self.out_folders[topic] 
            self.converters[topic].save_file(self.queue[topic], filefolder)
            np.savetxt(self.output_dir + '/' + self.out_folders[topic] + '/timestamps.txt', np.array(self.timesteps[topic]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--save_as', type=str, required=True, help='Name of the file to save as')
    parser.add_argument('--use_stamps', type=str, required=False, default=True, help='Whether to use the time provided in the stamps or just the rosbag time')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    bag = rosbag.Bag(args.bag_fp)

    converter = ConverterToFiles(spec, converters, remap, rates, args.use_stamps)

    dataset = converter.convert_bag(bag)

    for k in dataset['observation'].keys():
        print('{}:\n\t{}'.format(k, dataset['observation'][k].shape))

    try:
        print('action:\n\t{}'.format(dataset['action'].shape))
    except:
        print('No actions')

    # fp = os.path.join(args.save_to, args.save_as)
    # np.savez(fp, **dataset)
