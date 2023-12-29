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
    Rosnode that converts to numpy and saves to files using the format specified by the config spec.
    Because the recording time and the msg timestamp can be very different (in the case of re-recording the bags), 
    we only look at the timestamps in the messages. 
    1. find the starting time
    2. go through the bagfile the first time, find and store all the timestamps for each topic
    3. align the timestamps
    4. cut the sequence, fill the missing frames
    5. go through the bagfile the second time, save files to hard drive 
    """
    def __init__(self, outputdir, dt, converters, outfolders, rates):
        """
        Args:
            spec: As provided by the ConfigParser module, the sizes of the observation/action fields.
            converters: As provided by the ConfigParser module, the converters from ros to numpy.
        """
        self.outputdir = outputdir
        self.dt = dt
        self.converters = converters
        self.outfolders = outfolders
        self.rates = rates

    def reset_queue(self):
        self.queue = {}
        self.timesteps = {}
        self.bagtimestamps = {}
        self.topics = self.converters.keys()
        
    def find_first_msg_time(self, maintopic):
        self.start_timestamps = {tt: min(self.bagtimestamps[tt]) for tt in self.topics}
        starttime_list = [min(self.bagtimestamps[tt]) for tt in self.topics]
        max_starttime = max(starttime_list)
        main_topic_idx = np.searchsorted(self.bagtimestamps[maintopic], max_starttime)
        starttime = self.bagtimestamps[maintopic][main_topic_idx]

        for topic in self.topics: # find the closest time with startttime for each topic
            idx = np.searchsorted(self.bagtimestamps[topic], starttime) 
            diff1 = abs(starttime - self.bagtimestamps[topic][idx-1]) if idx>0 else 1000000
            diff2 = abs(starttime - self.bagtimestamps[topic][idx]) if idx<len(self.bagtimestamps[topic]) else 100000
            self.start_timestamps[topic] = self.bagtimestamps[topic][idx-1] if diff1<diff2 else self.bagtimestamps[topic][idx]
        return starttime

    def time_from_msg(self, msg, t):
        has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
        has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

        # Use the timestamp if it's valid. Otherwise default to rosbag time.
        if has_stamp or has_info:
            stamp = msg.header.stamp if has_stamp else msg.info.header.stamp
        else: 
            stamp = t
        stamp_sec = stamp.to_sec()

        return stamp_sec

    def extract_timestamps_from_bag(self, bag, logfile):
        '''
        extract all the timestamps from the bag for each topic
        self.bagtimestamps - a dictionary of list of timestamps
        '''
        logfile.logline('STEP 1: reading timestamps...')
        self.bagtimestamps = {k:[] for k in self.topics} 
        last_time = {k:0 for k in self.topics}

        for topic, msg, t in bag.read_messages(topics=list(self.topics)):
            stamp_sec = self.time_from_msg(msg, t)
            self.bagtimestamps[topic].append(stamp_sec)
            last_time[topic] = stamp_sec

        for tt in self.topics:
            logfile.logline('  {} \t {} \t {} - {}'.format( tt, len(self.bagtimestamps[tt]), min(self.bagtimestamps[tt]), max(self.bagtimestamps[tt])))

    def find_start_end_timesteps(self, main_topic, preload_timestamps, logfile):
        '''
        for each topic, what is the sampling timestamp, given the sampling frequency 
        self.timesteps - a dictionary of list of timestamps
        '''

        # # each topic samples from its own starting timestamp, to avoid the occilation problem
        # self.timesteps = {k:np.arange(self.start_timestamps[k], endtime, self.rates[k]) for k in self.topics} 
        # each topic samples from a fixed starting timestamp 
        if preload_timestamps is None:
            starttime = self.find_first_msg_time(main_topic)
            endtime = min([max(self.bagtimestamps[tt]) for tt in self.topics])
            logfile.logline("  starting time {}, ending time {}, duration {}".format(starttime, endtime, endtime-starttime))
            if starttime + self.rates[main_topic] >= endtime:
                logfile.logline("Could not find enough overlap between topics!")
                return False
            self.timesteps = {k:np.arange(starttime, endtime, self.rates[k]) for k in self.topics}
        else:
            self.timesteps = {k:np.arange(preload_timestamps[0], preload_timestamps[-1] + self.rates[k], self.rates[k]) for k in self.topics}
            # self.timesteps = {k:preload_timestamps for k in self.topics}
            starttime = max([min(self.bagtimestamps[tt]) for tt in self.topics])
            endtime = min([max(self.bagtimestamps[tt]) for tt in self.topics])
            if starttime > preload_timestamps[0] or endtime < preload_timestamps[-1]:
                logfile.logline("Sample with preloaded timestamps, but it is out of the topic range {} - {}".format(starttime, endtime))

        # make sure all topics are completed and aligned at the end, add a few more if needed
        for k in self.timesteps.keys():
            modality_shape = self.timesteps[k].shape
            N_per_step = int(self.dt / self.rates[k])
            logfile.logline("  -- extracting {} at {} x main freq..".format(k, N_per_step))
            if N_per_step > 1 and modality_shape[0]%N_per_step != 0:
                num_missing_frames = N_per_step - modality_shape[0]%N_per_step
                st_missing_frames = self.timesteps[k][-1]+self.rates[k]
                self.timesteps[k] = np.concatenate((self.timesteps[k],np.arange(st_missing_frames, st_missing_frames+self.rates[k] *(num_missing_frames - 0.1),self.rates[k])))
        
        return True

    def match_timestamps(self, logfile):
        '''
        this is a classic problem of finding the best match between two sequences
        '''
        logfile.logline('STEP 2: matching timestamps...')

        # find the topics with closest timestamps for the desired rates
        for topic in self.topics: 
            bagtimestamplist = self.bagtimestamps[topic]
            framenum = len(self.timesteps[topic])
            bagframenum = len(bagtimestamplist)

            match_idxs_sorted = np.zeros_like(self.timesteps[topic], dtype=np.int32)
            idx_be_matched_sorted = np.zeros_like(bagtimestamplist, dtype=np.int32)

            bagtimestamp_sort = np.array(bagtimestamplist)
            bagtimestamp_sort.sort() 

            difflist = []
            for k, timestep in enumerate(self.timesteps[topic]):
                idx = np.searchsorted(bagtimestamp_sort, timestep)
                diff1 = abs(timestep - bagtimestamp_sort[idx-1]) if idx>0 else 1000000
                diff2 = abs(timestep - bagtimestamp_sort[idx]) if idx<bagframenum else 100000
                if diff1 < diff2:
                    match_idxs_sorted[k] = idx-1 
                    idx_be_matched_sorted[idx-1] = 1
                    difflist.append(diff1)
                else:
                    match_idxs_sorted[k] = idx
                    idx_be_matched_sorted[idx] = 1
                    difflist.append(diff2)
            
            # sort out the conflicts
            idx = 0
            while idx < framenum: # go through the timesteps list again
                sel_bag_idx = match_idxs_sorted[idx]
                minidx = idx
                mindiff = difflist[idx]
                while (idx < framenum-1) and match_idxs_sorted[idx+1]==sel_bag_idx: # find out conflicting time steps 
                    diff = difflist[idx+1]
                    if diff<mindiff: # next frame is more match than the current frame
                        mindiff = diff
                        match_idxs_sorted[minidx] = -1
                        minidx = idx + 1
                    else:
                        match_idxs_sorted[idx+1] = -1
                    idx = idx + 1
                idx = idx + 1

            # pick up leftovers to fill the missing frames 
            count = 0
            for k, timestep in enumerate(self.timesteps[topic]): # go throught the list again 
                if match_idxs_sorted[k] == -1:
                    idx = np.searchsorted(bagtimestamp_sort, timestep)
                    matchleft = idx_be_matched_sorted[idx-1]==1 if idx > 0 else False
                    matchright = idx_be_matched_sorted[idx]==1 if idx < bagframenum else False
                    assert matchleft or matchright or idx<=bagframenum or idx>=bagframenum, "match_timestamps error! "

                    if idx-1 > 0 and not matchleft:
                        match_idxs_sorted[k] = idx-1
                        idx_be_matched_sorted[idx-1] = 1
                        count += 1
                    elif idx < bagframenum and (not matchright): 
                        match_idxs_sorted[k] = idx
                        idx_be_matched_sorted[idx] = 1
                        count += 1
            logfile.logline("  rematched {} frames for topic {}".format(count, topic))

            # update the timesteps to the actual time from the bag
            for k in range(framenum):
                if match_idxs_sorted[k] >= 0:
                    self.timesteps[topic][k] = bagtimestamp_sort[match_idxs_sorted[k]]
                else:
                    self.timesteps[topic][k] = -1

    def convert_bag(self, bag, main_topic, logfile, preload_timestamps=None):
        """
        Convert a bag into numpy arrays and save to files. 
        Due to network delay, the messages in the bags might be in wrong order
        The message will be sorted based on the timestamps in the header

        main_topic: use the time of the first frame of main_topic as the starting time
        If preload_timestamps is provided, use it as the sampling reference. 
            Note that the new sampled topics could be with different length and will be filled. 
        """
        logfile.logline('Convert bag to files...')
        self.reset_queue()

        bagtopics = bag.get_type_and_topic_info()[1].keys() # topics in the bagfile
        for k in self.topics:
            if k not in bagtopics:
                logfile.logline("Could not find topic {} from envspec in the list of topics for this bag.".format(k))
                return False
            # assert k in bagtopics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

        # extract timestamps from the bag -self.bagtimestamps
        self.extract_timestamps_from_bag(bag, logfile)

        # calculate the sampling timestamps given the frequency - self.timesteps
        res = self.find_start_end_timesteps(main_topic, preload_timestamps, logfile)
        if not res:
            return False

        # sample the frames by matching the time between self.bagtimestamps and self.timesteps
        self.match_timestamps(logfile)

        # fill the unmatched frames
        self.preprocess_queue(logfile)

        # save data to files
        self.convert_queue(bag, logfile)

        return True

    def preprocess_queue(self, logfile):
        """
        Do some smart things to fill in missing data if necessary.
        """
                
        #Start the dataset at the point where all topics become available
        logfile.logline('STEP 3: queue preprocessing...')
        # import ipdb; ipdb.set_trace()

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.

        for k in self.topics:
            data_exists = [x>=0 for x in self.timesteps[k]]
            framenum = len(data_exists)
            missingframe = framenum - np.sum(data_exists)
            start_frame = data_exists.index(True) #start_idxs[k] * strides[k]
            last_avail = start_frame
            for tt in range(0, start_frame): # fill the missing frames at the beginning
                self.timesteps[k][tt] = self.timesteps[k][last_avail]

            for t in range(start_frame, framenum):
                if data_exists[t]:
                    last_avail = t
                else:
                    self.timesteps[k][t] = self.timesteps[k][last_avail]
                    logfile.logline('   -- {} filled missing frame {} <- {}: {}'.format(k, t, last_avail, self.timesteps[k][t]))
            # import ipdb;ipdb.set_trace()

            # check all the frames are filled
            data_exists = [x>=0 for x in self.timesteps[k]] 
            assert np.array(data_exists).sum()==framenum, "Error in preprocessing queue! "

            # check all the timesteps make sense
            assert np.all(self.timesteps[k][:-1] <= self.timesteps[k][1:]), "Error in preprocessing temesteps! "
        
            logfile.logline("  topic {} frames {}, \t start time {}, end time {}, filling missing {} frames".format(k, framenum, 
                                     self.timesteps[k][0], 
                                     self.timesteps[k][framenum-1], 
                                     missingframe))
            logfile.logline("")

    def convert_queue(self, bag, logfile):
        """
        Actually convert the queue into files.
        """
        logfile.logline('STEP 4: converting the queue into files...')

        self.queue = {k:[[],]*len(self.timesteps[k]) for k in self.topics}

        for topic, msg, t in bag.read_messages(topics=list(self.topics)):

            stamp_sec = self.time_from_msg(msg, t)

            # find all the matched frames by timestamp 
            idx = np.searchsorted(self.timesteps[topic], stamp_sec)
            while idx < len(self.timesteps[topic]) and abs(stamp_sec - self.timesteps[topic][idx])<10e-6:
                filename = self.outputdir + '/' + self.outfolders[topic] + '/' + str(idx).zfill(6) 
                self.queue[topic][idx] = (self.converters[topic].save_file_one_msg(msg, filename) )
                idx += 1

        for topic in self.topics:
            assert len(self.queue[topic]) == self.timesteps[topic].shape[0], 'convert_queue error: queuelen {} != timestep_len {}'.format(len(self.queue[topic]), self.timesteps[topic].shape[0])
            for k, ele in enumerate(self.queue[topic]): # check all frames are filled! 
                assert (not isinstance(ele, list)) or len(ele)>0, 'convert_queue error: topic {}, queue idx {} at time {} is missing'.format(topic, k, self.timesteps[k])
            
            filefolder = self.outputdir + '/' + self.outfolders[topic] 
            self.converters[topic].save_file(self.queue[topic], filefolder)
            np.savetxt(self.outputdir + '/' + self.outfolders[topic] + '/timestamps.txt', np.array(self.timesteps[topic]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--save_as', type=str, required=True, help='Name of the file to save as')
    parser.add_argument('--use_stamps', type=bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the rosbag time')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    bag = rosbag.Bag(args.bag_fp)

    converter = ConverterToFiles(spec, converters, remap, rates)

    dataset = converter.convert_bag(bag)

    for k in dataset['observation'].keys():
        print('{}:\n\t{}'.format(k, dataset['observation'][k].shape))

    try:
        print('action:\n\t{}'.format(dataset['action'].shape))
    except:
        print('No actions')

    fp = os.path.join(args.save_to, args.save_as)
    np.savez(fp, **dataset)
