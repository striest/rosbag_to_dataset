#! /usr/bin/python
from __future__ import print_function

import rospy
import rosbag
import yaml
import numpy as np
import argparse
import os

from rosbag_to_dataset.config_parser.config_parser import ConfigParser

class Converter:
    """
    Rosnode that converts to numpy using the format specified by the config spec.
    Current way I'm handling this is by saving all the messages necessary, then processing them all at once.
    This way there shouldn't be any latency problems (as not all fields will be timestamped)

    Ok, the best paradigm for advancing time here is to have a single timer.
    Once a topic has added to its buf for that timestep, set a flag.
    Once dt has passed, reset the flags.
    We can error check by looking at flags.
    Real-time should be ok-ish as we don't process anything. If not, we can use stamps for the topics that have those.
    """
    def __init__(self, spec, converters, remap):
        """
        Args:
            spec: As provided by the ConfigParser module, the sizes of the observation/action fields.
            converters: As provided by the ConfigParser module, the converters from ros to numpy.
        """
        self.dt = spec.dt
        self.observation_space = spec.observation_space
        self.action_space = spec.action_space
        self.observation_converters = converters['observation']
        self.action_converters = converters['action']
        self.remap = remap

    def reset_queue(self):
        self.queue = {}

        for k,v in self.observation_converters.items():
            self.queue[k] = []

        for k,v in self.action_converters.items():
            self.queue[k] = []

    def convert_bag(self, bag):
        """
        Convert a bag into a dataset.
        """
        print('extracting messages...')
        self.reset_queue()
        info_dict = yaml.safe_load(bag._get_yaml_info())
        topics = bag.get_type_and_topic_info()[1].keys()
        for k in self.queue.keys():
            assert k in topics, "Could not find topic {} from envspec in the list of topics for this bag.".format(k)

        #For now, start simple. Just get the message that immediately follows the timestep
        #Assuming that messages are chronologically ordered per topic.
        timesteps = np.arange(info_dict['start'], info_dict['end'], self.dt)
        topic_curr_idx = {k:0 for k in self.queue.keys()}
        for topic, msg, t in bag.read_messages():
            t = t.to_time()
            if topic in self.queue.keys():
                tidx = topic_curr_idx[topic]
                if not tidx >= timesteps.shape[0] and t > timesteps[tidx]:
                    #Add to data. Find the smallest timestep that's less than t.
                    idx = np.searchsorted(timesteps, t)
                    topic_curr_idx[topic] = idx

                    #In case of missing data.
                    while len(self.queue[topic]) < idx:
                        self.queue[topic].append(None)

                    self.queue[topic].append(msg)

        #Make sure all queues same length
        for k in self.queue.keys():
            while len(self.queue[k]) < timesteps.shape[0]:
                self.queue[k].append(None)

        self.preprocess_queue()
        res = self.convert_queue()
        res['dt'] = self.dt
        return res

    def preprocess_queue(self):
        """
        Do some smart things to fill in missing data if necessary.
        """
        """
        import matplotlib.pyplot as plt
        for k in self.queue.keys():
            data = [0. if x is None else 1. for x in self.queue[k]]
            plt.plot(data, label=k)
        plt.legend()
        plt.show()
        """
        #Start the dataset at the point where all topics become available
        print('preprocessing...')
        data_exists = {}
        for k in self.queue.keys():
            data_exists[k] = [not x is None for x in self.queue[k]]

        #thankfully, index gives first occurrence of value.
        start_idx = max([x.index(True) for x in data_exists.values()])

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.
        for k in self.queue.keys():
            last_avail = start_idx
            for t in range(start_idx, len(self.queue[k])):
                if data_exists[k][t]:
                    last_avail = t
                else:
                    self.queue[k][t] = self.queue[k][last_avail]

        self.queue = {k:v[start_idx:] for k,v in self.queue.items()}

    def convert_queue(self):
        """
        Actually convert the queue into numpy.
        """
        print('converting...')
        out = {
            'observation':{},
            'action':{},
        }
        for topic, converter in self.observation_converters.items():
#            print(topic, converter)
            data = self.queue[topic]
            data = [converter.ros_to_numpy(x) for x in data]
            data = np.stack(data, axis=0)
            out['observation'][self.remap[topic]] = data

        for topic, converter in self.action_converters.items():
#            print(topic, converter)
            data = self.queue[topic]
            data = [converter.ros_to_numpy(x) for x in data]
            data = np.stack(data, axis=0)
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=1)
            out['action'][topic] = data

        out['action'] = np.concatenate([v for v in out['action'].values()], axis=1)

        return out

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--save_as', type=str, required=True, help='Name of the file to save as')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    spec, converters, remap = config_parser.parse_from_fp(args.config_spec)
    bag = rosbag.Bag(args.bag_fp)

    converter = Converter(spec, converters, remap)

    dataset = converter.convert_bag(bag)

    for k in dataset['observation'].keys():
        print('{}:\n\t{}'.format(k, dataset['observation'][k].shape))

    print('action:\n\t{}'.format(dataset['action'].shape))

    fp = os.path.join(args.save_to, args.save_as)
    np.savez(fp, **dataset)
