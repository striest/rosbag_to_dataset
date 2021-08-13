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
    def __init__(self, spec, converters, remap, rates, use_stamps):
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
        self.rates = rates
        self.use_stamps = use_stamps

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
        timesteps = {k:np.arange(info_dict['start'], info_dict['end'], self.rates[k]) for k in self.queue.keys()}

        topic_curr_idx = {k:0 for k in self.queue.keys()}
        prev_t = rospy.Time.from_sec(0.)
        durs = []

        #TODO: Write the code to check if stamp is available. Use it if so else default back to t.
        for topic, msg, t in bag.read_messages():
            if topic in self.queue.keys():
                tidx = topic_curr_idx[topic]

                #Check if there is a stamp and it has been set.
                has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.

                #Use the timestamp if its valid. Otherwise default to rosbag time.
                if has_stamp and self.use_stamps:
                    if (tidx < timesteps[topic].shape[0]) and (msg.header.stamp > rospy.Time.from_sec(timesteps[topic][tidx])):
                        #Add to data. Find the smallest timestep that's less than t.
                        idx = np.searchsorted(timesteps[topic], msg.header.stamp.to_sec())
                        topic_curr_idx[topic] = idx

                        #In case of missing data.
                        while len(self.queue[topic]) < idx:
                            self.queue[topic].append(None)

                        self.queue[topic].append(msg)
                else:
                    if (tidx < timesteps[topic].shape[0]) and (t > rospy.Time.from_sec(timesteps[topic][tidx])):
                        #Add to data. Find the smallest timestep that's less than t.
                        idx = np.searchsorted(timesteps[topic], t.to_sec())
                        topic_curr_idx[topic] = idx

                        #In case of missing data.
                        while len(self.queue[topic]) < idx:
                            self.queue[topic].append(None)

                        self.queue[topic].append(msg)


        #Make sure all queues same length
        for k in self.queue.keys():
            while len(self.queue[k]) < timesteps[k].shape[0]:
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
        strides = {}
        start_idxs = {}
        for k in self.queue.keys():
            data_exists[k] = [not x is None for x in self.queue[k]]
            strides[k] = int(self.dt/self.rates[k])
            start_idxs[k] = data_exists[k].index(True) // strides[k]

        #This trick doesn't work with differing dts
        #thankfully, index gives first occurrence of value.
        start_idx = max(start_idxs.values())

        #For now, just search backward to fill in missing data.
        #This is most similar to the online case, where you'll have stale data if the sensor doesn't return.
        for k in self.queue.keys():
            last_avail = start_idx * strides[k]
            for t in range(start_idx, len(self.queue[k])):
                if data_exists[k][t]:
                    last_avail = t
                else:
                    self.queue[k][t] = self.queue[k][last_avail]

        self.queue = {k:v[start_idx*strides[k]:] for k,v in self.queue.items()}

    def convert_queue(self):
        """
        Actually convert the queue into numpy.
        """
        print('converting...')
        out = {
            'observation':{},
            'action':{},
        }
        strides = {k:int(self.dt/self.rates[k]) for k in self.queue.keys()}

        for topic, converter in self.observation_converters.items():
#            print(topic, converter)

            data = self.queue[topic]
            data = [converter.ros_to_numpy(x) for x in data]
            data = np.stack(data, axis=0)

            if strides[topic] > 1:
                #If strided, need to reshape data (we also need to (same) pad the end)
                pad_t = strides[topic] - (data.shape[0] % strides[topic])
                data = np.concatenate([data, np.stack([data[-1]] * pad_t, axis=0)], axis=0)
                data = data.reshape(-1, strides[topic], *data.shape[1:])

            out['observation'][self.remap[topic]] = data

        for topic, converter in self.action_converters.items():
#            print(topic, converter)
            data = self.queue[topic]
            data = [converter.ros_to_numpy(x) for x in data]
            data = np.stack(data, axis=0)
            if len(data.shape) == 1:
                data = np.expand_dims(data, axis=1)
            out['action'][topic] = data

        if len(self.action_converters) > 0:
            out['action'] = np.concatenate([v for v in out['action'].values()], axis=1)

        min_t = min(out['action'].shape[0], min([v.shape[0] for v in out['observation'].values()]))

        return out

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--bag_fp', type=str, required=True, help='Path to the bag file to get data from')
    parser.add_argument('--save_to', type=str, required=True, help='Name of the dir to save the result to')
    parser.add_argument('--save_as', type=str, required=True, help='Name of the file to save as')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the rosbag time')

    args = parser.parse_args()

    print('setting up...')
    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    bag = rosbag.Bag(args.bag_fp)

    converter = Converter(spec, converters, remap, rates, args.use_stamps)

    dataset = converter.convert_bag(bag)

    for k in dataset['observation'].keys():
        print('{}:\n\t{}'.format(k, dataset['observation'][k].shape))

    try:
        print('action:\n\t{}'.format(dataset['action'].shape))
    except:
        print('No actions')

    fp = os.path.join(args.save_to, args.save_as)
    np.savez(fp, **dataset)
