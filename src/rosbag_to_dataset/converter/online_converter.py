#! /usr/bin/python3
import rospy
import torch
import rosbag
import yaml
import numpy as np
import argparse
import os

from rosbag_to_dataset.config_parser.config_parser import ConfigParser
from rosbag_to_dataset.util.os_util import str2bool

class OnlineConverter:
    """
    A converter that uses the same spec to maintain a buffer of the latest observations.

    As a datatype, this class will hold time-indexed buffers for each relevant topic, at the rate given by the topic, for an interval of 2*dt.
    (2x is a safety factor, the index of last advanced time is queue_len / 2.)
    Every dt, we will advance the queues.
    When we read a message, we will load it in the buffer if there's a time-slot for it. If there is, we will fill missing values between the msg and prev.
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
        self.max_queue_len = 100 #This is the max of expected num messages per dt over all time series (no need to change unless > 1000hz)

        #Printouts
        print('ONLINE CONVERTER:')
        print('Observation space:')
        print(self.observation_converters)
        print('ACtion space:')
        print(self.action_converters)

        self.init_queue()
        self.init_subscribers()

    def init_subscribers(self):
        self.subscribers = {}
        for k in self.observation_converters.keys():
            self.subscribers[k] = rospy.Subscriber(k, self.observation_converters[k].rosmsg_type(), self.handle_msg, callback_args=k)

        for k in self.action_converters.keys():
            self.subscribers[k] = rospy.Subscriber(k, self.action_converters[k].rosmsg_type(), self.handle_msg, callback_args=k)

    def init_queue(self):
        self.queue = {}
        self.times = {}
        self.curr_time = rospy.Time.now()
        for topic, cvt in self.observation_converters.items():
            if self.rates[topic] == self.dt:
                self.queue[topic] = None
            else:
                self.queue[topic] = [None] * self.max_queue_len
                self.times[topic] = [0.] * self.max_queue_len

        for topic, cvt in self.action_converters.items():
            self.queue[topic] = None

    def handle_msg(self, msg, topic):
        has_stamp = hasattr(msg, 'header') and msg.header.stamp.to_sec() > 1000.
        has_info = hasattr(msg, 'info') and msg.info.header.stamp.to_sec() > 1000.

        if self.use_stamps and (has_stamp or has_info):
            t = msg.header.stamp if has_stamp else msg.info.header.stamp
        else:
            t = rospy.Time.now()

        if self.rates[topic] == self.dt:
            self.queue[topic] = msg
        else:
            self.queue[topic] = self.queue[topic][1:] + [msg]
            self.times[topic] = self.times[topic][1:] + [t]

    def get_data(self):
        out = {
                'observation':{},
                'action':{}
        }

        for topic, cvt in self.observation_converters.items():
            if self.rates[topic] == self.dt:
                out['observation'][self.remap[topic]] = torch.tensor(cvt.ros_to_numpy(self.queue[topic])).float()
            else:
                """
                If time-series, look N steps backward from the last msg in buf.
                """
                msg_times = np.array([x.to_sec() for x in self.times[topic]])
                target_times = np.arange(msg_times[-1] - self.dt, msg_times[-1], self.rates[topic])
                dists = abs(np.expand_dims(target_times, 0) - np.expand_dims(msg_times, 1))
                msg_idxs = np.argmin(dists, axis=0)
                datas = [cvt.ros_to_numpy(self.queue[topic][i]) for i in msg_idxs]
                out['observation'][self.remap[topic]] = torch.stack([torch.tensor(x).float() for x in datas], dim=0)

        for topic, cvt in self.action_converters.items():
            data = torch.tensor(cvt.ros_to_numpy(self.queue[topic])).float()
            if len(data.shape) == 0:
                data = data.unsqueeze(0)
            out['action'][topic] = data

        if len(self.action_converters) > 0:
            out['action'] = torch.cat([v for v in out['action'].values()], axis=0)

        return out

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    parser = argparse.ArgumentParser()

    parser.add_argument('--config_spec', type=str, required=True, help='Path to the yaml file that contains the dataset spec.')
    parser.add_argument('--use_stamps', type=str2bool, required=False, default=True, help='Whether to use the time provided in the stamps or just the ros time')

    args = parser.parse_args()

    print('setting up...')
    rospy.init_node('ros_to_torch')
    rate = rospy.Rate(10)

    config_parser = ConfigParser()
    spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
    converter = OnlineConverter(spec, converters, remap, rates, args.use_stamps)

    print('waiting 1s for topics...')
    for i in range(10):
        rate.sleep()


    fig, axs = plt.subplots(2, 2)
    axs = axs.flatten()
    plt.show(block = False)

    i = 0
    while not rospy.is_shutdown():
        data = converter.get_data()
        print('____{}____'.format(i+1))
        print({k:v.shape for k,v in data['observation'].items()})
        print(data['action'].shape)

        for ax in axs:
            ax.cla()

        axs[0].imshow(data['observation']['image_rgb'].permute(1, 2, 0))
        plt.pause(1e-2)
        rate.sleep()
        i += 1
