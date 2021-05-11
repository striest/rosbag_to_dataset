import yaml
import gym
import numpy as np

from collections import OrderedDict

from rosbag_to_dataset.dtypes.float64 import Float64Convert
from rosbag_to_dataset.dtypes.odometry import OdometryConvert
from rosbag_to_dataset.dtypes.image import ImageConvert
from rosbag_to_dataset.dtypes.ackermann_drive import AckermannDriveConvert

class ConfigParser:
    """
    Class that reads in the spec of the rosbag ot convert to data.
    Expects input as a yaml file that generally looks like the following (currently WIP, subject to change).

    observation:
        topic:
            type:<one of the supported types>
            <option>:<value>
            ...
    action:
        topic:
            type:<one of the supported types, but probably not images>
            <option>:<value>
            ...

    I'm likely going to restrict actions to vectors, meaning that there are probably only a few dtypes that actions can actually be.
    Also, we're going to squish actions into a single vector, but leave states as dictionaries. End result of this class is the sizes of the various inputs.
    I.e. generate a pseudo-env that works with my replay buffers (a class/dict w/ action_space, observation_space.).
    """
    def __init__(self):
        pass

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    def parse(self, spec):
        obs_dict = {}
        obs_converters = OrderedDict()
        for k,v in spec['observation'].items():
            dtype = self.dtype_convert[spec['observation'][k]['type']]
            converter = dtype(**spec['observation'][k].get('options', {}))
            obs_shape = converter.N()
            obs_dict[k] = gym.spaces.Box(low = np.ones(obs_shape) * -float('inf'), high = np.ones(obs_shape) * float('inf'))
            obs_converters[k] = converter

        obs_space = gym.spaces.Dict(obs_dict)

        act_dim = 0
        act_converters = OrderedDict()
        for k,v in spec['action'].items():
            dtype = self.dtype_convert[spec['action'][k]['type']]
            converter = dtype(**spec['action'][k].get('options', {}))
            act_dim += converter.N()
            act_converters[k] = converter

        act_space = gym.spaces.Box(low = -np.ones(act_dim), high = np.ones(act_dim))

        converters = {
            'observation':obs_converters,
            'action':act_converters
        }

        return ParseObject(obs_space, act_space, spec['dt']), converters

    dtype_convert = {
        "Float64":Float64Convert,
        "Odometry":OdometryConvert,
        "Image":ImageConvert,
        "AckermannDrive":AckermannDriveConvert
    }

class ParseObject:
    """
    Basically a dummy class that has an observation_space and action_space field.
    """
    def __init__(self, observation_space, action_space, dt):
        self.observation_space = observation_space
        self.action_space = action_space
        self.dt = dt

if __name__ == "__main__":
    fp = open('sample.yaml')
    d = yaml.safe_load(fp)
    print(d)
    parser = ConfigParser()
    x, p = parser.parse(d)
    print(x.observation_space)
    print(x.action_space)
    print(p)
