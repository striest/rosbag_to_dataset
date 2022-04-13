import rosbag
import os
import math
import rospy
from copy import deepcopy

from rosbag_to_dataset.config_parser.config_parser import ConfigParser

sensor_specs = {
    'sensor_msgs/Image': {
        'type': 'Image',
        'remap': None,
        'N_per_step': None,
        'options': {
            'output_resolution': [1024, 544], #[544, 1024],
            'nchannels': None
        }        
    },
    'sensor_msgs/Imu': {
        'type': 'Imu',
        'remap': None,
        'N_per_step': None,
        'options': {
            'angular_velocity': True,
            'linear_acceleration': True
        }
    },
    'nav_msgs/Odometry': {
        'type': 'Odometry',
        'remap': None,
        'N_per_step': None,
        'options': {
            'use_vel': True
        }
    },
    # 'sensor_msgs/PointCloud2': {
    #     'type': 'PointCloud',
    #     'remap': None,
    #     'N_per_step': None,
    #     'options': {
    #         'fields': ['x', 'y', 'z'],
    #         'max_num_points': 10000
    #     }
    # }
}


def filter_bags(bag_fp, length=5, epsilon=0.01):
    '''Returns bag objects of input length in seconds.

    Args:
        bag_fp:
            Filepath to bag to be filtered
        length:
            Length in seconds of output bag files
        epsilon:
            Error admissible in length of input bags
    
    Returns:
        valid_bags:
            List of bag filepaths of size 5s
    '''

    valid_bags = []

    input_bag = rosbag.Bag(bag_fp)
    start_time = input_bag.get_start_time()
    end_time   = input_bag.get_end_time()
    duration   = end_time-start_time 

    num_output_bags = math.floor(duration/(length-epsilon))

    bag_starts = [rospy.Time.from_sec(i*length+start_time) for i in range(num_output_bags)]
    bag_ends   = [rospy.Time.from_sec((i+1)*length+start_time) for i in range(num_output_bags)]

    valid_bags = [f"{bag_fp[:-4]}_{i}.bag" for i in range(num_output_bags)]


    # Record all files in directory to make sure you don't split bags that have been already split
    directory, _ = os.path.split(bag_fp)
    files_in_dir = []

    for root, dirs, files in os.walk(directory):
        files_in_dir.extend([os.path.join(directory, file) for file in files])

    for i in range(num_output_bags):
        if valid_bags[i] in files_in_dir:
            continue
        with rosbag.Bag(valid_bags[i], 'w') as outbag:
            for topic, msg, t in input_bag.read_messages(start_time=bag_starts[i], end_time=bag_ends[i]):
                outbag.write(topic, msg, t)

    
    return valid_bags


def parse_dict(sensors_dict):
    '''Returns a dictionary of the same format as a spec from a YAML file 
    
    Args:
        sensors_dict:
            Dictionary of sensors
    '''

    # Get camera frequency to set this as the dt
    min_freq = 100000
    min_freq_sensor = None
    for k,v in sensors_dict.items():
        freq = v['freq']
        if freq < min_freq:
            min_freq = freq
            min_freq_sensor = k
    min_freq = round(min_freq)

    print(f"Sensor with min freq is: {min_freq_sensor}: {min_freq} Hz")
        
    dt = 1/10 #1/min_freq
    yaml_dict = {}
    yaml_dict["observation"] = {}
    yaml_dict["dt"] = dt

    max_freq = 100

    # import pdb;pdb.set_trace()
    for k,v in sensors_dict.items():
        sensor_type = v['type']
        print(f"sensor_type: {sensor_type}")
        spec_dict = deepcopy(sensor_specs[sensor_type])

        spec_dict['remap'] = k
        spec_dict['N_per_step'] = round(min(v['freq'], max_freq)*dt)

        # Special cases
        if "Image" in sensor_type:
            del spec_dict['N_per_step']
            if 'color' in v['topic']:
                spec_dict['options']['nchannels'] = 3
            else:
                spec_dict['options']['nchannels'] = 1
        
        yaml_dict["observation"][v['topic']] = spec_dict

    print(yaml_dict)

    # import pdb;pdb.set_trace()

    config_parser = ConfigParser()

    spec, converters, remap, rates = config_parser.parse(yaml_dict)


    return spec, converters, remap, rates
    

if __name__ == "__main__":
    bag_fp = "/home/mateo/Data/SARA/EvaluationSet/up_slope_bumpy/up_slope_bumpy1.bag"
    valid_bags = filter_bags(bag_fp)
    print(valid_bags)