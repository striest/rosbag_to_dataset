# rosbag_to_dataset
A library that converts rosbags into datasets for intervention learning, model-based RL, and online learning.

# Usage

## Spec Format
The user is required to provide the converters a YAML file that describes the format of the robot data to convert and run learning on. This YAML file can be proken into three components:

    1. Observations
    2. Actions
    3. Timestep (dt)

Examples of this yaml format can be found in ```rosbag_to_dataset/specs```.

### Modalities
This library currently supports the following modalities (more will be added as needed by our experiments). Note that we generally want our messages to be timestamped.

    - AckermannDriveStamped
    - BoolStamped
    - CompressedImage
    - Float64
    - Image
    - Imu
    - Odometry
    - PoseStamped
    - TwistStamped

### General Format
Generally speaking, the format of a config file is the following:

```
observation:
    <topic>:
        type: <message_type>
        remap: <remap name>
        N_per_step: <N>
        options:
            <type-specific options>
    ...

action:
    <same as observation>

dt: <dt>
```
```N_per_step``` is an argument to allow for higher-frequency messages (e.g. IMU) to get multiple measurements per dt. For most topics this value can be omitted.

## Data format
Data is returned as a dictionary of torch tensors with the following format:

```
{
    'observation':
    {
        <remap_name>:<T x N tensor of data for that topic>
    },
    'action': <T x N tensor of action data>
    'reward': <T x 1 float tensor>,
    'terminal': <T x 1 bool tensor>,
    'next_observation:<Same as observation, but shifted one step forward>
}
```

Note that ```observation``` is a dictionary, while ```action``` is a tensor. If there are multiple action topics, they will be concatenated (alphabetically by topic name). Multiple action topics are not recommended.

## Offline

To generate offline data, do the following:

```
python3 scripts/multi_convert_bag.py --config_spec <ARG> --bag_dir <ARG> --save_to <ARG> --use_stamps <ARG> --torch <ARG> --zero_pose_init <ARG>
```

### Args
    --config_spec: path to the configuration file
    --bag_dir: path to directory containing bags
    --save_to: new dir to save data files in
    --use_stamps: whether to use the rosbag timestamp or to use the timestamp in the message. If true, messages without stamps will default to the rosbag time. True is recommended.
    --torch: If true, results will be torch, else numpy
    --zero_pose_init: This will zero-out the position of the topic remapped to "state"

### Output
This will produce a torch file for each bag in the directory.

## Online
This code can also be instantiated as an object in a python script. 

```
from rosbag_to_dataset.converter.online_converter import OnlineConverter
from rosbag_to_dataset.config_parser.config_parser import ConfigParser

spec, converters, remap, rates = config_parser.parse_from_fp(args.config_spec)
converter = OnlineConverter(spec, converters, remap, rates, args.use_stamps)
```

This forms a converter that stores a buffer of the latest message(s) from ROS, and can be queried to give data via:

```
converter.get_data()
```

This will be in the same format as the offline data but with T=1.