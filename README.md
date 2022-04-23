# rosbag_to_dataset
A library that converts rosbags into datasets for intervention learning, model-based RL, and online learning.

# Usage

## Spec Format
The user is required to provide the converters a YAML file that describes the format of the robot data to convert and run learning on. This YAML file can be proken into three components:

    1. main_topic: other topics will be temporally aligned with the main_topic, according to its time stamps
    2. Timestep (dt): the frequency of the extracted data
    3. data: topics of interests
        - <topic-name>:
            * type: <topic-type>
            * folder: <output-folder>
            * N_per_step: <higher-freq-ratio>
            * options:
                <option-1>: <value-1>

Examples of this yaml format can be found in ```specs/sample_tartandrive.yaml```.

### Modalities
This library currently supports the following modalities (more will be added as needed by our experiments). Note that we generally want our messages to be timestamped.

    - AckermannDriveStamped
    - BoolStamped
    - CompressedImage
    - Float64
    - Image
    - Imu
    - Odometry
    - PointCloud
    - PoseStamped
    - TwistStamped
    - Vector3


## Output format
Data of specified modalities is stored in trajectory folders with the following format:

```
ROOT
|
--- TRAJ_NAME_0                 # trajectory folder
|       |
|       +--- image_left         # 000000.png - 000xxx.png 
|       +--- image_right        # 000000.png - 000xxx.png 
|       +--- image_left_color   # 000000.png - 000xxx.png
|       +--- depth_left         # 000000.npy - 000xxx.npy
|       +--- height_map         # 000000.npy - 000xxx.npy
|       +--- rgb_map            # 000000.npy - 000xxx.npy
|       +--- cmd                # twist.npy
|       +--- imu                # imu.npy
|       +--- odom               # odometry.npy
|       +--- tartanvo_odom      # motions.npy, poses.npy
|       +--- points_left        # 000000.npy - 000xxx.npy
|       
+-- TRAJ_NAME_1
.
.
|
+-- TRAJ_NAME_N
```

Note that inside each modality folder, there is a `timestamps.txt` file, specifying the corresponding timestamps of each frame. 

## Offline

To generate offline data, do the following:

```
python scripts/tartandrive_dataset.py --bag_fp <ARG> --config_spec <ARG> --save_to <ARG>
```

### Args
    --config_spec: path to the configuration file
    --bag_fp: path to bagfile
    --save_to: new dir to save data files in

### Output
This will produce a torch file for each bag in the directory.

# Code Notes
This code is derived from the Tartan_cost branch. The main feature is to store the data into files instead of returning as numpy array. Another key change is the way of aligning the timestamps. A main-topic is defined in the spec file, other topics are aligned with the main-topic by finding the msg with the closest timestamps. 

