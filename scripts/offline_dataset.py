import argparse
from copy import deepcopy
import numpy as np
from PIL import Image
import cv2
import rosbag
import os
import yaml
import shutil

from rosbag_to_dataset.util.bag_util import filter_bags, parse_dict
from rosbag_to_dataset.converter.converter import Converter



if __name__ == "__main__":

    ## 0. We take as input the following: directory where all the data is stored, filpeath of YAML file of which sensors we want to record and which type they are, directory where we want to store the dataset

    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', type=str, required=True, help='Path to the root directory containing all the subdirectories that contain bagfiles.')
    parser.add_argument('--topic_spec', type=str, required=True, help='Path to the yaml file that contains the mapping of data to topics that will be saved.')
    parser.add_argument('--dataset_dir', type=str, required=True, help='Path where the dataset will be stored')
    args = parser.parse_args()

    root_dir = args.root_dir
    topic_spec = args.topic_spec
    dataset_dir = args.dataset_dir
    # root_dir = "/home/mateo/Data/SARA/EvaluationSet"
    # topic_spec = "/home/mateo/Data/SARA/EvaluationSet/topic_spec.yaml"
    # dataset_dir = "/home/mateo/Data/SARA/TartanCost"

    annotations_dir = os.path.join(dataset_dir, 'Annotations')
    trajectories_dir = os.path.join(dataset_dir, 'Trajectories')

    if not os.path.exists(annotations_dir):
        os.makedirs(annotations_dir)

    if not os.path.exists(trajectories_dir):
        os.makedirs(trajectories_dir)

    required_topics = yaml.safe_load(open(topic_spec, 'r'))

    required_topic_dirs  = list(required_topics.keys())
    required_topic_names = [list(required_topics.values())[i]['name'] for i in range(len(required_topics))]
    required_topic_types = [list(required_topics.values())[i]['type'] for i in range(len(required_topics))]

    # Count how many trajectories already exist in the dataset and set 
    trajectory_count = len(os.listdir(trajectories_dir))

    bag_length = 5

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            bag_fp = os.path.join(subdir, file)
            if ".bag" in bag_fp:

                #Read in YAML file in directory if one exists and save labels from YAML file to later annotate specific trajectories with
                labels_fp = os.path.join(subdir, 'labels.yaml')
                with open(labels_fp, 'r') as f:
                    dir_labels = yaml.safe_load(f)
                
                labels = dir_labels['labels']
                label_names = dir_labels['label_names']

                #Check the duration of the rosbag. If it is shorter than 5s, ignore, if it is 5s, good, if it is greater than 5s, split it into 5s trajectories and get rid of the remainder. NOTE: this creates new bags in the filesystem of size 5.
                valid_bags = filter_bags(bag_fp, bag_length)

                for valid_bag_fp in valid_bags:
                    ignore_bag = False
                    print(f"Looking at bag: {valid_bag_fp}")

                    # Inside trajectory directory, make subdirectories for every key in required_topics. First, make sure that all of the required topics are recorded in the bag, that there is at least one message, and that they are of the right type

                    bag = rosbag.Bag(valid_bag_fp)
                    topics = list(bag.get_type_and_topic_info()[1].keys())
                    values = list(bag.get_type_and_topic_info()[1].values())

                    # Make sure all topics are in bag:
                    for i,req_topic in enumerate(required_topic_names):
                        if (req_topic not in topics):
                            ignore_bag = True
                            print(f"IGNORING BAG {valid_bag_fp}")
                            break

                    if ignore_bag:
                        continue

                    # Make trajectory directory e.g. 00000/. Only generate trajectory folder if we have all topics in the bag
                    dir_name = f"{trajectory_count:06}"
                    traj_dir = os.path.join(trajectories_dir, dir_name)
                    if not os.path.exists(traj_dir):
                        os.makedirs(traj_dir)

                    # Generate sensors dictionary which will collect data to create spec file for rosbag-to-dataset Converter interface

                    sensors = {}

                    for i,req_topic in enumerate(required_topic_names):
                        idx = topics.index(req_topic)
                        sensor_name = required_topic_dirs[i]
                        topic_name  = req_topic
                        topic_type  = values[idx][0]
                        topic_freq  = round(values[idx][3],2)
                        
                        sensors[sensor_name] = {"topic":topic_name, "type":topic_type, "freq":topic_freq}


                    # Generate spec file to be used with rosbag-to-dataset Converter interface for this specific bag, making use of frequency and required_topics as the remap names so that we can use these to access dictionary that will be returned by rosbag-to-dataset

                    spec, converters, remap, rates = parse_dict(sensors)

                    converter = Converter(spec, converters, remap, rates, use_stamps=False)

                    dataset = converter.convert_bag(bag, as_torch=False, zero_pose_init=True)


                    for i, req_dir in enumerate(required_topic_dirs):
                        sensor_dir = os.path.join(traj_dir, req_dir)
                        if not os.path.exists(sensor_dir):
                            os.makedirs(sensor_dir)
                        
                        # Gather specific sensor data and save in individual files
                        sensor_data = dataset["observation"][req_dir]

                        for k in range(sensor_data.shape[0]):

                            data_name = f"{k:06}"
                            # If image, save as png
                            if "image" in req_dir:
                                data_name += ".png"
                            # else, save as npy array
                            else:
                                data_name += ".npy"

                            data_fp = os.path.join(sensor_dir, data_name)
                            
                            data_point = sensor_data[k]

                            if "image" in req_dir:
                                # If image, and 3 dim, reshuffle so that order is (HxWxC)
                                if (data_point.shape[0] == 1 or data_point.shape[0] == 3):
                                    data_point = np.transpose(data_point, (1,2,0))

                                data_point = (data_point*255).astype(np.uint8)
                                if data_point.shape[2] == 1:
                                    im = Image.fromarray(data_point.squeeze())
                                    im.save(data_fp)
                                elif data_point.shape[2] == 3:
                                    data_point = cv2.cvtColor(data_point, cv2.COLOR_BGR2RGB)
                                    im = Image.fromarray(data_point)
                                    im.save(data_fp)
                            else:
                                np.save(data_fp, data_point)

                    # Store trajectory
                    numpy_dir = os.path.join(traj_dir, "numpy_trajectory")
                    if not os.path.exists(numpy_dir):
                            os.makedirs(numpy_dir)
                    numpy_fp = os.path.join(numpy_dir, "trajectory.npy")
                    np.save(numpy_fp, dataset)

                    # Store bag file
                    bagfile_dir = os.path.join(traj_dir, 'bag_file') 
                    if not os.path.exists(bagfile_dir):
                            os.makedirs(bagfile_dir)
                    bagfile_fp = os.path.join(bagfile_dir, "trajectory.bag")
                    shutil.copy2(valid_bag_fp, bagfile_fp)

                    # Set speed on label
                    bag_label = deepcopy(labels)

                    speeds = []

                    odom_shape = dataset['observation']['odom'].shape
                    for i in range(odom_shape[0]):
                        for j in range(odom_shape[1]):
                            speed = np.linalg.norm(dataset['observation']['odom'][i, j][7:10])
                            speeds.append(speed)


                    avg_speed = round(float(np.mean(speeds)),2)

                    if (avg_speed > 0.3) and (avg_speed <= 3):
                        bag_label[9] = 1
                    elif (avg_speed > 3) and (avg_speed <= 8):
                        bag_label[10] = 1
                    elif (avg_speed > 8):
                        bag_label[11] = 1
                    
                    # Create annotations
                    annotation = {
                        'directory': 'Trajectories/'+f"{trajectory_count:06}",
                        'sensors': sensors,
                        'labels': bag_label,
                        'label_names': label_names,
                        'average_speed': avg_speed
                    }

                    annotation_fp = os.path.join(annotations_dir, f'{trajectory_count:06}.yaml')

                    with open(annotation_fp, 'w') as outfile:
                        yaml.safe_dump(annotation, outfile, default_flow_style=False)


                    trajectory_count += 1
