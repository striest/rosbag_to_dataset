import sys
import os
import rosbag
import rospy
import argparse
import csv

def is_autonomous(bag, autonomy_topic):
    data = []
    topics = bag.get_type_and_topic_info()[1].keys()
    if not autonomy_topic in topics:
        print(f"=====\nERROR: {autonomy_topic} NOT in bag.")
        return None
    for topic, msg, t in bag.read_messages(topics=[autonomy_topic]):
        data.append(msg.data)
    print("Done reading bag!")
    is_autonomous = (False in data)
    print(f"Is bag autonomous? {is_autonomous}")

    return is_autonomous

def has_topics(bag, target_topics={}):
    target_topics_list = list(target_topics.keys())
    topics = bag.get_type_and_topic_info()[1].keys()
    for topic in topics:
        if topic in target_topics_list:
            msg_count = bag.get_message_count(topic)
            target_topics[topic] = msg_count
    return target_topics


def date_topics(dates_dict, target_topics):
    target_topics_list = list(target_topics.keys())
    header = ['Date'] + target_topics_list
    all_day_topics = []  # List of lists
    for date in dates_dict:
        date_topics = [True]*len(target_topics_list)
        date_list_bags = dates_dict[date]
        for ind_bag_dict in date_list_bags:
            for i, topic in enumerate(target_topics_list):
                if ind_bag_dict[topic] == 0:
                    date_topics[i] = False
        date_topics.insert(0, date)
        all_day_topics.append(date_topics)

    return all_day_topics, header


def main(args, target_topics):
    autonomy_topic = '/mux/intervention'

    with open(args.bag_list, 'r') as f:
        lines = f.readlines()
    bag_file_list = [line.strip() for line in lines]

    dates = [line.strip().split("/")[3] for line in lines]
    dates_dict = {date:[] for date in dates}

    # import pdb;pdb.set_trace()

    autonomous_list = []
    for bag_file in bag_file_list:
        bag = rosbag.Bag(bag_file)

        print(f"Reading bag: {bag_file}")
        is_bag_autonomous = is_autonomous(bag, autonomy_topic)
        autonomous_list.append(is_bag_autonomous)

        bag_topics = has_topics(bag, target_topics)

        for date in dates:
            if date in bag_file:
                dates_dict[date].append(bag_topics)
                break
    # import pdb;pdb.set_trace()  
    available_topics, topics_header = date_topics(dates_dict, target_topics)

    ## Write date_topics csv file
    with open(args.topics_output, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        writer.writerow(topics_header)

        # write multiple rows
        writer.writerows(available_topics)

    ## Write is_autonomous csv_file
    with open(args.autonomy_output, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        # write the header
        autonomy_header = ['bag_file', 'is_autonomous']
        writer.writerow(autonomy_header)

        # write multiple rows
        for i in range(len(bag_file_list)):
            row = [bag_file_list[i], autonomous_list[i]]
            writer.writerow(row)

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--bag_list', type=str, required=True, help='Path to the txt file that contains a list of the directories of all bags to be processed.')
    parser.add_argument('--autonomy_output', type=str, required=True, help='Path to the csv file that will store the output of whether a bag is autonomous or not.')
    parser.add_argument('--topics_output', type=str, required=True, help='Path to the csv file that will store the available topics for each date.')

    args = parser.parse_args()

    target_topics = {
        "/cmd": 0,
        "/cmd_mux/intervention": 0,
        "/controls": 0,	
        "/deep_cloud": 0,
        "/joy": 0,
        "/learned_costmap": 0,
        "/local_height_map_inflate": 0,
        "/local_rgb_map_inflate": 0,
        "/multisense/imu/imu_data": 0,
        "/multisense/left/image_rect": 0,
        "/multisense/left/image_color": 0,
        "/multisense/left/image_rect_color": 0,
        "/multisense/right/image_rect": 0,
        "/mux/intervention": 0,
        "/mux/joy": 0,
        "/novatel/imu/data": 0,
        "/odom": 0,
        "/odometry/filtered_odom": 0,
        "/tartanvo_odom": 0,
        "/tartanvo_pose": 0,
        "/tartanvo_transform": 0,
        "/velodyne_1/velodyne_points": 0,
        "/velodyne_2/velodyne_points": 0
    }

    main(args, target_topics)