import argparse

from os import mkdir, listdir
from os.path import isfile, isdir, join, split
from rosbag_to_dataset.post_processing.imucost.cost2gtcostmap import GTCostMapNode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--output_dir', type=str, default="", help='Path to the directory that the costmap file output to.')
    parser.add_argument('--bag_list', type=str, default="", help='Path to the bag file to get data from')
    args = parser.parse_args()

    print("Input arguments are the following: ")
    print(f"data_dir: {args.data_dir}")

    trajectories_dir = args.data_dir
    if args.output_dir == "":
        output_dir = trajectories_dir
    else:
        output_dir = args.output_dir
        assert isdir(output_dir)

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

        # find unique values of the folder
        outfolderlist = [join(trajectories_dir, bb[1]) for bb in bagfilelist]
        traj_dirs = set(outfolderlist)
        print('Find {} trajectories'.format(len(traj_dirs)))
    else:
        traj_dirs = list(filter(isdir, [join(trajectories_dir,x) for x in sorted(listdir(trajectories_dir))]))
        print('Find {} trajectories under the data dir {}'.format(len(traj_dirs), trajectories_dir))

    traj_dirs = list(traj_dirs)
    traj_dirs.sort()
    costmap = GTCostMapNode()

    for i, d in enumerate(traj_dirs):
        print("=====")
        print(f"Labeling directory {d}")
        traj_output_dir = join(output_dir, split(d)[-1])
        if not isdir(traj_output_dir):
            mkdir(traj_output_dir)
        print(f"Output to {traj_output_dir}")
        ## Load IMU data
        costmap.process(d, traj_output_dir, 'costmap')
