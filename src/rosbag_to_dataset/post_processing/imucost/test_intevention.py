# For tartan_drive there are a frames at the end driving towards an obstacles
import numpy as np
from os.path import join
import matplotlib.pyplot as plt
import argparse
import os
from os.path import isfile, join

def find_slow_stop_pattern(velx, thresh=0.1):
    vel_cum5 = (velx + np.append(velx[1:],[0]) + np.append(velx[2:],[0,0]) + np.append(velx[3:],[0,0,0]) + np.append(velx[4:],[0,0,0,0]))/5
    vel0 = vel_cum5 < thresh
    return vel0

def find_intevention(control, velx):
    '''
    intevention: 1. look at the brake signal
                 2. look at the velx if it come to zero after the brake
    '''
    print(control.shape)
    control = control[:,1] > 200
    vel0 = find_slow_stop_pattern(velx)
    k = 0
    interventionlist = []
    while control[k]:  # discard the initial interventions
        k+=1
        if k >= len(control):
            break
        else:
            continue

    while k < len(control):
        while not control[k]:
            k += 1
            if k >= len(control):
                break
            else:
                continue
        # find one intervention
        if k < len(control):
            #check if the vehicle stops after intevention
            if True in vel0[k:k+50]:
                interventionlist.append(k)
            
            while control[k]:  # discard the continuous interventions
                k+=1
                if k >= len(control):
                    break
                else:
                    continue

    print(interventionlist)
    return interventionlist
    # # plt.subplot(121)
    # plt.plot(control[:,0])
    # # plt.subplot(122)
    # # plt.plot(control[:,1])
    # plt.show()
    # # intervention_200
    # # shock_travel
    # # shock_travel_20
    # # costmap.process(base_dir, 'costmap')

def motion2velx(motion):
    '''
    odom: N x 13 numpy array (x, y, z, rx, ry, rz, rw, vx, vy, xz, vrx, vry, vrz) in the global frame
    res: (forward vel, yaw)
    '''
    # import ipdb;ipdb.set_trace()
    t = 0.1 # the interval between two frames
    return motion[:,0] / t    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--bag_list', type=str, default="", help='Path to the bag file to get data from')
    args = parser.parse_args()

    print("Input arguments are the following: ")
    print(f"data_dir: {args.data_dir}")

    trajectories_dir = args.data_dir
    intervention_folder = 'intervention' 

    # get the velocity to see 
    odom_folder = 'tartanvo_odom'

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

        # find unique values of the folder
        outfolderlist = [os.path.join(trajectories_dir, bb[1]) for bb in bagfilelist]
        traj_dirs = set(outfolderlist)
        print('Find {} trajectories'.format(len(traj_dirs)))
    else:
        traj_dirs = list(filter(os.path.isdir, [os.path.join(trajectories_dir,x) for x in sorted(os.listdir(trajectories_dir))]))
        print('Find {} trajectories under the data dir {}'.format(len(traj_dirs), trajectories_dir))

    # base_dir = '/home/wenshan/tmp/arl_data/full_trajs/20210826_61' # 20210826_61 # 20220531_lowvel_0
    # control = np.load(join(base_dir, intervention_folder ,'control.npy'))

    # find_intevention(control)

    for i, d in enumerate(traj_dirs):
        if "preview" in d:
            continue
        print("=====")
        print(f"Intervention directory {d}")

        ## Load IMU data
        intervention_dir = join(d, intervention_folder)
        intervention_fp = join(intervention_dir, "control.npy")
        
        odom_dir = join(d, odom_folder)
        odom_fp = join(odom_dir, 'motions.npy')

        if not isfile(intervention_fp):
            print('Missing intervention file', intervention_fp)
            continue

        if not isfile(odom_fp):
            print('Missing odom file', odom_fp)
            continue

        intervention_data = np.load(intervention_fp)
        motion_data = np.load(odom_fp)
        vel_x = motion2velx(motion_data)

        find_intevention(intervention_data, vel_x)

    #     ## use acc-z and pad the seq
    #     acc_z = imu_data[:, 3]
    #     if imu_offset_frame < 0:
    #         pad_arr = np.array([pad_val]*(-imu_offset_frame), dtype=np.float32)
    #         acc_z = np.concatenate((pad_arr, acc_z),axis=0)
    #     elif imu_offset_frame > 0:
    #         pad_arr = np.array([pad_val]*(imu_offset_frame), dtype=np.float32)
    #         acc_z = np.concatenate((acc_z, pad_arr), axis=0)

    #     ## Load IMU timestamps file
    #     imu_txt = os.path.join(imu_dir, "timestamps.txt")
    #     imu_times = np.loadtxt(imu_txt)

    #     ## Load image_left timestamps to use for reference for cost labeling
    #     image_txt = os.path.join(d, "image_left", "timestamps.txt")
    #     image_times = np.loadtxt(image_txt)


    # #     ## Initialize buffer
    # #     imu_to_img_freq = imu_data.shape[0]//image_times.shape[0]

    # #     ## Initialize cost array
    # #     bp_list = []

    # #     start_imu_idx = 0
    # #     for i, img_time in enumerate(image_times):
    # #         start_imu_idx = i * imu_to_img_freq
    # #         end_imu_idx = start_imu_idx + datalen
    # #         imu_segment = acc_z[start_imu_idx:end_imu_idx] # z-acc

    # #         # Calculate cost for buffer.data
    # #         bp = bandpower(imu_segment, imu_freq, band=[min_freq, max_freq], window_sec=num_seconds)
    # #         bp_list.append(bp)

    # #     bps = np.array(bp_list)
    # #     bp_delta = bps[1:]-bps[:-1]
    # #     costs = bps.copy()
    # #     costs[1:] = costs[1:] + delta_w * bp_delta
    # #     costs = np.clip(costs/cost_norm, 0, 1)

    # #     # plt.plot(bps/cost_norm, '.-')
    # #     # plt.plot(costs,'x-')
    # #     # plt.grid()
    # #     # plt.show()

    # #     # import ipdb;ipdb.set_trace()
    # #     # Write cost_vals and cost_times to own folder in the trajectory
    # #     cost_dir = os.path.join(d, cost_folder)
    # #     if not os.path.exists(cost_dir):
    # #         os.makedirs(cost_dir)
        
    # #     cost_val_fp = os.path.join(cost_dir, "cost.npy")
    # #     cost_times_fp = os.path.join(cost_dir, "timestamps.txt")

    # #     np.save(cost_val_fp, np.array(costs))
    # #     np.savetxt(cost_times_fp, np.array(image_times))
