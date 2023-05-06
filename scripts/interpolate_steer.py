import argparse
from os import listdir
from os.path import join,isdir
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
from rosbag_to_dataset.util.os_util import maybe_mkdir

if __name__ == '__main__':

    '''
    This script is useful to correctly interpolate steer from dt = 0.16 to dt = 0.1
    Here we assume source_fp has extracted delta at dt = 0.16 using normal extraction method - see readme 
    Then we allign the delta with odom timestamps with three cases
    1.If start of delta > start of odom : then pad final result at beginning with first delta element
    2.If end of delta < end of odom : then pad final result at endwith last delta element
    3. In case of overlap - use SciPy to interpolate source delta to dt = 0.1 (st for odom timestamps)
    '''

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_fp', type=str, required=True, help='Path to the source directory')
    parser.add_argument('--save_fp', type=str, required=True, help='Path to the destination trajectory')

    args = parser.parse_args()

    source_fp = args.source_fp
    save_fp = args.save_fp

    traj_folders = [x for x in listdir(source_fp) if isdir(join(source_fp,x))]
    for traj_name in traj_folders:
        source_traj_fp = join(source_fp,traj_name)
        dest_traj_fp = join(save_fp,traj_name)
        
        if isdir(join(dest_traj_fp,'delta')):
            continue
        source_delta = np.load(join(source_traj_fp,'delta','float.npy'))
        delta_timestamps = np.loadtxt(join(source_traj_fp,'delta','timestamps.txt'))

        dest_timestamps = np.loadtxt(join(dest_traj_fp,'odom','timestamps.txt'))
        interpolated_delta = scipy.interpolate.interp1d(delta_timestamps, source_delta)

        missed_first_timestamps = dest_timestamps[dest_timestamps < delta_timestamps[0]]
        missed_last_timestamps = dest_timestamps[dest_timestamps > delta_timestamps[-1]]

        missed_first_delta = np.full((len(missed_first_timestamps)), source_delta[0])
        missed_last_delta = np.full((len(missed_last_timestamps)), source_delta[-1])

        overlap_idx = np.logical_and(dest_timestamps >= delta_timestamps[0], dest_timestamps <= delta_timestamps[-1])

        final_delta = np.concatenate((missed_first_delta,interpolated_delta(dest_timestamps[overlap_idx]),missed_last_delta)).reshape((-1,1))
        maybe_mkdir(join(dest_traj_fp,'delta'))
        np.save(join(dest_traj_fp,'delta','float.npy'),final_delta)
        np.savetxt(join(dest_traj_fp,'delta','timestamps.txt'),dest_timestamps)
