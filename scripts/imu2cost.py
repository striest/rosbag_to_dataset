import argparse
import numpy as np

from scipy.signal import welch
from scipy.integrate import simps

import os
# import matplotlib.pyplot as plt

def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Taken from: https://raphaelvallat.com/bandpower.html

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        # nperseg = (2 / low) * sf
        nperseg = None

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simps(psd[idx_band], dx=freq_res)

    if relative:
        bp /= simps(psd, dx=freq_res)
    return bp

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, required=True, help='Path to the directory that contains the data split up into trajectories.')
    parser.add_argument('--cost_folder', type=str, default="cost", help='Path to the directory that contains the data split up into trajectories.')
    args = parser.parse_args()

    print("Input arguments are the following: ")
    print(f"data_dir: {args.data_dir}")

    trajectories_dir = args.data_dir
    cost_folder = args.cost_folder
    traj_dirs = list(filter(os.path.isdir, [os.path.join(trajectories_dir,x) for x in sorted(os.listdir(trajectories_dir))]))

    min_freq = 0
    max_freq = 10

    imu_freq = 100
    num_seconds = 1
    datalen = int(imu_freq * num_seconds)
    imu_offset_seconds = -0.5
    imu_offset_frame = int(imu_freq * imu_offset_seconds)
    pad_val = -9.8
    delta_w = 0.0
    cost_norm = 15.0

    for i, d in enumerate(traj_dirs):
        if "preview" in d:
            continue
        print("=====")
        print(f"Labeling directory {d}")

        ## Load IMU data
        imu_dir = os.path.join(d, "imu")
        imu_fp = os.path.join(imu_dir, "imu.npy")
        imu_data = np.load(imu_fp)

        ## use acc-z and pad the seq
        acc_z = imu_data[:, 3]
        if imu_offset_frame < 0:
            pad_arr = np.array([pad_val]*(-imu_offset_frame), dtype=np.float32)
            acc_z = np.concatenate((pad_arr, acc_z),axis=0)
        elif imu_offset_frame > 0:
            pad_arr = np.array([pad_val]*(imu_offset_frame), dtype=np.float32)
            acc_z = np.concatenate((acc_z, pad_arr), axis=0)

        ## Load IMU timestamps file
        imu_txt = os.path.join(imu_dir, "timestamps.txt")
        imu_times = np.loadtxt(imu_txt)

        ## Load image_left timestamps to use for reference for cost labeling
        image_txt = os.path.join(d, "image_left", "timestamps.txt")
        image_times = np.loadtxt(image_txt)


        ## Initialize buffer
        imu_to_img_freq = imu_data.shape[0]//image_times.shape[0]

        ## Initialize cost array
        bp_list = []

        start_imu_idx = 0
        for i, img_time in enumerate(image_times):
            start_imu_idx = i * imu_to_img_freq
            end_imu_idx = start_imu_idx + datalen
            imu_segment = acc_z[start_imu_idx:end_imu_idx] # z-acc

            # Calculate cost for buffer.data
            bp = bandpower(imu_segment, imu_freq, band=[min_freq, max_freq], window_sec=num_seconds)
            bp_list.append(bp)

        bps = np.array(bp_list)
        bp_delta = bps[1:]-bps[:-1]
        costs = bps.copy()
        costs[1:] = costs[1:] + delta_w * bp_delta
        costs = np.clip(costs/cost_norm, 0, 1)

        # plt.plot(bps/cost_norm, '.-')
        # plt.plot(costs,'x-')
        # plt.grid()
        # plt.show()

        # import ipdb;ipdb.set_trace()
        # Write cost_vals and cost_times to own folder in the trajectory
        cost_dir = os.path.join(d, cost_folder)
        if not os.path.exists(cost_dir):
            os.makedirs(cost_dir)
        
        cost_val_fp = os.path.join(cost_dir, "cost.npy")
        cost_times_fp = os.path.join(cost_dir, "timestamps.txt")

        np.save(cost_val_fp, np.array(costs))
        np.savetxt(cost_times_fp, np.array(image_times))



        # datalen = 100
        # datastride = 50
        # # labelstrs=['x','y','z']
        # # plt.subplot(133)
        # print(imu_data.shape[0])
        # for w in range(0,imu_data.shape[0],datastride):
        #     imu_seg = imu_data[w:w+datalen,3:]
        #     print(w)
        #     plt.subplot(221)
        #     plt.plot(imu_seg[:,-2:])
        #     plt.grid()
        #     plt.ylim(-12,12)
        #     plt.subplot(222)
        #     plt.plot(imu_seg[:,0])
        #     plt.grid()
        #     plt.ylim(-22,2)
        #     scores = []
        #     for k in range(3):
        #         imuacc_fft = fftpack.fft(imu_seg[:,k])
        #         imuacc_power = np.abs(imuacc_fft) **2
        #         imuacc_freq = fftpack.fftfreq(imuacc_fft.size, d=0.01)
        #         # plt.yscale('log')
        #         plt.subplot(223)
        #         plt.plot(imuacc_freq[3:int(imuacc_freq.shape[0]/2)], imuacc_power[3:int(imuacc_freq.shape[0]/2)])
        #         plt.grid()
        #         freqs, psd = welch(imu_seg[:,k], 100, nperseg=datalen)
        #         plt.subplot(224)
        #         plt.plot(freqs, psd)
        #         plt.grid()
        #         plt.ylim(0,1.2)

        #         idx_band = np.logical_and(freqs >= 0, freqs <= 10)
        #         bp = simps(psd[idx_band], dx=freqs[1]-freqs[0])
        #         scores.append(bp)
        #         # import ipdb;ipdb.set_trace()
        #     print(scores)
        #     plt.show()
