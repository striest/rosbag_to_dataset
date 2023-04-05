#!/bin/bash

cd ../src

python gen_dataset.py \
--data_dir /project/learningphysics/2022_traj \
--pts_subdir points_left \
--motion_subpath tartanvo_odom/motions.npy \
--pose_subpath odom/odometry.npy \
--max_n_frames 1500 \
--out_dir /project/learningphysics/saich/future_fusion