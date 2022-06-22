sudo pip3 install --no-cache-dir ~/wheeledsim_rl/ -U
sudo pip3 install --no-cache-dir ~/rosbag_to_dataset_parallel/ -U
cd rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/
python3 test_dataloader_caching.py --config_fp reconstruction.yaml