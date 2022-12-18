import numpy as np
import argparse
from os.path import join, isdir
import os
from tqdm import tqdm 
from os import listdir

def merge(x, file):
    trajfolders = file.readline()
    trajfolders = trajfolders.split(', ')
    trajfolders = [x for x in trajfolders if x.strip()!='']
    trajfolders.sort()
    x.extend(trajfolders) 
    return x


if __name__ == '__main__':
    root_fp = '/home/parvm/physics_atv_ws/src/learning/rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/data_partition/new'
    train_file = open(join(root_fp,'train.txt'))
    eval_file = open(join(root_fp,'eval.txt'))
    res = []
    res = merge(res,train_file)
    res = merge(res,eval_file)
    res = [*set(res)]
    res.sort()
    f = open("/home/parvm/physics_atv_ws/src/learning/rosbag_to_dataset/src/rosbag_to_dataset/post_processing/dataloader/data_partition/new/all.txt",'w')
    for x in res:
        f.write(f"{x}, ")
    f.close()

    
    print('Detected {} trajs'.format(len(res)))