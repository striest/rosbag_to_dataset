from os.path import isfile, join, isdir
from os import listdir
import random
# import numpy as np

# == Generate txt file for tartan dataset ===

def process_traj(trajdir, folderstr = 'image_left'):
    imglist = listdir(join(trajdir, folderstr))
    imglist = [ff for ff in imglist if ff[-3:]=='png']
    imglist.sort()
    imgnum = len(imglist)

    lastfileind = -1
    outlist = []
    framelist = []
    for k in range(imgnum):
        filename = imglist[k]
        framestr = filename.split('_')[0].split('.')[0]
        frameind = int(framestr)

        if frameind==lastfileind+1: # assume the index are continuous
            framelist.append(framestr)
        else:
            if len(framelist) > 0:
                outlist.append(framelist)
                framelist = []
        lastfileind = frameind

    if len(framelist) > 0:
        outlist.append(framelist)
        framelist = []
    print('Find {} trajs, traj len {}'.format(len(outlist), [len(ll) for ll in outlist]))

    return outlist 


def enumerate_trajs(data_root_dir):
    trajfolders = listdir(data_root_dir)
    trajfolders = [ee for ee in trajfolders if isdir(data_root_dir+'/'+ee)]
    trajfolders.sort()
    print('Detected {} trajs'.format(len(trajfolders)))
    num_traj = len(trajfolders)
    num_train = int(0.8 * num_traj)
    train_trajfolders = random.sample(trajfolders,num_train)
    train_trajfolders.sort()
    eval_trajfolders = list(set(trajfolders).symmetric_difference(set(train_trajfolders)))
    eval_trajfolders.sort()
    
    


    return trajfolders, train_trajfolders, eval_trajfolders

# data_root_dir = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2'
# outfile = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2/trainframes.txt'

data_root_dir = '/project/learningphysics/parv_dataset/affix_sys_id/extracted_tartan_bags'
train_outfile = '/project/learningphysics/parv_dataset/affix_sys_id/extracted_tartan_bags/trainframes.txt'
eval_outfile = '/project/learningphysics/parv_dataset/affix_sys_id/extracted_tartan_bags/evalframes.txt'

train_eval_split = 0.8
train_f = open(train_outfile, 'w')
eval_f = open(eval_outfile, 'w')


_, train_trajlist, eval_trajlist = enumerate_trajs(data_root_dir)
for trajdir in train_trajlist:
    trajindlist = process_traj(data_root_dir + '/' +trajdir,folderstr = 'image_left_color')
    for trajinds in trajindlist:
        train_f.write(trajdir)
        train_f.write(' ')
        train_f.write(str(len(trajinds)))
        train_f.write('\n')
        for ind in trajinds:
            train_f.write(ind)
            train_f.write('\n')
train_f.close()

for trajdir in eval_trajlist:
    trajindlist = process_traj(data_root_dir + '/' +trajdir,folderstr = 'image_left_color')
    for trajinds in trajindlist:
        eval_f.write(trajdir)
        eval_f.write(' ')
        eval_f.write(str(len(trajinds)))
        eval_f.write('\n')
        for ind in trajinds:
            eval_f.write(ind)
            eval_f.write('\n')
eval_f.close()

