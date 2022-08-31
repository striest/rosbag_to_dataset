from os.path import isfile, join, isdir
from os import listdir
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
    return trajfolders

def run(root_dir):
    # root_dir = '/home/parvm/RISS/affix_data/combined_sysid/experiment/only_test'
    # outfile = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2/trainframes.txt'

    single_trajlist = enumerate_trajs(root_dir)

    for i in range(len(single_trajlist)):

        # data_root_dir = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2'
        data_root_dir = join(root_dir,single_trajlist[i])
        outfile= join(data_root_dir,'frames.txt')
        # outfile = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2/trainframes.txt'


        f = open(outfile, 'w')

        trajlist = enumerate_trajs(data_root_dir)
        for trajdir in trajlist:
            trajindlist = process_traj(data_root_dir + '/' +trajdir,folderstr = 'image_left_color')
            for trajinds in trajindlist:
                f.write(trajdir)
                f.write(' ')
                f.write(str(len(trajinds)))
                f.write('\n')
                for ind in trajinds:
                    f.write(ind)
                    f.write('\n')
        f.close()

