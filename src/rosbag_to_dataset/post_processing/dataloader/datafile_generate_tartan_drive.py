from os.path import isfile, join, isdir
from os import listdir
from datafile_generate import process_traj
# == Generate txt file for tartan dataset from csv (in tartan drive repo) 

def enumerate_trajs_data_partition(file):
    trajfolders = file.readline()
    trajfolders = trajfolders.split(', ')
    trajfolders = [x for x in trajfolders if x.strip()!='']
    trajfolders.sort()
    print('Detected {} trajs'.format(len(trajfolders)))
    return trajfolders


def create_file(data_root_dir,data_file,outfolder,outfile_name):
    f = open(join(outfolder,outfile_name), 'w')
    trajlist = enumerate_trajs_data_partition(data_file)
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

data_root_dir = '/project/learningphysics/tartandrive_trajs'
outfolder = '/project/learningphysics/parv_dataset/affix_sys_id/baseline/tartan_drive_split'
train_file = open('data_partition/new/train.txt')
test_file = open('data_partition/new/eval.txt')

create_file(data_root_dir,train_file,outfolder,'trainframes.txt')
create_file(data_root_dir,test_file,outfolder,'evalframes.txt')
