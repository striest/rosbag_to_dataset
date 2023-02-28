from os.path import isfile, join, isdir
from os import listdir
from tqdm import tqdm
from datafile_generate import process_traj, enumerate_trajs
from rosbag_to_dataset.util.os_util import maybe_mkdir
def run(root_dir,frames_root_dir=None):
    # root_dir = '/home/parvm/RISS/affix_data/combined_sysid/experiment/only_test'
    # outfile = '/home/parvm/RISS/affix_data/combined_sysid/experiment/rosbag_sync_test/only_traj_2/trainframes.txt'

    single_trajlist = enumerate_trajs(root_dir)

    for i in tqdm(range(len(single_trajlist))):

        data_root_dir = join(root_dir,single_trajlist[i])
        if frames_root_dir is None:
            outfile= join(data_root_dir,'frames.txt')
        else:
            frame_dir = join(frames_root_dir,single_trajlist[i])
            maybe_mkdir(frame_dir,force=False)
            outfile = join(frame_dir,'frames.txt')

        f = open(outfile, 'w')

        # trajlist = enumerate_trajs(data_root_dir)
        # for trajdir in trajlist:
        trajindlist = process_traj(data_root_dir,folderstr = 'image_left_color')
        for trajinds in trajindlist:
            f.write(single_trajlist[i])
            f.write(' ')
            f.write(str(len(trajinds)))
            f.write('\n')
            for ind in trajinds:
                f.write(ind)
                f.write('\n')
        f.close()

if __name__ == '__main__':
    root_dir = '/home/abyss/parv_RISS/data/og_wenshan_extracted_traj'
    frames_root_dir = '/home/abyss/parv_RISS/data/frames_og_wenshan_extracted_traj'
    run(root_dir,frames_root_dir)
