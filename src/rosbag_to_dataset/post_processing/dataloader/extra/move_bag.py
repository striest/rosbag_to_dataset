from os import listdir
from os.path import join
from rosbag_to_dataset.util.os_util import maybe_mkdir
import shutil
from tqdm import tqdm 


if __name__ == '__main__':
    dest_fp = '/project/learningphysics/parv_dataset/affix_sys_id/eval_tartan_bags'
    maybe_mkdir(dest_fp,force = False)
    source_fp = '/project/learningphysics/torch_dataset/20210905/atv_20210905_eval'
    # source_folders = [x[:-4] for x in listdir(source_fp)]
    source_bags = [x[:-3] for x in listdir(source_fp)]

    bag_fps = [
            '/project/learningphysics/dataset/20210826_bags',
            '/project/learningphysics/dataset/20210828',
            '/project/learningphysics/dataset/20210902',
            '/project/learningphysics/dataset/20210903'
            ]

    all_bags = {}
    for bag_fp in bag_fps:
        cur_traj = [(x[:-4],join(bag_fp,x)) for x in listdir(bag_fp)]
        cur_traj_dict = dict(cur_traj)
        all_bags.update(cur_traj_dict)
    
    final_dict = {}
    for x in tqdm(source_bags):
        final_dict[x] = all_bags[x]
        shutil.copy(all_bags[x],dest_fp)
    
    # print(final_dict)
    # import pdb;pdb.set_trace()