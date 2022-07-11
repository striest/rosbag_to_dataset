import os
from rosbag_to_dataset.util.os_util import maybe_mkdir
import random
import shutil


def Diff(li1, li2):
    return list(set(li1) - set(li2)) + list(set(li2) - set(li1))

if __name__ == '__main__':

    random.seed(1)
    data_dir = "/data/datasets/parvm/tartandrive_trajs"
    save_data_dir = "/data/datasets/parvm/tartandrive_trajs_parv_final"

    train_data_dir = os.path.join(save_data_dir,"train")
    eval_data_dir = os.path.join(save_data_dir,"eval")
    train_ratio = 0.8


    maybe_mkdir(save_data_dir)
    maybe_mkdir(train_data_dir)
    maybe_mkdir(eval_data_dir)
    

    final_list =[]

    for fp in os.listdir(data_dir):
        if "intervention" not in fp and "test" not in fp and "trial" not in fp:
            final_list.append(fp)

    train_list_len = int(len(final_list)*train_ratio)
    train_list = random.sample(final_list,train_list_len)
    print(train_list_len)
    print(len(final_list))
    train_list.sort()
    eval_list = Diff(final_list, train_list)
    eval_list.sort()


    print("train")
    print(train_list)
    print("eval")
    print(eval_list)

    for fp in train_list:
        print(fp)
        traj_dir = os.path.join(data_dir,fp)
        if not (os.path.exists(os.path.join(train_data_dir,fp))):
            shutil.copytree(traj_dir,os.path.join(train_data_dir,fp))

    for fp in eval_list:
        print(fp)
        traj_dir = os.path.join(data_dir,fp)
        if not (os.path.exists(os.path.join(eval_data_dir,fp))):
            shutil.copytree(traj_dir,os.path.join(eval_data_dir,fp))
    



