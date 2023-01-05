from os import listdir
from os.path import join
def diff(list1, list2):
    return list(set(list1).symmetric_difference(set(list2)))
if __name__ == '__main__':
    source_fp = '/project/learningphysics/parv_dataset/affix_sys_id/train_tartan_bags'
    dest_fp = '/project/learningphysics/torch_dataset/20210905/atv_20210905_train'
    source_folders = [x[:-4] for x in listdir(source_fp)]
    dest_folders = [x[:-3] for x in listdir(dest_fp)]
    import pdb;pdb.set_trace()
    print(diff(source_folders,dest_folders))
