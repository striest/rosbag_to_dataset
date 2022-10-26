import os
import argparse
import shutil

def create_trajectory_list(dataset_dir, output_file, manual_subfolders=False):
    """
    dataset_dir can contain individual files and directories, but will convert all individual files into directories that contain a single file, so that the dataset_dir only contains subfolders, which in turn contain only rosbags.
    """
    root_folder = dataset_dir
    subfolders = []
    
    if manual_subfolders:
        # Define subfolders manually here. For example:
        # subfolders = [
        #     'exp1_warehouse_baseline_1',
        #     'exp1_warehouse_baseline_2',
        #     'exp1_warehouse_baseline_3',
        #     'exp1_warehouse_both_warehouse_1',
        #     'exp1_warehouse_both_warehouse_2',
        #     'exp2_figure8_both',
        #     'exp2_figure8_both_part2',
        # ] 
        pass
    else:
        dir_items = os.listdir(root_folder)
        for item in dir_items:
            if item.endswith(".bag"):
                new_dir = os.path.join(root_folder, item[:-4])
                os.makedirs(new_dir)
                shutil.move(os.path.join(root_folder, item), new_dir)
                subfolders.append(new_dir)
            else:
                subfolders.append(item)

    if not output_file.endswith(".txt"):
        output_file += ".txt"
    outtxt = open(output_file,'w')
    for k, subfolder in enumerate(subfolders):
        subdir = os.path.join(root_folder, subfolder)
        bagfiles = os.listdir(subdir)
        bagfiles = [bb for bb in bagfiles if bb.endswith('.bag')]
        bagfiles.sort()
        print(f'Found {len(bagfiles)} bagfiles in {subdir}')

        for k,bag in enumerate(bagfiles):
            outtxt.write(os.path.join(subdir, bag))
            outtxt.write(' ')
            outtxt.write(subfolder+'_'+ str(k))
            outtxt.write('\n')
    outtxt.close()

    print(f"The output text file is stored here: {output_file}")

    return output_file  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_dir', type=str, required=True, help="Directory where the rosbags are saved. This directory can contain subdirectories containing rosbags.")
    parser.add_argument('--output_file', type=str, required=True, help="String of text file where the bags to be processed will be stored")
    parser.add_argument("--manual_subfolders", default=False, action='store_true', help="Include this argument if you want to manually specify the subdirectories to be processed. See preprocess_dataset.py for an example.")

    args = parser.parse_args()

    create_trajectory_list(args.dataset_dir, args.output_file, args.manual_subfolders)