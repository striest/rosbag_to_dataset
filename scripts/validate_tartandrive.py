import argparse
from os.path import isfile, isdir
from os import listdir
import numpy as np

# python scripts/tartandrive_dataset.py --bag_fp /cairo/arl_bag_files/tartandrive/20210903_298.bag --config_spec specs/sample_tartandrive.yaml --save_to test_output/20210903_298
# python scripts/tartandrive_dataset.py --config_spec specs/sample_tartandrive.yaml --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/tartandrive_extract

if __name__ == '__main__':
    '''
    bag_list is a text file with the following content: 
      <Full path of the bagfile0> <Output folder name0>
      <Full path of the bagfile1> <Output folder name1>
      <Full path of the bagfile2> <Output folder name2>
      ...
    The extracted data will be stored in <save_to>/<Output folder name>

    if no bag_list is specified, the code will look at bag_fp and process the single bagfile. The output folder is specified by the save_to param
    '''
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the bag file to get data from')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='Path to the bag file to get data from')
    parser.add_argument('--outfile', type=str, required=True, help='Name of the dir to save the result to')

    args = parser.parse_args()

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

    outfile = open(args.outfile, 'w')
    for bagfile, outfolder in bagfilelist:
        print('---',outfolder,'---')
        trajdir = args.root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue

        data_valid = True
        cmd = np.load(trajdir + '/cmd/twist.npy')
        cmdtime = np.loadtxt(trajdir + '/cmd/timestamps.txt')
        if(len(cmd)!=len(cmdtime)):
            data_valid = False
            print('cmd missing frames {}, steps {}'.format(len(cmd), len(cmdtime)))

        imu = np.load(trajdir + '/imu/imu.npy')
        imutime = np.loadtxt(trajdir + '/imu/timestamps.txt')
        if(len(imu)!=len(imutime)):
            data_valid = False
            print('imu missing frames {}, steps {}'.format(len(imu), len(imutime)))

        odom = np.load(trajdir + '/odom/odometry.npy')
        odomtime = np.loadtxt(trajdir + '/odom/timestamps.txt')
        if(len(odom)!=len(odomtime)):
            data_valid = False
            print('odom missing frames {}, steps {}'.format(len(odom), len(odomtime)))

        if(len(odom)!=len(cmd)):
            data_valid = False
            print('odom {}, cmd {} missing frames'.format(len(odom), len(cmd)))

        if(len(imu)!=len(cmd)*10):
            data_valid = False
            print('imu {}, cmd {} missing frames'.format(len(imu), len(cmd)))

        imglist1 = listdir(trajdir + '/image_left')
        imglist1 = [img for img in imglist1 if img.endswith('.png')]
        imglist1.sort()

        imglist2 = listdir(trajdir + '/image_left_color')
        imglist2 = [img for img in imglist2 if img.endswith('.png')]
        imglist2.sort()

        imglist3 = listdir(trajdir + '/image_right')
        imglist3 = [img for img in imglist3 if img.endswith('.png')]
        imglist3.sort()

        if(len(imglist1)!=len(cmd)):
            data_valid = False
            print('image {}, cmd {} missing frames'.format(len(imglist1), len(cmd)))

        if(len(imglist2)!=len(cmd)):
            data_valid = False
            print('image {}, cmd {} missing frames'.format(len(imglist2), len(cmd)))

        if(len(imglist3)!=len(cmd)):
            data_valid = False
            print('image {}, cmd {} missing frames'.format(len(imglist3), len(cmd)))

        if not data_valid:
            outfile.write(bagfile)
            outfile.write(' ')            
            outfile.write(outfolder)            
            outfile.write('\n')      

    outfile.close()     
