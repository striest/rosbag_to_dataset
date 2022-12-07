# this image correction is used for correcting the exposure for the warthog data
import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isdir, isfile, join, split
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir

# python scripts/data_preview.py --root /project/learningphysics/tartandrive_trajs --bag_list scripts/trajlist.txt --save_to /project/learningphysics/tartandrive_preview
# python scripts/data_preview.py --root test_output --bag_list scripts/trajlist_local.txt --save_to test_output

def gammaCorrection(img_original, img_additional=None, roi=None, vis=False):
    if roi is None:
        roi = [0,img_original.shape[0],0,img_original.shape[1]]
    img = img_original[roi[0]:roi[1], roi[2]:roi[3],:]

    hist, _ = np.histogram(img[:], range=(0,256), bins=256)
    hist_acc = np.cumsum(hist)
    hist_acc = hist_acc / hist_acc[-1]
    lookUpTable = np.clip(hist_acc * 255.0, 0, 255).astype(np.uint8)

    res = cv2.LUT(img_original, lookUpTable)
    if img_additional is not None: # used in the stereo case
        res2 = cv2.LUT(img_additional, lookUpTable)
        res = (res, res2)
        
    if vis:
        # plt.plot(hist_acc)
        # plt.show()
        img_original_disp = cv2.rectangle(img_original, (roi[2], roi[0]), (roi[3], roi[1]), (255,0,0), thickness=2)
        img_gamma_corrected = cv2.hconcat([img_original_disp, res])
        # img_gamma_corrected = cv2.resize(img_gamma_corrected, (0,0), fx=0.5, fy=0.5)
        cv2.imshow("Gamma correction", img_gamma_corrected)
        cv2.waitKey(0)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the trajectories data')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='List of bagfiles and output folders')
    parser.add_argument('--save_to', type=str, required=True, default="", help='Path to save the preview video')

    args = parser.parse_args()

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

        # find unique values of the folder
        outfolderlist = [bb[1] for bb in bagfilelist]
        outfolders = set(outfolderlist)
        print('Find {} trajectories'.format(len(outfolders)))
    else:
        print('Find no trajectory in the bag list file {}'.format(args.bag_list))
        exit()

    maybe_mkdir(args.save_to)
    for outfolder in outfolders:
        print('--- {} ---'.format(outfolder))
        trajdir = args.root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue
        leftfolder = trajdir + '/image_left'
        rightfolder = trajdir + '/image_right' 

        leftfolder_correction = trajdir + '/image_left_correction'
        rightfolder_correction = trajdir + '/image_right_correction' 
        maybe_mkdir(leftfolder_correction)
        maybe_mkdir(rightfolder_correction)

        outvidfile = join(args.save_to, outfolder + '_color_correction.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 10.0, (1440, 540))

        leftimglist = listdir(leftfolder)
        leftimglist = [leftfolder + '/' + img for img in leftimglist if img.endswith('.png')]
        leftimglist.sort()

        rightimglist = listdir(rightfolder)
        rightimglist = [leftfolder + '/' + img for img in rightimglist if img.endswith('.png')]
        rightimglist.sort()
            
        datanum = len(leftimglist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            leftimg = cv2.imread(leftimglist[k])
            rightimg = cv2.imread(rightimglist[k])

            (left_correction, right_correction) = gammaCorrection(leftimg, rightimg, roi=None, vis=False)
            # import ipdb;ipdb.set_trace()
            cv2.imwrite(join(leftfolder_correction, split(leftimglist[k])[-1]), left_correction)
            cv2.imwrite(join(rightfolder_correction, split(rightimglist[k])[-1]), right_correction)

            vis = cv2.resize(cv2.hconcat([leftimg, left_correction]), (0,0), fx=0.5, fy=0.5)
            fout.write(vis)

        fout.release()