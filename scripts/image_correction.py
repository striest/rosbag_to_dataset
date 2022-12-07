# this image correction is used for correcting the exposure for the warthog data
import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isdir, isfile, join, split
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir
import time

# python scripts/data_preview.py --root /project/learningphysics/tartandrive_trajs --bag_list scripts/trajlist.txt --save_to /project/learningphysics/tartandrive_preview
# python scripts/data_preview.py --root test_output --bag_list scripts/trajlist_local.txt --save_to test_output

def gammaCorrection(img_original, img_additional=None, roi=None, vis=False):
    if roi is None:
        roi = [0,img_original.shape[0],0,img_original.shape[1]]
    img = img_original[roi[0]:roi[1], roi[2]:roi[3],:]

    img_small = cv2.resize(img, (0,0), fx=0.2, fy=0.2) # resize the image to make it faster
    hist, _ = np.histogram(img_small[:], range=(0,256), bins=256)
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

def autowhitebalance(img_original, img_additional=None, vis=False):
    img_small = cv2.resize(img_original, (0,0), fx=0.2, fy=0.2) # resize the image to make it faster
    # whitepatch = np.percentile(img_small, 95, axis=(0,1))
    meanvalue = np.mean(img_small, axis=(0,1))
    # print(whitepatch, meanvalue)

    res = np.clip(img_original*128.0/meanvalue.reshape(1,1,3), 0, 255).astype(np.uint8)
    # img_percentile = np.clip(img_small*255.0/whitepatch.reshape(1,1,3), 0, 255).astype(np.uint8)
    if img_additional is not None: # used in the stereo case
        res2 = np.clip(img_additional*128.0/meanvalue.reshape(1,1,3), 0, 255).astype(np.uint8)
        res = (res, res2)

    if vis:
        cv2.imshow('awb', cv2.hconcat([img_small, img_mean, img_percentile]))
        cv2.waitKey(0)
        cv2.imwrite('/home/wenshan/tmp/arl_data/image_correction/'+str(np.random.randint(100))+'.png', cv2.hconcat([img_small, img_mean, img_percentile]))
    return res

def test_speed():
    rootfolder = '/home/wenshan/tmp/arl_data/image_correction'
    imglist = ['001032.png',
                '001358.png',
                '001620.png',
                '001757.png',
                '001932.png',
                '002368.png',
                '002835.png']
    starttime = time.time()

    for imgfile in imglist:
        leftimg = cv2.imread(join(rootfolder, 'left', imgfile))
        rightimg = cv2.imread(join(rootfolder, 'right', imgfile))

        (left_correction, right_correction) = gammaCorrection(leftimg, rightimg, roi=None, vis=False)
        (left_correction, right_correction) = autowhitebalance(left_correction, right_correction)
        # import ipdb;ipdb.set_trace()
        # cv2.imshow("img", left_correction)
        # cv2.waitKey(0)
        # cv2.imwrite(join(leftfolder_correction, split(leftimglist[k])[-1]), left_correction)
        # cv2.imwrite(join(rightfolder_correction, split(rightimglist[k])[-1]), right_correction)
    print(time.time()-starttime)

def main():
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

        leftfolder_correction = trajdir + '/image_left_correction_awb'
        rightfolder_correction = trajdir + '/image_right_correction_awb' 
        maybe_mkdir(leftfolder_correction)
        maybe_mkdir(rightfolder_correction)

        outvidfile = join(args.save_to, outfolder + '_color_correction_awb.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 10.0, (1080, 270))

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
            (left_correction_awb, right_correction_awb) = autowhitebalance(left_correction, right_correction)
            # import ipdb;ipdb.set_trace()
            cv2.imwrite(join(leftfolder_correction, split(leftimglist[k])[-1]), left_correction_awb)
            cv2.imwrite(join(rightfolder_correction, split(rightimglist[k])[-1]), right_correction_awb)

            vis = cv2.resize(cv2.hconcat([leftimg, left_correction, left_correction_awb]), (0,0), fx=0.25, fy=0.25)
            fout.write(vis)

        fout.release()    

if __name__ == '__main__':
    main()