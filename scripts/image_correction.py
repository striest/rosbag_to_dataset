# this image correction is used for correcting the exposure for the warthog data
# Note that the original image_left and image_right folder will be renamed !!

import numpy as np
import cv2
import argparse
from os import listdir, system
from os.path import isdir, isfile, join, split
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir
import time

# python3 image_correction.py --root /project/learningphysics/arl_20220922_traj --bag_list trajlist_arl.txt --save_to /project/learningphysics/arl_20220922_preview
# python scripts/image_correction.py --root /cairo/arl_bag_files/sara_traj --bag_list scripts/trajlist_local.txt --save_to /cairo/arl_bag_files/arl_20220922_preview

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

    # if vis:
    #     cv2.imshow('awb', cv2.hconcat([img_small, img_mean, img_percentile]))
    #     cv2.waitKey(0)
    #     cv2.imwrite('/home/wenshan/tmp/arl_data/image_correction/'+str(np.random.randint(100))+'.png', cv2.hconcat([img_small, img_mean, img_percentile]))
    return res

# the following function try to use more gentle gamma correction
def init_gamma_curves():
    gamma_curves = []
    gammas = np.arange(0.1,1.01,0.005)
    for gamma in gammas:
        gamma_curve = np.empty((256), np.uint8)
        for i in range(256):
            gamma_curve[i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
        gamma_curves.append(gamma_curve)
    return np.array(gamma_curves), gammas

def get_hist_curve(img):
    img_small = cv2.resize(img, (0,0), fx=0.2, fy=0.2) # resize the image to make it faster
    hist, _ = np.histogram(img_small[:], range=(0,256), bins=256)
    hist_acc = np.cumsum(hist)
    hist_acc = hist_acc / hist_acc[-1]
    lookUpTable = np.clip(hist_acc * 255.0, 0, 255).astype(np.uint8)

    return lookUpTable

def find_gamma_curve(img, gamma_curves, gammas, soft_factor = 1.0):
    img_curve = get_hist_curve(img)
    # find closest match
    diff = gamma_curves.astype(np.float32) - img_curve.reshape(1,256).astype(np.float32)
    diff = np.abs(diff).mean(axis=1)
    min_ind = np.argmin(diff)

    # soft the gamma curve further
    gamma = 1-(1-gammas[min_ind]) * soft_factor
    soft_ind = np.argmin(np.abs(gammas - gamma))
    # print("closese gamma {}, soft gamma {}".format(gammas[min_ind], gammas[soft_ind]))

    return gamma_curves[soft_ind]

def gentleGammaCorrection(img_original, gamma_curves, gammas, img_additional=None, soft_factor = 1.0):
    gamma_curve = find_gamma_curve(img_original, gamma_curves, gammas, soft_factor)
    res = cv2.LUT(img_original, gamma_curve)
    if img_additional is not None: # used in the stereo case
        res2 = cv2.LUT(img_additional, gamma_curve)
        res = (res, res2)
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

    gamma_curves, gammas = init_gamma_curves()
    for imgfile in imglist:
        leftimg = cv2.imread(join(rootfolder, 'left', imgfile))
        rightimg = cv2.imread(join(rootfolder, 'right', imgfile))

        # (left_correction, right_correction) = gammaCorrection(leftimg, rightimg, roi=None, vis=False)
        # (left_correction, right_correction) = autowhitebalance(left_correction, right_correction)
        sstime = time.time()
        (left_correction, right_correction) = gentleGammaCorrection(leftimg, gamma_curves, gammas, rightimg, soft_factor=0.7)
        print(time.time()-sstime)
        # import ipdb;ipdb.set_trace()
        # cv2.imshow("img", left_correction)
        # cv2.waitKey(0)
        vis = cv2.vconcat([cv2.hconcat([leftimg, left_correction]), cv2.hconcat([rightimg, right_correction])])
        cv2.imwrite(join(rootfolder, imgfile), vis)
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

        # mv the raw folder
        leftfolder = trajdir + '/image_left'
        rightfolder = trajdir + '/image_right' 
        # if isdir(leftfolder + '_raw') or isdir(rightfolder + '_raw'):
        #     print("Raw folder already exists.. please check the folders")
        #     continue
        # system('mv ' + leftfolder + ' ' + leftfolder + '_raw')
        # system('mv ' + rightfolder + ' ' + rightfolder + '_raw')
        # leftfolder = trajdir + '/image_left_raw'
        # rightfolder = trajdir + '/image_right_raw' 

        leftfolder_correction = trajdir + '/image_left_correction_awb_gen' #'/image_left'#
        rightfolder_correction = trajdir + '/image_right_correction_awb_gen' #'/image_right'# 
        leftfolder_correction0 = trajdir + '/image_left_correction_gen' #'/image_left'#
        rightfolder_correction0 = trajdir + '/image_right_correction_gen' #'/image_right'# 
        maybe_mkdir(leftfolder_correction)
        maybe_mkdir(rightfolder_correction)
        maybe_mkdir(leftfolder_correction0)
        maybe_mkdir(rightfolder_correction0)

        # # for debugging
        # maybe_rmdir(trajdir + '/image_left_correction_awb')
        # maybe_rmdir(trajdir + '/image_right_correction_awb')
        # maybe_rmdir(trajdir + '/image_left_correction')
        # maybe_rmdir(trajdir + '/image_right_correction')

        outvidfile = join(args.save_to, outfolder + '_color_correction_gen.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 10.0, (1080, 270))

        leftimglist = listdir(leftfolder)
        leftimglist = [leftfolder + '/' + img for img in leftimglist if img.endswith('.png')]
        leftimglist.sort()

        rightimglist = listdir(rightfolder)
        rightimglist = [rightfolder + '/' + img for img in rightimglist if img.endswith('.png')]
        rightimglist.sort()
            
        gamma_curves, gammas = init_gamma_curves()

        datanum = len(leftimglist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            leftimg = cv2.imread(leftimglist[k])
            rightimg = cv2.imread(rightimglist[k])

            # (left_correction, right_correction) = gammaCorrection(leftimg, rightimg, roi=None, vis=False)
            # (left_correction_awb, right_correction_awb) = autowhitebalance(left_correction, right_correction)
            # left_correction = gammaCorrection(leftimg)
            # right_correction = gammaCorrection(rightimg)
            left_correction = gentleGammaCorrection(leftimg, gamma_curves, gammas, soft_factor=0.7)
            right_correction = gentleGammaCorrection(rightimg, gamma_curves, gammas, soft_factor=0.7)
            left_correction_awb = autowhitebalance(left_correction)
            right_correction_awb = autowhitebalance(right_correction)
            # import ipdb;ipdb.set_trace()
            cv2.imwrite(join(leftfolder_correction, split(leftimglist[k])[-1]), left_correction_awb)
            cv2.imwrite(join(rightfolder_correction, split(rightimglist[k])[-1]), right_correction_awb)

            cv2.imwrite(join(leftfolder_correction0, split(leftimglist[k])[-1]), left_correction)
            cv2.imwrite(join(rightfolder_correction0, split(rightimglist[k])[-1]), right_correction)

            vis = cv2.resize(cv2.hconcat([leftimg, left_correction, left_correction_awb]), (0,0), fx=0.25, fy=0.25)
            fout.write(vis)
            if k % 100 == 0:
                print("  process {}...".format(k))
        # fout.release()    

if __name__ == '__main__':
    main()
    # test_speed()