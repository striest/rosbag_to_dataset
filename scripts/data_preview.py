import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isdir, isfile
from rosbag_to_dataset.post_processing.tartanvo.utils import disp2vis
from scipy.spatial.transform import Rotation
from rosbag_to_dataset.util.os_util import maybe_mkdir, maybe_rmdir

# The visualization view looks like this: 
# - vel_body
# - slope angle
# RGB   | Heightmap
# -----------------
# Depth | RGBmap

# python scripts/data_preview.py --root /project/learningphysics/tartandrive_trajs --bag_list scripts/trajlist.txt --save_to /project/learningphysics/tartandrive_preview
# python scripts/data_preview.py --root test_output --bag_list scripts/trajlist_local.txt --save_to test_output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the trajectories data')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='List of bagfiles and output folders')
    parser.add_argument('--save_to', type=str, required=True, default="", help='Path to save the preview video')

    args = parser.parse_args()

    imgvissize = (512, 256)
    mapvissize = (256, 256)
    dt = 0.1

    if args.bag_list != "" and isfile(args.bag_list): # process multiple files specified in the text file
        with open(args.bag_list, 'r') as f:
            lines = f.readlines()
        bagfilelist = [line.strip().split(' ') for line in lines]

    maybe_mkdir(args.save_to)
    for bagfile, outfolder in bagfilelist:
        print('--- {} ---'.format(outfolder))
        trajdir = args.root + '/' + outfolder

        if not isdir(trajdir):
            print('!!! Trajectory Not Found {}'.format(trajdir))
            continue

        if not isdir(trajdir + '/image_left_color') or \
             not isdir(trajdir + '/cmd') or \
             not isdir(trajdir + '/tartanvo_odom') or \
             not isdir(trajdir + '/odom') or \
             not isdir(trajdir + '/depth_left') or \
             not isdir(trajdir + '/rgb_map') or \
             not isdir(trajdir + '/height_map'):
             print('!!! Missing data folders')

        outvidfile = args.save_to + '/' + outfolder + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 1.0/dt, (768, 512))

        imglist = listdir(trajdir + '/image_left_color')
        imglist = [trajdir + '/image_left_color/' + img for img in imglist if img.endswith('.png')]
        imglist.sort()

        depthlist = listdir(trajdir + '/depth_left')
        depthlist = [trajdir + '/depth_left/' + img for img in depthlist if img.endswith('.npy')]
        depthlist.sort()

        rgbmaplist = listdir(trajdir + '/rgb_map')
        rgbmaplist = [trajdir + '/rgb_map/' + img for img in rgbmaplist if img.endswith('.npy')]
        rgbmaplist.sort()

        heightmaplist = listdir(trajdir + '/height_map')
        heightmaplist = [trajdir + '/height_map/' + img for img in heightmaplist if img.endswith('.npy')]
        heightmaplist.sort()

        cmds = np.load(trajdir + '/cmd/twist.npy')
        motions = np.load(trajdir + '/tartanvo_odom/motions.npy')
        motions = np.concatenate((motions, motions[-1:,:])) # add one more frame
        odoms = np.load(trajdir + '/odom/odometry.npy')
        costs = np.load(trajdir + '/cost/cost.npy')

        datanum = len(imglist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            disp1 = cv2.imread(imglist[k])
            disp1 = disp1[32:-32, 64:-64, :] # crop and resize the image in the same way with the stereo matching code
            disp1 = cv2.resize(disp1, imgvissize)

            disp2 = np.load(depthlist[k])
            disp2 = disp2vis(disp2)
            disp2 = cv2.resize(disp2, imgvissize)

            disp3 = np.load(rgbmaplist[k])
            disp3 = cv2.resize(disp3, mapvissize)
            disp3 = cv2.flip(disp3, -1)

            disp4 = np.load(heightmaplist[k])
            mask = disp4[:,:,0]>10000
            disp4 = disp4[:,:,2] # mean channel
            disp4 = np.clip((disp4 - (-1.5))*73, 0, 255).astype(np.uint8) # convert height to 0-255
            disp4[mask] = 0
            disp4 = cv2.resize(disp4, mapvissize)
            disp4 = cv2.flip(disp4, -1)
            disp4 = cv2.applyColorMap(disp4, cv2.COLORMAP_JET)

            disp = np.concatenate((np.concatenate((disp1, disp2), axis=0), np.concatenate((disp3, disp4), axis=0)), axis=1)

            cmd = cmds[k] 
            motion = motions[k]
            odom = odoms[k]
            cost = costs[k]

            velx = motion[0] / dt
            _, _, yaw = Rotation.from_quat(motion[3:7]).as_euler("XYZ", degrees=True)
            yaw = yaw / dt

            orientation = odom[3:7]
            _, slope, _ = Rotation.from_quat(orientation).as_euler("ZXY", degrees=True)

            text1 = "{} Throttle: {:.2f},         Steering: {:.2f}".format(str(k).zfill(4), cmd[0], cmd[1])
            text2 = "      Velx:    {:.2f} m/s,    Yawrate: {:.2f} deg/s".format(velx, yaw)
            text3 = "      Slope angle: {:.2f} deg,  Cost: {:.2f}".format(slope, cost)

            # pts = np.array([[0,0],[320,0],[320,20],[0,20]],np.int32)
            # put a bg rect
            disp[10:75, 0:370, :] = disp[10:75, 0:370, :]/5 * 2 + np.array([70, 40, 10],dtype=np.uint8)/5 * 3
            # cv2.fillConvexPoly(disp, pts, (70,30,10))
            cv2.putText(disp,text1, (15,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
            cv2.putText(disp,text2, (15,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
            cv2.putText(disp,text3, (15,65),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)

            # print('cmd {}, velx {}, yaw {}, slope {}'.format(cmd, velx, yaw, slope))
            # cv2.imshow('img', disp)
            # cv2.waitKey(0)

            fout.write(disp)
        fout.release()