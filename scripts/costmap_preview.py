import numpy as np
import cv2
import argparse
from os import listdir
from os.path import isdir, isfile, join
from scipy.spatial.transform import Rotation
from rosbag_to_dataset.util.os_util import maybe_mkdir

# The visualization view looks like this: 
# - vel_body
# - slope angle
# RGB      | 
# ----------
# Costmaps | 

# python scripts/data_preview.py --root /project/learningphysics/tartandrive_trajs --bag_list scripts/trajlist.txt --save_to /project/learningphysics/tartandrive_preview
# python scripts/data_preview.py --root test_output --bag_list scripts/trajlist_local.txt --save_to test_output
# python3 scripts/costmap_preview.py --root /ocean/projects/cis220039p/shared/tartandrive/2023_traj/v1 --bag_list scripts/trajlist_2023_v1.txt --save_to /ocean/projects/cis220039p/shared/tartandrive/2023_costmap_preview

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, required=True, default="", help='Path to the trajectories data')
    parser.add_argument('--bag_list', type=str, required=True, default="", help='List of bagfiles and output folders')
    parser.add_argument('--save_to', type=str, required=True, default="", help='Path to save the preview video')
    parser.add_argument('--cost_folder', type=str, default="cost", help='Cost folder name')

    args = parser.parse_args()

    costmapsize = (600, 480)
    fpvsize = (600, 320)
    dt = 0.1

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

        fpvfolder = trajdir + '/image_left_color' if isdir(trajdir + '/image_left_color') else (trajdir + '/image_left')
        costmapfolder1 = trajdir + '/costmap'
        costmapfolder2 = trajdir + '/costmap_v3' 
        superodomfolder = trajdir + '/super_odom'
        gpsodomfolder = trajdir + '/gps_odom'
        costfolder1 = trajdir + '/traversability_breakdown'
        costfolder2 = trajdir + '/traversability_v3_breakdown' 

        if not isdir(fpvfolder) or \
             not isdir(trajdir + '/cmd') or \
             not isdir(superodomfolder) or \
             not isdir(gpsodomfolder) or \
             not isdir(costmapfolder1) or \
             not isdir(costmapfolder2):
             print('!!! Missing data folders')

        outvidfile = join(args.save_to, outfolder + '.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        fout=cv2.VideoWriter(outvidfile, fourcc, 1.0/dt, (600, 800))

        imglist = listdir(fpvfolder)
        imglist = [fpvfolder + '/' + img for img in imglist if img.endswith('.png')]
        imglist.sort()

        costmap1list = listdir(costmapfolder1)
        costmap1list = [costmapfolder1 + '/' + img for img in costmap1list if img.endswith('.png')]
        costmap1list.sort()

        costmap2list = listdir(costmapfolder2)
        costmap2list = [costmapfolder2 + '/' + img for img in costmap2list if img.endswith('.png')]
        costmap2list.sort()

        costvellist = listdir(costmapfolder2)
        costvellist = [costmapfolder2 + '/' + velfile for velfile in costvellist if velfile.endswith('vel.txt')]
        costvellist.sort()

        cmds = np.load(trajdir + '/cmd/twist.npy')
        odoms = np.load(gpsodomfolder + '/odometry.npy')
        superodoms = np.load(superodomfolder + '/odometry.npy')
        costfile1 = join(costfolder1, 'float.npy')
        costfile2 = join(costfolder2, 'float.npy')
        if isfile(costfile1):
            costs1 = np.load(costfile1)
        else:
            costs1 = None
        if isfile(costfile2):
            costs2 = np.load(costfile2)
        else:
            costs2 = None
            
        datanum = len(imglist)
        for k in range(datanum):
            # import ipdb;ipdb.set_trace()
            disp1 = cv2.imread(imglist[k])
            disp1 = cv2.resize(disp1, fpvsize)

            disp2 = cv2.imread(costmap1list[k])
            disp2 = cv2.rotate(disp2[:,:240,:], cv2.ROTATE_90_COUNTERCLOCKWISE)
            disp3 = cv2.imread(costmap2list[k])
            disp3 = cv2.rotate(disp3[:,:240,:], cv2.ROTATE_90_COUNTERCLOCKWISE)

            cost1 = costs1[k * (len(costs1)//datanum)] if costs1 is not None else [-1]
            cost2 = costs2[k * (len(costs2)//datanum)] if costs2 is not None else [-1]
            text1 = 'Cost 1: ' + ''.join([" {:.2f} ".format(cost1[k]) for k in range(len(cost1))])
            text2 = 'Cost 1: ' + ''.join([" {:.2f} ".format(cost2[k]) for k in range(len(cost2))])
            costvel = np.loadtxt(costvellist[k])
            cost_text = 'Cost-Vel: ' + ''.join([" {:.2f} ".format(costvelk) for costvelk in costvel[::5]])

            disp2[10:55, 0:370, :] = disp2[10:55, 0:370, :]/5 * 2 + np.array([70, 40, 10],dtype=np.uint8)/5 * 3
            cv2.putText(disp2,text1, (15,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)
            cv2.putText(disp2,cost_text, (15,45),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)

            disp3[10:35, 0:370, :] = disp3[10:35, 0:370, :]/5 * 2 + np.array([70, 40, 10],dtype=np.uint8)/5 * 3
            cv2.putText(disp3,text2, (15,25),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,210,245),thickness=1)


            disp2 = cv2.resize(cv2.vconcat((disp2, disp3)), costmapsize)

            disp = cv2.vconcat((disp1, disp2))

            cmd = cmds[k * (len(cmds)//datanum)] 
            odom = odoms[k * (len(odoms)//datanum)] 
            superodom = superodoms[k * (len(superodoms)//datanum)]

            velx = superodom[7]
            yaw = superodom[-1] * 180 / 3.14

            orientation = odom[3:7]
            _, slope, _ = Rotation.from_quat(orientation).as_euler("ZXY", degrees=True)

            text1 = "{} Throttle: {:.2f},         Steering: {:.2f}".format(str(k).zfill(4), cmd[0], cmd[1])
            text2 = "      Velx:    {:.2f} m/s,    Yawrate: {:.2f} deg/s".format(velx, yaw)
            text3 = "      Slope angle: {:.2f} deg ".format(slope)

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