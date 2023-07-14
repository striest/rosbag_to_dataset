#!/usr/bin/env python3

# Software License Agreement (BSD License)
#
# Copyright (c) 2020, Wenshan Wang, Yaoyu Hu,  CMU
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#  * Neither the name of CMU nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
import cv2
import numpy as np
from torch.utils.data import DataLoader
import time
from os import mkdir
from os.path import isdir, dirname, realpath
# from .arguments_wanda import *
from .arguments import *
# from .arguments_warthog5 import *

from .TrajFolderDataset import TrajFolderDataset
from .utils import se2SE, SO2quat, se2quat
from .TartanSVO import TartanSVO

class TartanVOInference(object):
    def __init__(self):
        '''
        Two options are: model_name = 43_6_2_vonet_30000.pkl, network = 1
                         model_name = 43_6_2_vonet_30000_wo_pwc.pkl, network = 2
        '''
        self.curdir = dirname(realpath(__file__))

        self.parse_params()

        # self.cam_intrinsics = [self.w, self.h, self.focalx, self.focaly, self.pu, self.pv]
        self.load_model()

        # self.imgbuf = []
        # self.scale = 1.0

    def load_dataset(self, traj_root_folder, left_input_folder, right_input_folder):
        testDataset = TrajFolderDataset(traj_root_folder, leftfolder=left_input_folder, rightfolder=right_input_folder, colorfolder="", forvo=True, \
                                        imgw=self.w, imgh=self.h, crop_w=self.crop_w, crop_h_low=self.crop_h, crop_h_high= self.crop_h, resize_w=self.resize_w, resize_h=self.resize_h, \
                                        focalx=self.focalx, focaly=self.focaly, centerx=self.pu, centery=self.pv, blxfx=self.fxbl,stereomaps=self.stereomaps)

        testDataloader = DataLoader(testDataset, batch_size=self.batch_size, 
                                        shuffle=False, num_workers=self.worker_num)
        self.testDataiter = iter(testDataloader)

        self.pose = np.matrix(np.eye(4,4))

    def parse_params(self):
        self.modelname = vo_args['model_name'] # 'models/43_6_2_vonet_30000.pkl'
        self.network_type = vo_args['network_type'] # 0

        # camera parameters
        self.w = common_args['image_width'] # , 1024)
        self.h = common_args['image_height'] # , 544)
        self.focalx = common_args['focal_x'] # , 477.6049499511719)
        self.focaly = common_args['focal_y'] # , 477.6049499511719)
        self.pu = common_args['center_x'] # , 499.5)
        self.pv = common_args['center_y'] # , 252.0)
        self.fxbl = common_args['focal_x_baseline'] # , 100.14994812011719)

        # depth generation parameters
        self.resize_w = vo_args['image_resize_w'] # after resizing the size is (896, 448)
        self.resize_h = vo_args['image_resize_h'] # 
        self.crop_w = vo_args['image_crop_w'] # 
        self.crop_h = vo_args['image_crop_h'] # after cropping the size is (640, 448)
        self.input_w = vo_args['image_input_w'] # , 640)
        self.input_h = vo_args['image_input_h'] # , 448)
        self.visualize = vo_args['visualize'] # , True)

        self.batch_size = common_args['batch_size']
        self.worker_num = common_args['worker_num']
        self.stereo_maps = common_args['stereo_maps']
        if self.stereo_maps != '':
            loadmap = np.load(self.curdir+'/'+self.stereo_maps, allow_pickle=True)
            loadmap = loadmap.item()
            # import ipdb;ipdb.set_trace()
            map1, map2 = cv2.initUndistortRectifyMap(\
                        loadmap['k1'], loadmap['d1'],\
                        loadmap['r1'], loadmap['p1'],\
                        (self.w, self.h), cv2.CV_32FC1)

            map3, map4 = cv2.initUndistortRectifyMap(\
                        loadmap['k2'], loadmap['d2'],\
                        loadmap['r2'], loadmap['p2'],\
                        (self.w, self.h), cv2.CV_32FC1)
            self.stereomaps = [map1, map2, map3, map4]
        else:
            self.stereomaps = None

    def process_motion(self, motion):
        motionlist = []
        poselist = []
        for k in range(motion.shape[0]):
            # filestr = filenames[k].split('.')[0]
            motion_quat = se2quat(motion[k]) # x, y, z, rx, ry, rz, rw
            motionlist.append(motion_quat)
            # np.savetxt(outdir + '/' + filestr +'_motion.txt', motion_quat)
            motion_mat = se2SE(motion[k])
            self.pose = self.pose * motion_mat
            quat = SO2quat(self.pose[0:3,0:3])
            pose_quat = [self.pose[0,3], self.pose[1,3], self.pose[2,3], quat[0], quat[1], quat[2], quat[3]]
            poselist.append(pose_quat)
            # np.savetxt(outdir + '/' + filestr +'_pose.txt', pose_quat)
        return motionlist, poselist

    def load_model(self,):
        # model_name = rospy.get_param('~model_name', '43_6_2_vonet_30000_wo_pwc.pkl') # 43_6_2_vonet_30000.pkl
        # network_type = rospy.get_param('~network_type', '2') # 0
        model_name = self.curdir + '/models/' + self.modelname
        # resize_factor = float(self.input_w)/(self.w-2*self.crop_w)
        resize_factor = self.resize_w/self.w  # debug vo
        self.tartanvo = TartanSVO(model_name, network=self.network_type, blxfx=self.fxbl*resize_factor)

    def process(self, traj_root_folder, vo_output_folder, left_input_folder='image_left', right_input_folder='image_right'):
        # prepare output folder
        if vo_output_folder is not None:
            outdir = traj_root_folder + '/' + vo_output_folder
            if not isdir(outdir):
                mkdir(outdir)
                print('Create folder: {}'.format(outdir))

        self.load_dataset(traj_root_folder, left_input_folder, right_input_folder)
        motionlist = []
        poselist = [[0.,0.,0.,0.,0.,0.,1.0]]
        count = 0
        while True: # loop just for visualization

            starttime = time.time()
            try:
                sample = next(self.testDataiter)
            except StopIteration:
                break
            # import ipdb;ipdb.set_trace()
            motion = self.tartanvo.test_batch(sample, vis=self.visualize)
            print(motion)
            # import ipdb;ipdb.set_trace()
            print('{}. VO inference time: {}'.format(count, time.time()-starttime))
            motions, poses = self.process_motion(motion)
            motionlist.extend(motions)
            poselist.extend(poses)
            count += len(motion)

        np.savetxt(outdir + '/motions.txt', np.array(motionlist))
        np.savetxt(outdir + '/poses.txt', np.array(poselist))
        np.save(outdir + '/motions.npy', np.array(motionlist))
        np.save(outdir + '/poses.npy', np.array(poselist))

# python imgs2odom.py --model-name 43_6_2_vonet_30000_wo_pwc.pkl --network-type 2  --image-input-w 640 --image-input-h 448
if __name__ == '__main__':
    node = TartanVOInference()
    # node.process(traj_root_folder='/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210805_slope2', vo_output_folder = 'tartanvo_odom')
    node.process(traj_root_folder='/cairo/arl_bag_files/SARA/arl0608/wanda_cmu_0', vo_output_folder = 'tartanvo_odom')
