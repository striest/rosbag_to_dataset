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
import numpy as np

from TrajFolderDataset import TrajFolderDataset
from torch.utils.data import DataLoader

from utils import se2SE, SO2quat, se2quat
from TartanSVO import TartanSVO
import time
import os
from os import mkdir
from os.path import isdir, dirname, realpath
from arguments import get_args

class TartanVOInference(object):
    def __init__(self):
        '''
        Two options are: model_name = 43_6_2_vonet_30000.pkl, network = 2
                         model_name = 43_6_2_vonet_30000_wo_pwc.pkl, network = 0
        '''
        self.curdir = dirname(realpath(__file__))

        args = get_args()
        self.parse_params(args)

        # self.cam_intrinsics = [self.w, self.h, self.focalx, self.focaly, self.pu, self.pv]
        self.load_model()

        # self.imgbuf = []
        # self.scale = 1.0

    def load_dataset(self, traj_root_folder):
        testDataset = TrajFolderDataset(traj_root_folder, leftfolder='image_left', rightfolder='image_right', colorfolder=None, forvo=True, \
                                        imgw=self.w, imgh=self.h, crop_w=self.crop_w, crop_h=self.crop_h, resize_w=self.input_w, resize_h=self.input_h, \
                                        focalx=self.focalx, focaly=self.focaly, centerx=self.pu, centery=self.pv, blxfx=self.fxbl)

        testDataloader = DataLoader(testDataset, batch_size=self.batch_size, 
                                        shuffle=False, num_workers=self.worker_num)
        self.testDataiter = iter(testDataloader)

        self.pose = np.matrix(np.eye(4,4))

    def parse_params(self, args):
        self.modelname = args.model_name # 'models/43_6_2_vonet_30000.pkl'
        self.network_type = args.network_type # 0

        # camera parameters
        self.w = args.image_width # , 1024)
        self.h = args.image_height # , 544)
        self.focalx = args.focal_x # , 477.6049499511719)
        self.focaly = args.focal_y # , 477.6049499511719)
        self.pu = args.center_x # , 499.5)
        self.pv = args.center_y # , 252.0)
        self.fxbl = args.focal_x_baseline # , 100.14994812011719)

        # depth generation parameters
        self.crop_w = args.image_crop_w # , 64) # to deal with vignette effect, crop the image
        self.crop_h = args.image_crop_h # , 32) # after cropping the size is (960, 512)
        self.input_w = args.image_input_w # , 512)
        self.input_h = args.image_input_h # , 256)
        self.visualize = args.visualize_depth # , True)

        self.batch_size = args.batch_size
        self.worker_num = args.worker_num
        # # some flags to control the point cloud processing
        # self.transform_ground = args.pc_transform_ground # , True)

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
        resize_factor = self.input_w/(self.w-2*self.crop_w)
        self.tartanvo = TartanSVO(model_name, network=self.network_type, blxfx=self.fxbl*resize_factor)

    def process(self, traj_root_folder, vo_output_folder):
        # prepare output folder
        if vo_output_folder is not None:
            outdir = traj_root_folder + '/' + vo_output_folder
            if not isdir(outdir):
                mkdir(outdir)
                print('Create folder: {}'.format(outdir))

        self.load_dataset(traj_root_folder)
        motionlist = []
        poselist = [[0.,0.,0.,0.,0.,0.,1.0]]
        while True: # loop just for visualization

            starttime = time.time()
            try:
                sample = self.testDataiter.next()
            except StopIteration:
                break
            motion = self.tartanvo.test_batch(sample, vis=False)
            print(motion)
            # import ipdb;ipdb.set_trace()
            print('Inference time: {}'.format(time.time()-starttime))
            motions, poses = self.process_motion(motion)
            motionlist.extend(motions)
            poselist.extend(poses)

        np.savetxt(outdir + '/motions.txt', np.array(motionlist))
        np.savetxt(outdir + '/poses.txt', np.array(poselist))
        np.save(outdir + '/motions.npy', np.array(motionlist))
        np.save(outdir + '/poses.npy', np.array(poselist))

# python imgs2odom.py --model-name 43_6_2_vonet_30000_wo_pwc.pkl --network-type 2  --image-input-w 640 --image-input-h 448
if __name__ == '__main__':
    node = TartanVOInference()
    # node.process(traj_root_folder='/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_298', vo_output_folder = 'tartanvo_odom')
    # node.process(traj_root_folder='/home/mateo/Data/SARA/TartanDriveCost/Trajectories/000009', vo_output_folder = 'tartanvo_odom')

    # node.process(traj_root_folder='/home/mateo/rosbag_to_dataset/test_output/20210903_42', vo_output_folder = 'tartanvo_odom')


    dataset_folder = '/home/mateo/Data/SARA/TartanDriveCost/'
    trajectories_dir = os.path.join(dataset_folder, "Trajectories")
    traj_dirs = list(filter(os.path.isdir, [os.path.join(trajectories_dir,x) for x in sorted(os.listdir(trajectories_dir))]))

    for i, d in enumerate(traj_dirs):
        print(f"Processing trajectory in {d} for odom")
        node.process(traj_root_folder=d, vo_output_folder = 'tartanvo_odom')