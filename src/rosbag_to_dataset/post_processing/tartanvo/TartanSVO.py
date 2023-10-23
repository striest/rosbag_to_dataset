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

import torch
import numpy as np
import time

np.set_printoptions(precision=4, suppress=True, threshold=10000)

from .StereoVONet import StereoVONet

# debug
from .utils import visflow, disp2vis, tensor2img
import cv2

class TartanSVO(object):
    def __init__(self, model_name, network, blxfx):
        # import ipdb;ipdb.set_trace()
        self.vonet = StereoVONet(network=network, intrinsic=True, 
                            flowNormFactor=1.0, stereoNormFactor=0.02, poseDepthNormFactor=0.25, 
                            down_scale=True, config=1, 
                            fixflow=True, fixstereo=True, autoDistTarget=False,
                            blxfx=blxfx)  # consider the scale width 844

        # load the whole model
        if model_name.endswith('.pkl'):
            self.load_model(self.vonet, model_name)

        self.vonet.cuda()

        self.test_count = 0
        self.pose_std = np.array([ 0.13,  0.13,  0.13,  0.013 ,  0.013,  0.013], dtype=np.float32) # the output scale factor
        self.flow_norm = 20 # scale factor for flow
        self.stereo_norm = 50 # scale factor for stereo
        self.network = network

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            print("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]
                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

        if ( 0 == len(preTrainDictTemp) ):
            raise Exception("Could not load model from %s." % (modelname), "load_model")

        model_dict.update(preTrainDictTemp)
        model.load_state_dict(model_dict)
        print('Model loaded...')
        return model

    def test_batch(self, sample, vis=False):
        self.test_count += 1
        if self.network == 0 or self.network == 1: # PWC-Net for flow
            img0_flow   = sample['img0'].cuda() 
            img1_flow   = sample['img0n'].cuda()
        else:
            img0_flow   = sample['img0_norm'].cuda() 
            img1_flow   = sample['img0n_norm'].cuda()

        intrinsic = sample['intrinsic'].cuda()
        # if sample.has_key('scale_w'):
        #     scale_w = sample['scale_w'].cuda()
        #     scale_w = scale_w.view(scale_w.shape + (1,1))
        # else:
        #     scale_w = 1.0

        img0_stereo   = sample['img0_norm'].cuda()
        img1_stereo   = sample['img1_norm'].cuda()


        # blxfx = sample['blxfx'].view((sample['blxfx'].shape[0], 1, 1, 1)).cuda()

        self.vonet.eval()
        with torch.no_grad():
            # starttime = time.time()
            # flow_output, stereo_output, pose_output = self.vonet(img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsic, 
            #                                                      scale_w=1, blxfx = blxfx)
            # flow_output, stereo_output, pose_output = self.vonet(inputTensor)
            inputTensor = torch.cat((img0_flow, img1_flow, img0_stereo, img1_stereo, intrinsic), dim=1)
            flow_output, stereo_output, pose_output = self.vonet(inputTensor)
            # inferencetime = time.time()-starttime
        # import ipdb;ipdb.set_trace()
        posenp = pose_output.data.cpu().numpy()
        posenp = posenp * self.pose_std
        flownp = flow_output.cpu().numpy()
        flownp = flownp[0].transpose(1,2,0)
        flownp = flownp * self.flow_norm
        stereonp = stereo_output.cpu().numpy()
        stereonp = stereonp * self.stereo_norm
        stereonp = stereonp[0,0]

        # debug
        if vis:
            flowvis = visflow(flownp)
            stereovis = disp2vis(stereonp, 3)
            disp1 = np.concatenate((flowvis, stereovis), axis=0) # 224 x 160
            disp1 = cv2.resize(disp1, (640, 896))
            # leftvis = (img0_flow.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            # rightvis = (img1_flow.cpu().numpy()[0].transpose(1,2,0)*255).astype(np.uint8)
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            leftvis = tensor2img(img0_stereo[0].cpu(), mean, std)
            rightvis = tensor2img(img1_stereo[0].cpu(), mean, std)
            disp2 = np.concatenate((leftvis, rightvis), axis=0)
            disp = cv2.resize(np.concatenate((disp2, disp1), axis=1), (0,0), fx=0.5, fy=0.5)
            cv2.imshow('img', disp)
            cv2.waitKey(1)
        # import ipdb;ipdb.set_trace()

        # print("{} Pose inference using {}s: \n{}".format(self.test_count, inferencetime, posenp))
        return posenp #, flownp

