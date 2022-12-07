import cv2
import torch
import numpy as np
import time
from os import mkdir
from os.path import isdir, dirname, realpath

# from PSM import stackhourglass as StereoNet
# from .StereoNet7 import StereoNet7 as StereoNet
from .StereoFlowNet import StereoNet
from .TrajFolderDataset import TrajFolderDataset

from torch.utils.data import DataLoader
# from .arguments_wanda import *
# from .arguments import *
from .arguments_warthog5 import *

def depth_to_point_cloud(depth, focalx, focaly, pu, pv, filtermin=-1, filtermax=-1, colorimg=None, mask=None):
    """
    Convert depth image to point cloud based on intrinsic parameters
    :param depth: depth image
    :colorimg: a colored image that aligns with the depth, if not None, will return the color
    :mask: h x w bool array, throw away the points if mask value is false
    :return: xyz point array
    """
    h, w = depth.shape
    depth64 = depth.astype(np.float64)
    wIdx = np.linspace(0, w - 1, w, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    hIdx = np.linspace(0, h - 1, h, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    u, v = np.meshgrid(wIdx, hIdx)

    if filtermax!=-1:
        maskf = depth<filtermax
        if filtermin!=-1:
            maskf = np.logical_and(maskf, depth>filtermin)
    else:
        if filtermin!=-1:
            maskf = depth>filtermin

    if mask is not None:
        if maskf is not None:
            maskf = np.logical_and(maskf, mask)
        else:
            maskf = mask

    if maskf is not None:
        depth64 = depth64[maskf]
        depth = depth[maskf]
        if colorimg is not None:
            colorimg = colorimg[maskf]
        u = u[maskf]
        v = v[maskf]
    # print('Depth mask {} -> {}'.format(h*w, mask.sum()))

    x = (u - pu) * depth64 / focalx
    y = (v - pv) * depth64 / focaly
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    points = np.stack([depth, x, y], axis=1) #rotate_points(depth, x, y, mode) # amigo: this is in NED coordinate
    return points, colorimg

def coord_transform(points):
    # the R and T come from gound calibration
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                    [ 0.,         -0.99969192, -0.02482067],
                    [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    return points_trans.transpose(1, 0)

def points_height_filter(points, maxhight, colorimg=None):
    # import ipdb;ipdb.set_trace()
    # points = points.reshape(-1, 3)
    mask = points[:,2]<maxhight
    points_filter = points[mask, :]
    if colorimg is not None:
        colorimg_filter = colorimg[mask, :]
    else:
        colorimg_filter = None
    return points_filter, colorimg_filter

def image_rectify(left_color, right_color, maps):
    limg = cv2.remap( left_color, maps[0], maps[1], cv2.INTER_LINEAR )
    rimg = cv2.remap( right_color, maps[2], maps[3], cv2.INTER_LINEAR )
    return limg, rimg

class StereoInference:
    def __init__(self):
        '''
        traj_root_folder: the root folder for the input trajectory
        depth_output_folder: if given, the data will be output to the folder
        '''
        self.curdir = dirname(realpath(__file__))
        self.parse_params()

        self.stereonet = StereoNet()
        self.load_model(self.stereonet, self.curdir+'/models/'+self.modelname)
        self.stereonet.cuda()
        self.stereonet.eval()

        # self.mean=[0.485, 0.456, 0.406]
        # self.std=[0.229, 0.224, 0.225]
        if self.mask_file != "":
            self.atvmask = np.load(self.curdir+'/' + self.mask_file)
        else:
            self.atvmask = np.zeros((self.h, self.w))

        self.crop_intrinsics()
        self.scale_intrinsics()

        self.crop_resize_atvmask()
        self.atvmask = self.atvmask < 10 # a threshold
        self.mask_boarder_points()

    def load_dataset(self, traj_root_folder):
        testDataset = TrajFolderDataset(traj_root_folder, leftfolder='image_left', rightfolder='image_right', colorfolder=self.colored_folder, 
                                        forvo=False,  crop_w=self.crop_w, crop_h_low=self.crop_h_low, crop_h_high= self.crop_h_high, resize_w=self.input_w, resize_h=self.input_h, 
                                        stereomaps=self.stereomaps)
        testDataloader = DataLoader(testDataset, batch_size=self.batch_size, 
                                        shuffle=False, num_workers=self.worker_num)
        self.testDataiter = iter(testDataloader)

    def parse_params(self):
        self.modelname = stereo_args["model_name"] #'models/5_5_4_stereo_30000.pkl'

        # camera parameters
        self.w = common_args["image_width"] # , 1024)
        self.h = common_args["image_height"] # , 544)
        self.focalx = common_args["focal_x"] # , 477.6049499511719)
        self.focaly = common_args["focal_y"] # , 477.6049499511719)
        self.pu = common_args["center_x"] # , 499.5)
        self.pv = common_args["center_y"] # , 252.0)
        self.fxbl = common_args["focal_x_baseline"] # , 100.14994812011719)

        # depth generation parameters
        self.crop_w = stereo_args["image_crop_w"] # , 64) # to deal with vignette effect, crop the image
        # self.crop_h = stereo_args["image_crop_h"] # , 32) # after cropping the size is (960, 512)
        self.input_w = stereo_args["image_input_w"] # , 512)
        self.input_h = stereo_args["image_input_h"] # , 256)
        self.visualize = stereo_args["visualize_depth"] # , True)
        self.colored_folder = stereo_args['colored_folder']

        # point cloud processing parameters
        self.mindist = stereo_args["pc_min_dist"] # , 2.5) # not filter if set to -1 
        self.maxdist = stereo_args["pc_max_dist"] # , 10.0) # not filter if set to -1
        self.maxhight = stereo_args["pc_max_height"] # , 2.0) # not filter if set to -1

        self.crop_h_low = stereo_args['image_crop_h_low']
        self.crop_h_high = stereo_args['image_crop_h_high']

        self.batch_size = common_args["batch_size"]
        self.worker_num = common_args["worker_num"]

        self.mask_file = stereo_args['mask_file']
        self.uncertainty_thresh = stereo_args['uncertainty_thresh']
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

    def load_model(self, model, modelname):
        preTrainDict = torch.load(modelname)
        model_dict = model.state_dict()
        # print 'preTrainDict:',preTrainDict.keys()
        # print 'modelDict:',model_dict.keys()
        preTrainDictTemp = {k:v for k,v in preTrainDict.items() if k in model_dict}

        if( 0 == len(preTrainDictTemp) ):
            # self.logger.info("Does not find any module to load. Try DataParallel version.")
            for k, v in preTrainDict.items():
                kk = k[7:]

                if ( kk in model_dict ):
                    preTrainDictTemp[kk] = v

            preTrainDict = preTrainDictTemp

        model_dict.update(preTrainDict)
        model.load_state_dict(model_dict)
        return model

    def crop_intrinsics(self):
        self.pu = self.pu - self.crop_w
        self.pv = self.pv - self.crop_h_low
        self.w = self.w - 2 * self.crop_w
        self.h = self.h - self.crop_h_low - self.crop_h_high

    def scale_intrinsics(self):
        scalex = float(self.input_w)/self.w
        scaley = float(self.input_h)/self.h
        self.focalx = self.focalx * scalex
        self.focaly = self.focaly * scaley
        self.pu = self.pu * scalex
        self.pv = self.pv * scaley
        self.fxbl = self.fxbl * scalex

    def crop_resize_atvmask(self):
        h, w = self.atvmask.shape
        self.atvmask = self.atvmask[self.crop_h_low:h-self.crop_h_high, self.crop_w:w-self.crop_w]
        self.atvmask = cv2.resize(self.atvmask,(self.input_w, self.input_h))

    def mask_boarder_points(self, maskw=10, maskh=10):
        self.atvmask[0:maskh, :] = False
        self.atvmask[-maskh:, :] = False
        self.atvmask[:, 0:maskw] = False
        self.atvmask[:, -maskw:] = False
        # import ipdb;ipdb.set_trace()

    def disp2vis(self, disp, scale=10,):
        '''
        disp: h x w float32 numpy array
        return h x w x 3 uint8 numpy array
        '''
        disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        return disp_color

    def save_as_points_file(self, disparity, colored_image, filename, uncertainty=None):
        # import ipdb;ipdb.set_trace()
        depth = self.fxbl / (disparity+1e-8)

        if uncertainty is not None:
            mask_unc = uncertainty < self.uncertainty_thresh
            mask = np.logical_and(self.atvmask, mask_unc)
        else:
            mask = self.atvmask

        point_array, color_array = depth_to_point_cloud(depth, self.focalx, self.focaly, self.pu, self.pv, 
                                                        self.mindist, self.maxdist, colored_image, mask)
        points_data = np.concatenate((point_array, color_array), axis=1)
        np.save(filename, points_data)

    def process(self, traj_root_folder, depth_output_folder=None, points_output_folder=None):
        self.load_dataset(traj_root_folder)
        # prepare output folder
        if depth_output_folder is not None:
            outdir = traj_root_folder + '/' + depth_output_folder
            if not isdir(outdir):
                mkdir(outdir)
                print('Create folder: {}'.format(outdir))
        if points_output_folder is not None:
            pcoutdir = traj_root_folder + '/' + points_output_folder
            if not isdir(pcoutdir):
                mkdir(pcoutdir)
                print('Create folder: {}'.format(pcoutdir))

        count = 0
        while True: # for all the images in the trajectory
            # starttime0 = time.time()
            try:
                sample = next(self.testDataiter)
            except StopIteration:
                break
            # print('Data load time {}'.format(time.time()-starttime0))

            # import ipdb;ipdb.set_trace()
            starttime = time.time()
            with torch.no_grad():        
                leftTensor = sample['img0'].cuda()
                rightTensor = sample['img1'].cuda()
                inputTensor = torch.cat((leftTensor, rightTensor), dim=1)
                output, output_unc = self.stereonet((inputTensor))
                torch.cuda.synchronize()        
            print ('{}. Stereo estimation forward time {}'.format(count, time.time()-starttime))
            colored_np = sample['imgc'].numpy()

            disp = output.cpu().squeeze(1).numpy() * 50 # 50 is the normalization factor used in training
            count += len(disp)

            if depth_output_folder is not None: # save the output files to the folder
                filenames = sample['filename0']
                for k in range(disp.shape[0]):
                    dispk = disp[k]
                    filestr = outdir+'/'+filenames[k].split('.')[0]
                    filestrpc = pcoutdir+'/'+filenames[k].split('.')[0]
                    np.save(filestr+'.npy',dispk)
                    if output_unc is not None:
                        unc = output_unc[k].squeeze().cpu().numpy()
                    else:
                        unc = None
                    self.save_as_points_file(dispk, colored_np[k], filestrpc+'.npy', uncertainty=unc)

                    # if self.visualize: # save visualization file to the folder
                    dispvis = self.disp2vis(dispk, scale=3)
                    cv2.imwrite(filestr+'.jpg',dispvis)

            if self.visualize:
                cv2.imshow('img', dispvis)
                cv2.waitKey(1)
            # import ipdb;ipdb.set_trace()

# python imgs2pointcloud.py --visualize-depth --image-input-w 512 --image-input-h 256
if __name__ == '__main__':

    # rospy.init_node("stereo_net", log_level=rospy.INFO)

    # rospy.loginfo("stereo_net_node initialized")
    node = StereoInference()
    # node.process(traj_root_folder='/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210805_slope2', 
    #             depth_output_folder='depth_left', points_output_folder='points_left')
    node.process(traj_root_folder='/cairo/arl_bag_files/SARA/arl0608/wanda_cmu_0', 
                depth_output_folder='depth_left', points_output_folder='points_left')
    # rospy.spin()

