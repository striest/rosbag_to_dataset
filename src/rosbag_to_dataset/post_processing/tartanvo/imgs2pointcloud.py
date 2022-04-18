import cv2
import torch
import numpy as np
import time
from os import mkdir
from os.path import isdir, dirname, realpath

# from PSM import stackhourglass as StereoNet
from StereoNet7 import StereoNet7 as StereoNet
from TrajFolderDataset import TrajFolderDataset
from arguments import get_args

from torch.utils.data import DataLoader


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

class Depth2Points:
    def __init__(self):
        '''
        Project depth image to points
        '''

class StereoInference:
    def __init__(self):
        '''
        traj_root_folder: the root folder for the input trajectory
        depth_output_folder: if given, the data will be output to the folder
        '''
        self.curdir = dirname(realpath(__file__))
        args = get_args()
        self.parse_params(args)

        self.stereonet = StereoNet(group_norm=False)
        self.load_model(self.stereonet, self.curdir+'/models/'+self.modelname)
        self.stereonet.cuda()
        self.stereonet.eval()

        # self.mean=[0.485, 0.456, 0.406]
        # self.std=[0.229, 0.224, 0.225]

        self.crop_intrinsics()
        self.scale_intrinsics()

        self.atvmask = np.load(self.curdir+'/atvmask.npy')
        self.crop_resize_atvmask()
        self.atvmask = self.atvmask < 10 # a threshold

    def load_dataset(self, traj_root_folder):
        testDataset = TrajFolderDataset(traj_root_folder, leftfolder='image_left', rightfolder='image_right', colorfolder='image_left_color', 
                                        forvo=False,  crop_w=self.crop_w, crop_h=self.crop_h, resize_w=self.input_w, resize_h=self.input_h)
        testDataloader = DataLoader(testDataset, batch_size=self.batch_size, 
                                        shuffle=False, num_workers=self.worker_num)
        self.testDataiter = iter(testDataloader)

    def parse_params(self, args):
        self.modelname = args.model_name #'models/5_5_4_stereo_30000.pkl'

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

        # point cloud processing parameters
        self.mindist = args.pc_min_dist # , 2.5) # not filter if set to -1 
        self.maxdist = args.pc_max_dist # , 10.0) # not filter if set to -1
        self.maxhight = args.pc_max_height # , 2.0) # not filter if set to -1

        self.batch_size = args.batch_size
        self.worker_num = args.worker_num
        # # some flags to control the point cloud processing
        # self.transform_ground = args.pc_transform_ground # , True)

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
        self.pv = self.pv - self.crop_h
        self.w = self.w - 2 * self.crop_w
        self.h = self.h - 2 * self.crop_h

    def scale_intrinsics(self):
        scalex = float(self.input_w)/self.w
        scaley = float(self.input_h)/self.h
        self.focalx = self.focalx * scalex
        self.focaly = self.focaly * scaley
        self.pu = self.pu * scalex
        self.pv = self.pv * scaley
        self.fxbl = self.fxbl * scalex

    def crop_resize_atvmask(self):
        self.atvmask = self.atvmask[self.crop_h:-self.crop_h, self.crop_w:-self.crop_w]
        self.atvmask = cv2.resize(self.atvmask,(self.input_w, self.input_h))

    def crop_intrinsics(self):
        self.pu = self.pu - self.crop_w
        self.pv = self.pv - self.crop_h
        self.w = self.w - 2 * self.crop_w
        self.h = self.h - 2 * self.crop_h

    def scale_intrinsics(self):
        scalex = float(self.input_w)/self.w
        scaley = float(self.input_h)/self.h
        self.focalx = self.focalx * scalex
        self.focaly = self.focaly * scaley
        self.pu = self.pu * scalex
        self.pv = self.pv * scaley
        self.fxbl = self.fxbl * scalex

    def crop_resize_atvmask(self):
        self.atvmask = self.atvmask[self.crop_h:-self.crop_h, self.crop_w:-self.crop_w]
        self.atvmask = cv2.resize(self.atvmask,(self.input_w, self.input_h))

    def disp2vis(self, disp, scale=10,):
        '''
        disp: h x w float32 numpy array
        return h x w x 3 uint8 numpy array
        '''
        disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        return disp_color

    def save_as_points_file(self, disparity, colored_image, filename):
        # import ipdb;ipdb.set_trace()
        depth = self.fxbl / disparity
        point_array, color_array = depth_to_point_cloud(depth, self.focalx, self.focaly, self.pu, self.pv, 
                                                        self.mindist, self.maxdist, colored_image, self.atvmask)
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

        while True: # for all the images in the trajectory
            starttime0 = time.time()
            try:
                sample = self.testDataiter.next()
            except StopIteration:
                break
            # print('Data load time {}'.format(time.time()-starttime0))

            # import ipdb;ipdb.set_trace()
            starttime = time.time()
            with torch.no_grad():        
                leftTensor = sample['img0'].cuda()
                rightTensor = sample['img1'].cuda()
                inputTensor = torch.cat((leftTensor, rightTensor), dim=1)
                output = self.stereonet((inputTensor))
                # import ipdb;ipdb.set_trace()
            #     torch.cuda.synchronize()        
            # print ('Stereo estimation forward time {}'.format(time.time()-starttime))
            colored_np = sample['imgc'].numpy()
            disp = output.cpu().squeeze(1).numpy() * 50 # 50 is the normalization factor used in training
            
            if depth_output_folder is not None: # save the output files to the folder
                filenames = sample['filename0']
                for k in range(disp.shape[0]):
                    dispk = disp[k]
                    filestr = outdir+'/'+filenames[k].split('.')[0]
                    filestrpc = pcoutdir+'/'+filenames[k].split('.')[0]
                    np.save(filestr+'.npy',dispk)
                    self.save_as_points_file(dispk, colored_np[k], filestrpc+'.npy')

                    if self.visualize: # save visualization file to the folder
                        dispvis = self.disp2vis(dispk, scale=3)
                        cv2.imwrite(filestr+'.jpg',dispvis)

            if self.visualize:
                cv2.imshow('img', dispvis)
                cv2.waitKey(1)
            # import ipdb;ipdb.set_trace()

# python imgs2pointcloud.py --visualize --image-input-w 640 --image-input-h 448
if __name__ == '__main__':

    # rospy.init_node("stereo_net", log_level=rospy.INFO)

    # rospy.loginfo("stereo_net_node initialized")
    node = StereoInference()
    node.process(traj_root_folder='/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_298', 
                depth_output_folder='depth_left', points_output_folder='points_left')
    # rospy.spin()

