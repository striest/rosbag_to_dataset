import cv2
import torch
import numpy as np

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters

# from PSM import stackhourglass as StereoNet
from StereoNet7 import StereoNet7 as StereoNet
import time

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
from math import pi, tan

def depth_to_point_cloud(depth, focalx, focaly, pu, pv, filtermin=-1, filtermax=-1):
    """
    Todo: change the hard stack to transformation matrix
    Convert depth image to point cloud based on intrinsic parameters
    :param depth: depth image
    :return: xyz point array
    """
    h, w = depth.shape
    depth64 = depth.astype(np.float64)
    wIdx = np.linspace(0, w - 1, w, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    hIdx = np.linspace(0, h - 1, h, dtype=np.float64) + 0.5 # put the optical center at the middle of the image
    u, v = np.meshgrid(wIdx, hIdx)

    if filtermax!=-1 and filtermin!=-1:
        mask = np.logical_and(depth>filtermin, depth<filtermax)
        depth64 = depth64[mask]
        depth = depth[mask]
        u = u[mask]
        v = v[mask]
        # print('Depth mask {} -> {}'.format(h*w, mask.sum()))

    x = (u - pu) * depth64 / focalx
    y = (v - pv) * depth64 / focaly
    x = x.astype(np.float32)
    y = y.astype(np.float32)

    points = np.stack([depth, x, y], axis=1) #rotate_points(depth, x, y, mode) # amigo: this is in NED coordinate
    return points


def coord_transform(points):
    # the R and T come from gound calibration
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
     [ 0.,         -0.99969192, -0.02482067],
     [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    return points_trans.transpose(1, 0)

def points_height_filter(points, maxhight):
    # import ipdb;ipdb.set_trace()
    # points = points.reshape(-1, 3)
    mask = points[:,2]<maxhight
    return points[mask, :]


def xyz_array_to_point_cloud_msg(points, timestamp=None):
    """
    Please refer to this ros answer about the usage of point cloud message:
        https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
    :param points:
    :param header:
    :return:
    """
    header = Header()
    header.frame_id = 'multisense'
    if timestamp is None:
        timestamp = rospy.Time().now()
    header.stamp = timestamp
    msg = PointCloud2()
    msg.header = header
    if len(points.shape)==3:
        msg.width = points.shape[0]
        msg.height = points.shape[1]
    else:
        msg.width = points.shape[0]
        msg.height = 1
    msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1), ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
    # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
    msg.is_dense = False
    xyz = points.astype(np.float32)
    msg.data = xyz.tostring()
    return msg



class StereoNode:
    def __init__(self, ):
        self.cv_bridge = CvBridge()

        # modelname = 'models/4_3_3_stereo_60000.pkl'
        modelname = 'models/5_5_4_stereo_30000.pkl'
        self.stereonet = StereoNet(group_norm=False)
        self.load_model(self.stereonet, modelname)
        self.stereonet.cuda()
        self.stereonet.eval()

        leftimg_sync = message_filters.Subscriber('/multisense/left/image_rect', Image)
        rightimg_sync = message_filters.Subscriber('/multisense/right/image_rect', Image)
        ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync], 1, 0.01)
        ts.registerCallback(self.handle_imgs)

        self.cloud_pub_ = rospy.Publisher('deep_cloud', PointCloud2, queue_size=1)

        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]

        # camera parameters
        self.w = rospy.get_param('~image_width', 1024)
        self.h = rospy.get_param('~image_height', 544)
        self.focalx = rospy.get_param('~focal_x', 477.6049499511719)
        self.focaly = rospy.get_param('~focal_y', 477.6049499511719)
        self.pu = rospy.get_param('~center_x', 499.5)
        self.pv = rospy.get_param('~center_y', 252.0)
        self.fxbl = rospy.get_param('~focal_x_baseline', 100.14994812011719)

        # depth generation parameters
        self.input_w = rospy.get_param('~image_input_w', 512)
        self.input_h = rospy.get_param('~image_input_h', 256)
        self.visualize = rospy.get_param('~visualize_depth', True)

        # point cloud processing parameters
        self.mindist = rospy.get_param('~pc_min_dist', 2.5) # not filter if set to -1 
        self.maxdist = rospy.get_param('~pc_max_dist', 10.0) # not filter if set to -1
        self.maxhight = rospy.get_param('~pc_max_height', 2.0) # not filter if set to -1

        # some flags to control the point cloud processing
        self.transform_ground = rospy.get_param('~pc_transform_ground', True)

        self.scale_intrinsics()


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

        # if ( 0 == len(preTrainDict) ):
        #     raise WorkFlow.WFException("Could not load model from %s." % (modelname), "load_model")

        # for item in preTrainDict:
        #     print("Load pretrained layer:{}".format(item) )
        model_dict.update(preTrainDict)
        model.load_state_dict(model_dict)
        return model

    def scale_imgs(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = cv2.resize(leftImg,(self.input_w, self.input_h))
        rightImg = cv2.resize(rightImg,(self.input_w, self.input_h))

        return {'left': leftImg, 'right': rightImg}

    def scale_intrinsics(self):
        scalex = float(self.input_w)/self.w
        scaley = float(self.input_h)/self.h
        self.focalx = self.focalx * scalex
        self.focaly = self.focaly * scaley
        self.pu = self.pu * scalex
        self.pv = self.pv * scaley
        self.fxbl = self.fxbl * scalex

    def scale_back(self, disparity):
        disparity = cv2.resize(disparity, (self.w, self.h))
        disparity = disparity * self.w / self.input_w
        return disparity

    def to_tensor(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        leftImg = leftImg.transpose(2,0,1)/float(255)
        rightImg = rightImg.transpose(2,0,1)/float(255)
        leftTensor = torch.from_numpy(leftImg).float()
        rightTensor = torch.from_numpy(rightImg).float()
        # rgbsTensor = torch.cat((leftTensor, rightTensor), dim=0)
        return {'left': leftTensor, 'right': rightTensor}


    def normalize(self, sample):
        leftImg, rightImg = sample['left'], sample['right']
        for t, m, s in zip(leftImg, self.mean, self.std):
            t.sub_(m).div_(s)
        for t, m, s in zip(rightImg, self.mean, self.std):
            t.sub_(m).div_(s)
        return {'left': leftImg, 'right': rightImg}

    def disp2vis(self, disp, scale=10,):
        '''
        disp: h x w float32 numpy array
        return h x w x 3 uint8 numpy array
        '''
        disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        # disp = np.tile(disp[:,:,np.newaxis], (1, 1, 3))
        # disp = cv2.resize(disp,(640,480))

        return disp_color

    def pub_pc2(self, disparity, timestamp):
        # import ipdb;ipdb.set_trace()
        starttime = time.time()
        depth = self.fxbl / disparity
        point_array = depth_to_point_cloud(depth, self.focalx, self.focaly, self.pu, self.pv, self.mindist, self.maxdist)
        print('depth_to_point_cloud time {}'.format(time.time()-starttime))
        # starttime = time.time()
        # if self.mindist != -1 and self.maxdist != -1:
        #     point_array = points_distance_filter(point_array, self.mindist, self.maxdist) # 2.5, 10
        # print('points_distance_filter time {}'.format(time.time()-starttime))
        starttime = time.time()
        if self.transform_ground:
            point_array = coord_transform(point_array)
        print('coord_transform time {}'.format(time.time()-starttime))
        starttime = time.time()
        if self.maxhight != -1:
            point_array = points_height_filter(point_array, self.maxhight)
        print('points_height_filter time {}'.format(time.time()-starttime))
        starttime = time.time()
        pc_msg = xyz_array_to_point_cloud_msg(point_array, timestamp)
        print('xyz_array_to_point_cloud_msg time {}'.format(time.time()-starttime))
        starttime = time.time()
        self.cloud_pub_.publish(pc_msg)

    def handle_imgs(self, leftmsg, rightmsg):
        # print 'img received..'
        starttime0 = time.time()
        left_image_np = self.cv_bridge.imgmsg_to_cv2(leftmsg, "bgr8")
        right_image_np = self.cv_bridge.imgmsg_to_cv2(rightmsg, "bgr8") 
        sample = {'left': left_image_np, 'right': right_image_np}
        if self.w != self.input_w:
            sample = self.scale_imgs(sample)
        sample = self.to_tensor(sample)
        sample = self.normalize(sample) 
        print('Preprocess time {}'.format(time.time()-starttime0))

        starttime = time.time()
        with torch.no_grad():        
            leftTensor = sample['left'].unsqueeze(0).cuda()
            rightTensor = sample['right'].unsqueeze(0).cuda()
            output = self.stereonet((leftTensor,rightTensor))
            torch.cuda.synchronize()        
        print ('Stereo estimation forward time {}'.format(time.time()-starttime))

        starttime1 = time.time()
        disp = output.cpu().squeeze().numpy() * 50
        print('data transfer time {}'.format(time.time()-starttime1))
        # if self.w != self.input_w:
        #     disp = self.scale_back(disp)
        # starttime15 = time.time()
        # print('Scale back time {}'.format(time.time()-starttime15))

        if self.visualize:
            dispvis = self.disp2vis(disp, scale=3)
            cv2.imshow('img', dispvis)
            cv2.waitKey(1)
        # import ipdb;ipdb.set_trace()

        starttime2 = time.time()
        self.pub_pc2(disp, leftmsg.header.stamp)
        print('PC pub time {}'.format(time.time()-starttime2))


if __name__ == '__main__':

    rospy.init_node("stereo_net", log_level=rospy.INFO)

    rospy.loginfo("stereo_net_node initialized")
    node = StereoNode()
    rospy.spin()

