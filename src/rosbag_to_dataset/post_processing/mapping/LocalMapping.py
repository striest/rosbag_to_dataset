#!/usr/bin/env python

import numpy as np

import rospy
from sensor_msgs.point_cloud2 import PointCloud2, PointField
import ros_numpy

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge

from grid_map_msgs.msg import GridMap
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, Header

from ScrollGrid import ScrollGrid

import time


def pointcloud2_to_xyzrgb_array(cloud_msg, remove_nans=True, dtype=np.float):
    '''Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
    '''
    # remove crap points
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(cloud_msg)
    if remove_nans:
        mask = np.isfinite(cloud_array['x']) & np.isfinite(cloud_array['y']) & np.isfinite(cloud_array['z'])
        cloud_array = cloud_array[mask]
    
    # pull out x, y, and z values
    points = np.zeros(cloud_array.shape + (3,), dtype=dtype)
    points[...,0] = cloud_array['x']
    points[...,1] = cloud_array['y']
    points[...,2] = cloud_array['z']

    color = cloud_array['rgb']
    dt = np.dtype((np.int32, {'r':(np.uint8,0),'g':(np.uint8,1),'b':(np.uint8,2), 'a':(np.uint8,3)}))
    color = color.view(dtype=dt)
    colors = np.zeros(cloud_array.shape + (3,), dtype=np.uint8)
    colors[:,0] = color['r']
    colors[:,1] = color['g']
    colors[:,2] = color['b']

    print points.shape[0], colors.shape[0]

    return points, colors


def coord_transform(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    # R = np.array([[0.00610748, -0.24598853, 0.9692535],
    #               [-0.99969192, -0.02482067,0.],
    #               [0.02405752,  -0.96895489, -0.24606434, ]] )
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )

    R0 = np.array([[0., 0., 1.],
                  [1., 0., 0],
                  [0., 1., 0.]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = points.transpose(1,0)
    points_trans = np.matmul(R0, points_trans)
    points_trans = np.matmul(R, points_trans) + T
    return points_trans.transpose(1, 0)

def points_height_filter(points, minx, colorimg=None):
    mask = points[:,0]> minx
    points_filter = points[mask, :]
    colorimg_filter = colorimg[mask, :]
    return points_filter, colorimg_filter

class LocalMappingNode(object):
    def __init__(self):

        rospy.Subscriber('/statistical_outlier_removal/output', PointCloud2, self.handle_pc, queue_size=1)

        self.resolution = rospy.get_param('~resolution', 0.05)
        self.min_x = rospy.get_param('~min_x', 2.0)
        self.max_x = rospy.get_param('~max_x', 8.0)
        self.min_y = rospy.get_param('~min_y', -4.)
        self.max_y = rospy.get_param('~max_y', 4.)
        self.visualize_maps = rospy.get_param('~visualize_maps', True)
        self.transform_ground = rospy.get_param('~pc_transform_ground', False) # used for colored pc directly from the multisense

        self.localmap = ScrollGrid(self.resolution, (self.min_x, self.max_x, self.min_y, self.max_y))

        self.height_pub_ = rospy.Publisher('local_height_map', GridMap, queue_size=1)
        self.rgb_pub_ = rospy.Publisher('local_rgb_map', Image, queue_size=1)
        self.cvbridge = CvBridge()

    def to_gridmap_rosmsg(self, data, stamp):
        '''
        data: heightmap: h x w x 2, first channel min-height, second channel max-height
        '''
        msg = GridMap()
        msg.info.header.stamp = stamp
        msg.info.header.frame_id = "multisense"
        msg.info.resolution = self.resolution
        msg.info.length_x = self.max_x - self.min_x
        msg.info.length_y = self.max_y - self.min_y
        msg.layers.append("height")

        data_msg_low = Float32MultiArray()
        data_msg_low.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize)]
        data_msg_low.data = data[:,:,0].reshape([1, -1])[0].tolist()

        data_msg_high = Float32MultiArray()
        data_msg_high.layout.dim = [MultiArrayDimension("column_index", data.shape[0], data.shape[0] * data.dtype.itemsize), MultiArrayDimension("row_index", data.shape[1], data.shape[1] * data.dtype.itemsize)]
        data_msg_high.data = data[:,:,1].reshape([1, -1])[0].tolist()

        msg.data = [data_msg_low, data_msg_high]
        return msg

    def handle_pc(self, msg):
        xyz_array, color_array = pointcloud2_to_xyzrgb_array(msg)
        if self.transform_ground:
            xyz_array = coord_transform(xyz_array)
            # filter the point cloud, filter out the point on the ATV
            xyz_array, color_array = points_height_filter(xyz_array, 1.5, color_array)

        self.localmap.pc_to_map(xyz_array, color_array)

        heightmap = self.localmap.get_heightmap()
        grid_map_msg = self.to_gridmap_rosmsg(heightmap, msg.header.stamp) # use the max height for now TODO
        self.height_pub_.publish(grid_map_msg)
        rgbmap = self.localmap.get_rgbmap()
        imgmsg = self.cvbridge.cv2_to_imgmsg(rgbmap, encoding="rgb8")
        self.rgb_pub_.publish(imgmsg)

        print('Receive {} points'.format(xyz_array.shape[0]))
        # import ipdb;ipdb.set_trace()

if __name__ == '__main__':

    rospy.init_node("local_mapping", log_level=rospy.INFO)

    rospy.loginfo("local_mapping node initialized")
    node = LocalMappingNode()
    r = rospy.Rate(10)
    while not rospy.is_shutdown(): # loop just for visualization
        if node.visualize_maps:
            node.localmap.show_heightmap()
            node.localmap.show_colormap()
        r.sleep()


        




        # model_name = rospy.get_param('~model_name', '43_6_2_vonet_30000.pkl')
        # w = rospy.get_param('~image_width', 1024)
        # h = rospy.get_param('~image_height', 544)
        # fx = rospy.get_param('~focal_x', 477.6049499511719)
        # fy = rospy.get_param('~focal_y', 477.6049499511719)
        # ox = rospy.get_param('~center_x', 499.5)
        # oy = rospy.get_param('~center_y', 252.0)
        # self.cam_intrinsics = [w, h, fx, fy, ox, oy]
        # self.blxfx = 100.14994812011719

        # mean = [0.485, 0.456, 0.406]
        # std = [0.229, 0.224, 0.225]
        # self.cv_bridge = CvBridge()
        # self.transform = Compose([ResizeData((448,843)), 
        #                           CropCenter((448, 640)), 
        #                           DownscaleFlow(), 
        #                           Normalize(mean=mean,std=std,keep_old=True), 
        #                           ToTensor()]) 
        # self.intrinsic = make_intrinsics_layer(w, h, fx, fy, ox, oy)
        # self.tartanvo = TartanSVO(model_name)

        # self.map_pub = rospy.Publisher("local_heightmap", Image, queue_size=10)

        # pc_sync = message_filters.Subscriber('/deep_cloud', Image)
        # odom_sync = message_filters.Subscriber('/tartanvo_pose', Image)
        # ts = message_filters.ApproximateTimeSynchronizer([leftimg_sync, rightimg_sync], 1, 0.01)
        # ts.registerCallback(self.handle_imgs)
        # rospy.Subscriber('cam_info', CameraInfo, self.handle_caminfo)

        # self.tf_broadcaster = TransformBroadcaster()

        # # rospy.Subscriber('rgb_image', Image, self.handle_img)
        # # rospy.Subscriber('vo_scale', Float32, self.handle_scale)

        # self.last_left_img = None
        # self.last_right_img = None
        # self.pose = np.matrix(np.eye(4,4))
        # # self.scale = 1.0
