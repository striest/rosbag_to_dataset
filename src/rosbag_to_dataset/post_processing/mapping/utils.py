import ros_numpy
import numpy as np

import rospy
from std_msgs.msg import Header
from sensor_msgs.point_cloud2 import PointCloud2, PointField

from scipy.spatial.transform import Rotation

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

    # print points.shape[0], colors.shape[0]

    return points, colors

def xyz_array_to_point_cloud_msg(points, timestamp=None, frame_id='multisense', colorimg=None):
    """
    Please refer to this ros answer about the usage of point cloud message:
        https://answers.ros.org/question/234455/pointcloud2-and-pointfield/
    :param points:
    :param header:
    :return:
    """
    header = Header()
    header.frame_id = frame_id
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
    msg.is_bigendian = False
    # organized clouds are non-dense, since we have to use std::numeric_limits<float>::quiet_NaN()
    # to fill all x/y/z value; un-organized clouds are dense, since we can filter out unfavored ones
    msg.is_dense = False

    if colorimg is None:
        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), ]
        msg.point_step = 12
        msg.row_step = msg.point_step * msg.width
        xyz = points.astype(np.float32)
        msg.data = xyz.tostring()
    else:
        msg.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1), 
                      PointField('rgb', 12, PointField.UINT32, 1),]
        msg.point_step = 16
        msg.row_step = msg.point_step * msg.width
        xyzcolor = np.zeros( (points.shape[0], 1), \
        dtype={ 
            "names": ( "x", "y", "z", "rgba" ), 
            "formats": ( "f4", "f4", "f4", "u4" )} )
        xyzcolor["x"] = points[:, 0].reshape((-1, 1))
        xyzcolor["y"] = points[:, 1].reshape((-1, 1))
        xyzcolor["z"] = points[:, 2].reshape((-1, 1))
        color_rgba = np.zeros((points.shape[0], 4), dtype=np.uint8) + 255
        color_rgba[:,:3] = colorimg
        xyzcolor["rgba"] = color_rgba.view('uint32')
        msg.data = xyzcolor.tostring()

    return msg


def quat2SE(quat_data):
    '''
    quat_data: 7
    SE: 4 x 4
    '''
    SO = Rotation.from_quat(quat_data[3:7]).as_dcm()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    return SE

def SE2quat(SE_data):
    '''
    SE_data: 4 x 4
    quat: 7
    '''
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_data[0:3,0:3])
    pos_quat[:3] = SE_data[0:3,3].T
    return pos_quat

def SO2quat(SO_data):
    rr = Rotation.from_dcm(SO_data)
    return rr.as_quat()

def pose2motion(pose1, pose2, skip=0):
    '''
    pose1, pose2: [x, y, z, rx, ry, rz, rw]
    return motion: [x, y, z, rx, ry, rz, rw]
    '''
    pose1_SE = quat2SE(pose1)
    pose2_SE = quat2SE(pose2)
    motion = np.matrix(pose1_SE).I*np.matrix(pose2_SE)
    return SE2quat(motion)

