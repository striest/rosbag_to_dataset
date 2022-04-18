import rospy
import numpy as np
import ros_numpy

from sensor_msgs.msg import PointCloud2, PointField

from rosbag_to_dataset.dtypes.base import Dtype

class PointCloudConvert(Dtype):
    """
    Convert a pointcloud message to numpy
    Note that this one in particular needs some hacks to work.
    """
    def __init__(self, fields, max_num_points, fill_value=-1.):
        """
        Args:
            fields: List of fields in the pointcloud
            max_num_points: The max number of points in the cloud. Since we stack everything, this value needs to be set
            fill_value: What value to fill empty paces as
        """
        self.fields = fields
        self.max_num_points = max_num_points
        self.fill_value = fill_value

    def N(self):
        return [self.max_num_points, len(self.fields)]

    def rosmsg_type(self):
        return PointCloud2

    def ros_to_numpy(self, msg):
        #BIG HACK TO GET TO WORK WITH ROS_NUMPY
        #The issue is that the rosbag package uses a different class for messages, so we need to convert back to PointCloud2
        msg2 = PointCloud2()
        msg2.header = msg.header
        msg2.height = msg.height
        msg2.width = msg.width
        msg2.fields = msg.fields
        msg2.is_bigendian = msg.is_bigendian
        msg2.point_step = msg.point_step
        msg2.row_step = msg.row_step
        msg2.data = msg.data
        msg2.is_dense = msg.is_dense

        pts = ros_numpy.numpify(msg2)
        pts = np.stack([pts[f].flatten() for f in self.fields], axis=-1)

        out = np.ones([self.max_num_points, len(self.fields)]) * self.fill_value
        out[:pts.shape[0]] = pts

        return out

    def save_file_one_msg(self, data, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data is stored frame by frame like image or point cloud
        """
        return self.ros_to_numpy(data)

    def save_file(self, data, filename):
        """
        Save the data to hard drive.
        This function should be implemented where the data of the whole trajectory is stored in one file, like imu, odometry, etc.
        """
        np.save(filename+'/pointcloud.npy', data)

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
