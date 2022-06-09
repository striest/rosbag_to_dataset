import rospy
import numpy as np

from rosbag_to_dataset.msg import FloatStamped

from rosbag_to_dataset.dtypes.base import Dtype

class FloatStampedConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self):
        pass

    def N(self):
        return 1

    def rosmsg_type(self):
        return FloatStamped

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        return np.array(msg.data)

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
        np.save(filename+'/float.npy', data)

if __name__ == "__main__":
    c = FloatStampedConvert()
    msg = FloatStamped()

    print(c.ros_to_numpy(msg))
