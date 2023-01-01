import rospy
import numpy as np

# from std_msgs.msg import Float64
from torch_mpc.msg import AffixlogKBMParameters

from rosbag_to_dataset.dtypes.base import Dtype

class AffixParamsConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self):
        pass

    def N(self):
        return 1

    def rosmsg_type(self):
        return AffixlogKBMParameters

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        return np.array([msg.a,msg.b,msg.c,msg.d,msg.e])

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
    c = Float64Convert()
    msg = Float64()

    print(c.ros_to_numpy(msg))
