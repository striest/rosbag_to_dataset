import rospy
import numpy as np

from common_msgs.msg import BoolStamped

from rosbag_to_dataset.dtypes.base import Dtype

class BoolConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self):
        pass

    def N(self):
        return 1

    def rosmsg_type(self):
        return BoolStamped

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        return np.array(msg.data)

if __name__ == "__main__":
    c = Float64Convert()
    msg = Float64()

    print(c.ros_to_numpy(msg))
