import rospy
import numpy as np

from std_msgs.msg import Float32
from common_msgs.msg import Float32Stamped

from rosbag_to_dataset.dtypes.base import Dtype

class Float32Convert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self, stamped=False):
        self.stamped = stamped

    def N(self):
        return 1

    def rosmsg_type(self):
        return Float32Stamped if self.stamped else Float32

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        return np.array([msg.data])

if __name__ == "__main__":
    c = Float32Convert()
    msg = Float32()

    print(c.ros_to_numpy(msg))
