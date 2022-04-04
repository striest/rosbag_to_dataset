import rospy
import numpy as np

from geometry_msgs.msg import Vector3, Vector3Stamped

from rosbag_to_dataset.dtypes.base import Dtype

class Vector3Convert(Dtype):
    """
    Convert a vector3
    """
    def __init__(self, stamped=False):
        self.stamped = stamped

    def N(self):
        return 3

    def rosmsg_type(self):
        return Vector3Stamped if self.stamped else Vector3

    def ros_to_numpy(self, msg):
        vec = msg.vector if self.stamped else msg
        return np.array([vec.x, vec.y, vec.z])

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
