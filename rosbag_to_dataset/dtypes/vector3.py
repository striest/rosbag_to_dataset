import rospy
import numpy as np

from geometry_msgs.msg import Vector3, Vector3Stamped

from rosbag_to_dataset.dtypes.base import Dtype

class Vector3Convert(Dtype):
    """
    """
    def __init__(self, stamped=True):
        """
        """
        self.stamped = stamped

    def N(self):
        return 3

    def rosmsg_type(self):
        if self.stamped:
            return Vector3Stamped
        else:
            return Vector3

    def ros_to_numpy(self, msg):
        vec = msg
        if self.stamped:
            vec = msg.vector

        return np.array([vec.x, vec.y, vec.z])

if __name__ == "__main__":
    c1 = TwistConvert('state')
    c2 = TwistConvert('action')
    msg = TwistStamped()

    print(c1.ros_to_numpy(msg))
    print(c2.ros_to_numpy(msg))
