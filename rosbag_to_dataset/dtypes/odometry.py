import rospy
import numpy as np

from nav_msgs.msg import Odometry

from rosbag_to_dataset.dtypes.base import Dtype

class OdometryConvert(Dtype):
    """
    Convert an odometry message into a 13d vec.
    """
    def __init__(self):
        pass

    def N(self):
        return 13

    def rosmsg_type(self):
        return Odometry

    def ros_to_numpy(self, msg):
        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())
        p = [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z]
        q = [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z, msg.pose.pose.orientation.w]
        pdot = [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z]
        qdot = [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z]
        return np.array(p + q + pdot + qdot)

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
