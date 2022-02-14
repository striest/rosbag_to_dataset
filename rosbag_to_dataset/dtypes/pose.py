import rospy
import numpy as np

from geometry_msgs.msg import Pose, PoseStamped

from rosbag_to_dataset.dtypes.base import Dtype

class PoseConvert(Dtype):
    """
    Convert a pose message into a 7d vec.
    """
    def __init__(self, zero_position=False, use_vel=True, stamped=False):
        self.zero_position = zero_position
        self.initial_position = None if self.zero_position else np.zeros(3)
        self.use_vel = use_vel
        self.stamped = stamped

    def N(self):
        return 7

    def rosmsg_type(self):
        return PoseStamped if self.stamped else Pose

    def ros_to_numpy(self, msg):
        pose = msg
        if self.stamped:
            pose = msg.pose

        if self.initial_position is None:
            self.initial_position = np.array([pose.position.x, pose.position.y, pose.position.z])

        p = [pose.position.x, pose.position.y, pose.position.z]
        q = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        res = np.array(p + q)
        res[:3] -= self.initial_position

        return res if self.use_vel else res[:7]

if __name__ == "__main__":
    c = OdometryConvert()
    msg = Odometry()

    print(c.ros_to_numpy(msg))
