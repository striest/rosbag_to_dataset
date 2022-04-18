import rospy
import numpy as np

from geometry_msgs.msg import Twist, TwistStamped

from rosbag_to_dataset.dtypes.base import Dtype

class TwistConvert(Dtype):
    """
    There's a bit of overloading here. This converter either:
    1. Converts a twist message into a 6D velocity/orientation
    2. Converts a twist message into a 2D throttle/steer (using linear.x, angular.z)
    """
    def __init__(self, mode='state', stamped=True):
        """
        Args:
            mode: One of {'state', 'action'}. How to interpret the twist command.
        """
        assert mode in {'state', 'action'}, "Expected mode to be one of ['state', 'action']. Got {}".format(mode)
        self.mode = mode
        self.stamped = stamped

    def N(self):
        return 2 if self.mode == 'action' else 6

    def rosmsg_type(self):
        if self.stamped:
            return TwistStamped
        else:
            return Twist

    def ros_to_numpy(self, msg):
        twist = msg
        if self.stamped:
            twist = msg.twist

        if self.mode == 'state':
            vx = twist.linear.x
            vy = twist.linear.y
            vz = twist.linear.z
            wx = twist.angular.x
            wy = twist.angular.y
            wz = twist.angular.z
            return np.array([vx, vy, vz, wx, wy, wz])
        elif self.mode == 'action':
            throttle = twist.linear.x
            steer = twist.angular.z
            return np.array([throttle, steer])

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
        np.save(filename+'/twist.npy', data)

if __name__ == "__main__":
    c1 = TwistConvert('state')
    c2 = TwistConvert('action')
    msg = TwistStamped()

    print(c1.ros_to_numpy(msg))
    print(c2.ros_to_numpy(msg))
