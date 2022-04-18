import rospy
import numpy as np

from ackermann_msgs.msg import AckermannDriveStamped

from rosbag_to_dataset.dtypes.base import Dtype

class AckermannDriveConvert(Dtype):
    """
    Convert an AckermannDrive cmd into a 2d action (throttle, steer)
    """
    def __init__(self, throttle='speed', steer='position'):
        """
        Args:
            throttle: one of {'speed', 'acceleration', 'jerk'}. The part of the cmd used for throttle.
            steer: one of {'position', 'speed'}, The part of the msg used for commands.
        """
        self.throttle_field = throttle
        self.steer_field = steer

    def N(self):
        return 2

    def rosmsg_type(self):
        return AckermannDriveStamped

    def ros_to_numpy(self, msg):
        if self.throttle_field == 'speed':
            throttle = msg.drive.speed
        elif self.throttle_field == 'acceleration':
            throttle = msg.drive.acceleration
        elif self.throttle_field == 'jerk':
            throttle = msg.drive.jerk

        if self.steer_field == 'position':
            steer = msg.drive.steering_angle
        elif self.steer_field == 'speed':
            steer = msg.drive.steering_angle_velocity
        
        res = np.array([throttle, steer])
        return res

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
        np.save(filename+'/drive.npy', data)

if __name__ == "__main__":
    c = AckermannDriveConvert()
    msg = AckermannDriveStamped()

    print(c.ros_to_numpy(msg))
