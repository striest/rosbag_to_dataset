import rospy
import numpy as np
import cv2

from sensor_msgs.msg import Image

from rosbag_to_dataset.dtypes.base import Dtype

class ImageConvert(Dtype):
    """
    For image, we'll rescale and 
    """
    def __init__(self, nchannels, output_resolution):
        self.nchannels = nchannels
        self.output_resolution = output_resolution

    def N(self):
        return [self.nchannels] + self.output_resolution

    def rosmsg_type(self):
        return Image

    def ros_to_numpy(self, msg):
        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        data = np.frombuffer(msg.data, dtype=np.uint8)
        data = data.reshape(msg.height, msg.width, self.nchannels)
        data = cv2.resize(data, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)
        if self.nchannels == 1:
            data = np.expand_dims(data, axis=2)
        else:
            data = np.moveaxis(data, 2, 0) #Switch to channels-first
        data = data.astype(np.float32) / 255. #Convert to float.

        return data

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
