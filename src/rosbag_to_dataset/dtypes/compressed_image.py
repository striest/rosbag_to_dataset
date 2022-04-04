import rospy
import numpy as np
import cv2

from sensor_msgs.msg import CompressedImage

from rosbag_to_dataset.dtypes.base import Dtype

class CompressedImageConvert(Dtype):
    """
    For image, we'll rescale and 
    """
    def __init__(self, nchannels, output_resolution, is_compressed=False):
        """
        Args:
            nchannels: The number of channels in the image
            output_resolution: The size to rescale the image to
            aggregate: One of {'none', 'bigendian', 'littleendian'}. Whether to leave the number of channels alone, or to combine with MSB left-to-right or right-to-left respectively.
            empty_value: A value signifying no data. Replace with the 99th percentile value to make learning simpler.
        """
        self.nchannels = nchannels
        self.output_resolution = output_resolution

    def N(self):
        return [self.nchannels] + self.output_resolution

    def rosmsg_type(self):
        return CompressedImage

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        data = np.frombuffer(msg.data, dtype=np.uint8).copy()

        data = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)

        data = cv2.resize(data, dsize=(self.output_resolution[0], self.output_resolution[1]), interpolation=cv2.INTER_AREA)

        data = np.moveaxis(data, 2, 0) #Switch to channels-first

        data = data.astype(np.float32) / (255.)

        return data

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
