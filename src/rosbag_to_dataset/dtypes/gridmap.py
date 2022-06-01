import rospy
import numpy as np
import cv2

from grid_map_msgs.msg import GridMap

from rosbag_to_dataset.dtypes.base import Dtype

class GridMapConvert(Dtype):
    """
    Handle GridMap msgs (very similar to images)
    """
    def __init__(self, channels, size, fill_value=None):
        """
        Args:
            channels: The names of the channels to stack into an image.
            output_resolution: The size to rescale the image to
            fill_value: The value to look for if no data available at that point. Fill with 99th percentile value of data.
        """
        self.channels = channels
        self.size = size
        self.fill_value = fill_value

    def N(self):
        return {
                'data': [len(self.channels)] + self.size,
                'origin': [2],
                'resolution': [1],
                'width': [1],
                'height': [1]
            }

    def rosmsg_type(self):
        return GridMap

    def ros_to_numpy(self, msg):
#        assert isinstance(msg, self.rosmsg_type()), "Got {}, expected {}".format(type(msg), self.rosmsg_type())

        data_out = []

        origin = np.array([
            msg.info.pose.position.x - msg.info.length_x/2.,
            msg.info.pose.position.y - msg.info.length_y/2.
        ])

        map_width = np.array([msg.info.length_x])
        map_height = np.array([msg.info.length_y])

        res_x = []
        res_y = []

        for channel in self.channels:
            idx = msg.layers.index(channel)
            layer = msg.data[idx]
            height = layer.layout.dim[0].size
            width = layer.layout.dim[1].size
            data = np.array(list(layer.data), dtype=np.float32) #Why was hte data a tuple?
            data = data.reshape(height, width)

            data[~np.isfinite(data)] = self.fill_value

            data = cv2.resize(data, dsize=(self.size[0], self.size[1]), interpolation=cv2.INTER_AREA)
            
            data_out.append(data[::-1, ::-1]) #gridmaps index from the other direction.
            res_x.append(msg.info.length_x / data.shape[0])
            res_y.append(msg.info.length_y / data.shape[1])

        data_out = np.stack(data_out, axis=0)

        reses = np.concatenate([np.stack(res_x), np.stack(res_y)])
        assert max(np.abs(reses - np.mean(reses))) < 1e-4, 'got inconsistent resolutions between gridmap dimensions/layers. Check that grid map layes are same shape and that size proportional to msg size'
        output_resolution = np.mean(reses, keepdims=True)

        return {
                'data': data_out,
                'origin': origin,
                'resolution': output_resolution,
                'width': map_width,
                'height': map_height
                }

if __name__ == "__main__":
    c = ImageConvert(nchannels=1, output_resolution=[32, 32])
    msg = Image(width=64, height=64, data=np.arange(64**2).astype(np.uint8))

    print(c.ros_to_numpy(msg))
