import yaml
import numpy as np

from collections import OrderedDict

from rosbag_to_dataset.dtypes.bool import BoolConvert
from rosbag_to_dataset.dtypes.float64 import Float64Convert
from rosbag_to_dataset.dtypes.odometry import OdometryConvert
from rosbag_to_dataset.dtypes.image import ImageConvert
from rosbag_to_dataset.dtypes.compressed_image import CompressedImageConvert
from rosbag_to_dataset.dtypes.ackermann_drive import AckermannDriveConvert
from rosbag_to_dataset.dtypes.twist import TwistConvert
from rosbag_to_dataset.dtypes.imu import ImuConvert
from rosbag_to_dataset.dtypes.pose import PoseConvert
from rosbag_to_dataset.dtypes.vector3 import Vector3Convert
from rosbag_to_dataset.dtypes.pointcloud import PointCloudConvert
# from rosbag_to_dataset.dtypes.float_stamped import FloatStampedConvert

class ConfigParser:
    """
    Class that reads in the spec of the rosbag to convert to a dataset.
    Expects input as a yaml file that generally looks like the following.

    data:
        "topic name str":
            type:<one of the supported types>
            folder:<output folder for this modality>
            N_per_step:<frequency factor based on dt>
            <option>:<value>
            ...
    dt: 0.1
    main_topic:<this frame is used to align the timestamp>
    """
    def __init__(self):
        pass

    def parse_from_fp(self, fp):
        x = yaml.safe_load(open(fp, 'r'))
        return self.parse(x)

    def parse(self, spec):
        data_converters = OrderedDict()
        outfolder = {}
        rates = {}

        for k,v in spec['data'].items():
            dtype = self.dtype_convert[spec['data'][k]['type']]
            converter = dtype(**spec['data'][k].get('options', {}))
            outfolder_k = v['folder'] if 'folder' in v.keys() else k
            data_converters[k] = converter
            outfolder[k] = outfolder_k
            if 'N_per_step' in v.keys():
                N = spec['data'][k]['N_per_step']
                rates[k] = spec['dt'] / N
            else:
                rates[k] = spec['dt']
        if 'main_topic' in spec:
            maintopic = spec['main_topic']
        else:
            maintopic = list(spec['data'].keys())[0] # use first topic in the yaml file
        return data_converters, outfolder, rates, spec['dt'], maintopic

    dtype_convert = {
        "AckermannDrive":AckermannDriveConvert,
        "Bool":BoolConvert,
        "CompressedImage":CompressedImageConvert,
        "Float64":Float64Convert,
        "Image":ImageConvert,
        "Imu":ImuConvert,
        "Odometry":OdometryConvert,
        "PointCloud":PointCloudConvert,
        "Pose":PoseConvert,
        "Twist":TwistConvert,
        "Vector3":Vector3Convert,
        # "FloatStamped":FloatStampedConvert,
    }


if __name__ == "__main__":
    fp = open('specs/wanda_cost.yaml')
    d = yaml.safe_load(fp)
    print("-----")
    print("Parsed YAML file:")
    print(d)
    print("-----")
    print("Type of parsed YAML file:")
    print(type(d))
    print("-----")
    parser = ConfigParser()
    x, p, r, dt, mt = parser.parse(d)
    print("Data converters:")
    print(x)
    print("-----")
    print("Out folder:")
    print(p)
    print("-----")
    print("Rates:")
    print(r)
    print("-----")
    print("dt:")
    print(dt)
    print("-----")
    print("Main topic:")
    print(mt)
    print("-----")