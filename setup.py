from distutils.core import setup


from setuptools import find_packages, setup
import os 
from glob import glob 
package_name = 'rosbag_to_dataset'
data_files = [
        ('share/' + package_name, ['package.xml']),
    ]

setup(
    name='rosbag_to_dataset',
    version='0.1.0',
    packages=['rosbag_to_dataset',
              'rosbag_to_dataset.util',
              'rosbag_to_dataset.converter',
              'rosbag_to_dataset.config_parser',
              'rosbag_to_dataset.dtypes',
              'rosbag_to_dataset.post_processing'],
    package_dir={'': 'src'},
    install_requires=['setuptools'],
    zip_safe=True,
)