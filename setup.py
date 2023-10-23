from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
  packages=['rosbag_to_dataset', 
            'rosbag_to_dataset.util', 
            'rosbag_to_dataset.converter', 
            'rosbag_to_dataset.config_parser', 
            'rosbag_to_dataset.dtypes',
            'rosbag_to_dataset.post_processing',
            'rosbag_to_dataset.post_processing.dataloader'
            ],
  package_dir={'': 'src'}
)

setup(**d)
