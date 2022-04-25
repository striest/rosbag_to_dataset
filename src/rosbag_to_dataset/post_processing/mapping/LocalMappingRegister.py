#!/usr/bin/env python3

# from importlib_metadata import files
import numpy as np
import cv2

from ScrollGrid import ScrollGrid
from GridFilter import GridFilter

from scipy.spatial.transform import Rotation
import time
from arguments import get_args
import os
from os import listdir
from os import mkdir
from os.path import isdir, dirname, realpath
# from utils import pointcloud2_to_xyzrgb_array, xyz_array_to_point_cloud_msg



def coord_transform(points, R, T):
    points_trans = np.matmul(R.transpose(1,0), points.transpose(1,0)) - T
    return points_trans.transpose(1, 0)

def transform_ground(points):
    # the R and T come from gound calibration
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    # starttime = time.time()
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    points_trans = np.matmul(R, points.transpose(1,0)) + T
    # rospy.loginfo("transform ground pt num {} time {}".format(points.shape[0], time.time()-starttime))
    return points_trans.transpose(1, 0)


class LocalMappingRegisterNode(object):
    def __init__(self):
        self.curdir = dirname(realpath(__file__))
        args = get_args()
        self.parse_params(args)

        self.xyz_register = None
        self.color_register = None

        self.localmap = ScrollGrid(self.resolution, (self.min_x, self.max_x, self.min_y, self.max_y))
        self.gridfilter = GridFilter(self.resolution, (self.min_x, self.max_x, self.min_y, self.max_y, -1.0, 2.0))

    def parse_params(self, args):
        self.resolution = args.resolution  # ', 0.05)
        self.min_x = args.min_x  # ', 0.0)
        self.max_x = args.max_x  # ', 10.0)
        self.min_y = args.min_y  # ', -5.)
        self.max_y = args.max_y  # ', 5.)
        self.max_points_num = args.max_points_num  # ', 1000000) # throw away old frames
        self.visualize_maps = args.visualize_maps  # ', True)

    def load_data(self, trajectory_root_folder, points_folder = 'points_left', vo_folder = 'tartanvo_odom'):
        pointsfolder = trajectory_root_folder + '/' + points_folder
        filenames = listdir(pointsfolder)
        filenames = [ff for ff in filenames if ff.endswith('.npy')]
        filenames.sort()
        pointsfiles = [(pointsfolder +'/'+ ff) for ff in filenames]

        vofolder = trajectory_root_folder + '/' + vo_folder
        motions = np.load(vofolder + '/motions.npy')
        assert len(pointsfiles)==motions.shape[0]+1, "Points and odoms are not consistent! "

        return pointsfiles, motions, filenames
        
    def points_filter(self, points_3d, points, colors):
        # import ipdb;ipdb.set_trace()
        # points = points.reshape(-1, 3)
        mask_x = points_3d[:,0] > self.min_x - 1
        mask_y = points_3d[:,1] > self.min_y - 1
        mask_y2 = points_3d[:,1] < self.max_y + 1
        mask = np.logical_and(mask_x, mask_y)
        mask = np.logical_and(mask, mask_y2)
        points_filter = points[mask, :]
        colors_filter = colors[mask, :]
        points_3d_filterd = points_3d[mask, :]
        print('Points filter: {} - {}'.format(points_3d.shape[0], points_3d_filterd.shape[0]))
        return points_3d_filterd[:self.max_points_num, :], points_filter[:self.max_points_num, :], colors_filter[:self.max_points_num, :]
        # return points_3d_filterd, points_filter, colors_filter

    def savefiles(self, heightmap, rgbmap, heightmap_vis, rgbmap_vis, heightoutdir, rgboutdir, filestr):
        np.save(heightoutdir + '/' + filestr + '.npy', heightmap)
        np.save(rgboutdir + '/' + filestr + '.npy', rgbmap)
        cv2.imwrite(heightoutdir + '/' + filestr + '.png', heightmap_vis)
        cv2.imwrite(rgboutdir + '/' + filestr + '.png', rgbmap_vis)

    def process(self, traj_root_folder, heightmap_output_folder=None, rgbmap_output_folder=None):
        if heightmap_output_folder is not None:
            heightoutdir = traj_root_folder + '/' + heightmap_output_folder
            if not isdir(heightoutdir):
                mkdir(heightoutdir)
                print('Create folder: {}'.format(heightoutdir))
        if rgbmap_output_folder is not None:
            rgboutdir = traj_root_folder + '/' + rgbmap_output_folder
            if not isdir(rgboutdir):
                mkdir(rgboutdir)
                print('Create folder: {}'.format(rgboutdir))

        pointsfiles, motions, filenames = self.load_data(traj_root_folder)
        for k in range(len(filenames)):
            starttime = time.time()
            points = np.load(pointsfiles[k])
            xyz_array = points[:,:3]
            color_array = points[:,3:].astype(np.uint8)
            # print("Frame {} points {}".format(k, points.shape[0]))
            if k>0:
                # import ipdb;ipdb.set_trace()
                T = motions[k-1,:3].reshape(3,1)
                quat = motions[k-1,3:]
                R = Rotation.from_quat(quat).as_matrix()
                points_trans = coord_transform(self.xyz_register, R, T)
                self.xyz_register = np.concatenate((xyz_array, points_trans),axis=0)
                self.color_register = np.concatenate((color_array, self.color_register),axis=0)
                xyz_register_ground = transform_ground(self.xyz_register)

                xyz_register_ground, self.xyz_register, self.color_register = self.points_filter(xyz_register_ground, self.xyz_register, self.color_register)
                
                # # filter by density 
                # # points_ordered = xyz_register_ground[xyz_register_ground[:,0].argsort()]
                # # filter_mask = self.gridfilter.grid_filter_multicore(points_ordered, corenum=8)
                # filter_mask = self.gridfilter.grid_filter(xyz_register_ground)
                # xyz_register_ground = xyz_register_ground[filter_mask, :]
                # self.xyz_register = self.xyz_register[filter_mask, :]
                # self.color_register = self.color_register[filter_mask, :]
                # print("GridFilter: {} - > {}".format(filter_mask.shape[0], xyz_register_ground.shape[0]))

            else:
                self.xyz_register = xyz_array
                self.color_register = color_array
                xyz_register_ground = transform_ground(self.xyz_register)


            self.localmap.pc_to_map(xyz_register_ground, self.color_register)
            self.localmap.inflate_maps()

            heightmap = self.localmap.get_heightmap()
            rgbmap = self.localmap.get_rgbmap()
            heightmap_vis = self.localmap.get_vis_heightmap()
            rgbmap_vis = self.localmap.get_vis_rgbmap()

            if heightoutdir is not None and rgboutdir is not None:
                self.savefiles(heightmap, rgbmap, heightmap_vis, rgbmap_vis, \
                                heightoutdir, rgboutdir, filenames[k].split('.npy')[0])

            if self.visualize_maps:
                self.localmap.show_heightmap()
                self.localmap.show_colormap()

            print("Frame {} new points {}, map points {}, time: {}".format(k, points.shape[0], xyz_register_ground.shape[0], time.time()-starttime))

        # import ipdb;ipdb.set_trace()

# python LocalMappingRegister.py --visualize-maps
if __name__ == '__main__':

    node = LocalMappingRegisterNode()
    # node.process('/home/mateo/Data/SARA/TartanDriveCost/Trajectories/000009', heightmap_output_folder='height_map', rgbmap_output_folder='rgb_map')

    # node.process('/home/mateo/rosbag_to_dataset/test_output/20210903_42', heightmap_output_folder='height_map', rgbmap_output_folder='rgb_map')

    dataset_folder = '/home/mateo/Data/SARA/TartanDriveCost/'
    trajectories_dir = os.path.join(dataset_folder, "Trajectories")
    traj_dirs = list(filter(os.path.isdir, [os.path.join(trajectories_dir,x) for x in sorted(os.listdir(trajectories_dir))]))

    for i, d in enumerate(traj_dirs):
        print(f"Processing trajectory in {d} for height maps and rgb maps")
        node.process(d, heightmap_output_folder='height_map', rgbmap_output_folder='rgb_map')