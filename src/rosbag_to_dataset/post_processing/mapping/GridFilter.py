import numpy as np
from math import ceil
import time
from cscrollgrid import voxel_filter
from multiprocessing import Pool

FLOATMAX = 1000000.0

class GridFilter(object):
    def __init__(self, 
                 resolution, 
                 filter_range,
                 ):
        '''
        resoluation: grid size in meter
        range: (xmin, xmax, ymin, ymax) in meter, 
               fiter the points out of the range [(xmin, ymin, zmin), (xmax, ymax, zmax)))
        
        '''

        self.resolution = resolution
        self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax  = filter_range
        # the (xmin, ymin) is located at the center of a grid
        # so if the (center_pt - min_pt) can be devided by reso, the center_pt will also be located at the center of a grid
        self.xnum = int(ceil((self.xmax - self.xmin)/self.resolution)) 
        self.ynum = int(ceil((self.ymax - self.ymin)/self.resolution))
        self.znum = int(ceil((self.zmax - self.zmin)/self.resolution))

        print('GridFilter initialized, resolution {}, range {}, shape {}'.format(self.resolution, filter_range, (self.xnum, self.ynum, self.znum)))


    def pc_xy_to_grid_ind(self, x, y, z):
        xind = np.int((x - self.xmin)/self.resolution)
        yind = np.int((y - self.ymin)/self.resolution)
        zind = np.int((z - self.zmin)/self.resolution)
        return xind, yind, zind

    def pc_coord_to_grid_ind(self, pc_coord):
        return ((pc_coord-np.array([self.xmin, self.ymin, self.zmin]))/self.resolution).astype(np.int32)

    def grid_ind_center(self, ind):
        return ind * self.resolution + np.array([[self.xmin, self.ymin, self.zmin]])

    def dist(self, sources, targets):
        '''
        sources: N x 3 numpy array
        targets: N x 3 numpy array
        '''
        return np.linalg.norm(sources - targets, axis=1) 

    def grid_filter(self, points):
        '''
        points: N x 3 numpy array
        return: M index representing the remaining points after filtering
        '''
        starttime = time.time()
        # import ipdb; ipdb.set_trace()
        grid_inds = self.pc_coord_to_grid_ind(points) # one frame: 0.008s
        # print('1',time.time()-starttime)

        grid_centers = self.grid_ind_center(grid_inds) # 0.002s
        points_dist = self.dist(points, grid_centers) # 0.004s
        # print('2',time.time()-starttime)

        grid_inds = grid_inds.astype(np.uint16)
        points_dist = points_dist.astype(np.float32)

        mindist = np.ones((self.xnum, self.ynum, self.znum)).astype(np.float32) * FLOATMAX
        mininds = np.ones((self.xnum, self.ynum, self.znum)).astype(np.int32)
        resmask = np.zeros(points.shape[0]).astype(np.uint8)
        res_points_mask = voxel_filter(grid_inds, points_dist, mindist, mininds, resmask, self.xnum, self.ynum, self.znum)
        # print('3',time.time()-starttime)

        # mindist = np.ones((self.xnum, self.ynum, self.znum),dtype=np.float32) * FLOATMAX # 0.02s
        # mininds = np.ones((self.xnum, self.ynum, self.znum),dtype=np.int32) * -1
        # res_points_mask = np.zeros((points.shape[0]), np.uint8)
        # for i, ind in enumerate(grid_inds):
        #     if ind[0]>=0 and ind[0]<self.xnum \
        #         and ind[1]>=0 and ind[1]<self.ynum \
        #         and ind[2]>=0 and ind[2]<self.znum:
        #         dist = points_dist[i]
        #         if dist < mindist[ind[0], ind[1], ind[2]]:
        #             if mindist[ind[0], ind[1], ind[2]] < FLOATMAX -1:
        #                 res_points_mask[mininds[ind[0], ind[1], ind[2]]] = 0 # unset the mask for the previous point
        #             mindist[ind[0], ind[1], ind[2]] = dist
        #             mininds[ind[0], ind[1], ind[2]] = i
        #             res_points_mask[i] = 1

        print('Grid filter convert time: {}, filter points {} -> {}'.format(time.time()-starttime, points.shape[0], res_points_mask.sum()))

        # import ipdb;ipdb.set_trace()
        return res_points_mask==1

    def grid_filter_multicore(self, points, corenum = 8):
        '''
        Devide the points into subsets and run them on multi-core in parallel
        points: N x 3 numpy array
        return: M index representing the remaining points after filtering
        '''
        
        pointsnum = points.shape[0]
        pointsnum_div = np.linspace(0,pointsnum, corenum+1).astype(np.int32)
        params = [points[pointsnum_div[k]:pointsnum_div[k+1],:] for k in range(corenum)]
        starttime = time.time()
        with Pool(corenum) as p:
            res = (p.map(self.grid_filter, params))
            print (len(res))

        newpoints = []
        for k in range(corenum): # combine the points from each subset together
            newpoints.extend(params[k][res[k]])

        res_points_mask = self.grid_filter(np.array(newpoints))

        print('Multi Core Grid filter convert time: {}, filter points {} -> {}'.format(time.time()-starttime, points.shape[0], res_points_mask.sum()))
        # import ipdb;ipdb.set_trace()

        return res_points_mask==1


# from utils import pointcloud2_to_xyzrgb_array, xyz_array_to_point_cloud_msg
# from sensor_msgs.point_cloud2 import PointCloud2
# import rospy
# gridfilter = GridFilter(0.05, (-1., 1., -1., 1., 1, 4))
# cloud_pub_ = rospy.Publisher('pc_filter', PointCloud2, queue_size=1)

# def handle_pc(pc_msg):
#     # import ipdb;ipdb.set_trace()
#     xyz_array, color_array = pointcloud2_to_xyzrgb_array(pc_msg)
#     points_mask = gridfilter.grid_filter(xyz_array)
#     points = xyz_array[points_mask, :]
#     colors = color_array[points_mask, :]
#     pc_msg = xyz_array_to_point_cloud_msg(points, pc_msg.header.stamp, frame_id='camera_depth_optical_frame', colorimg = colors)
#     cloud_pub_.publish(pc_msg)


if __name__ == '__main__':
    # gridfilter = GridFilter(0.02, (1., 5., -3., 3., 0, 1))
    # points = np.random.rand(1000000, 3)
    # points[:, 0] = points[:, 0] *3
    # points[:, 1] = points[:, 1] *4 -2

    gridfilter = GridFilter(0.05, (-2., 14., -8., 8., -0.5, 2.0))
    points=np.load('testpts.npy')
    gridfilter.grid_filter(points)

    tt = time.time()
    points_ordered = points[points[:,0].argsort()]
    print('Sort time {}'.format(time.time()-tt))
    gridfilter.grid_filter_multicore(points_ordered)
    # rospy.init_node("test_filter", log_level=rospy.INFO)
    # rospy.Subscriber('/camera/depth/color/points', PointCloud2, handle_pc, queue_size=1) # points need to be in NED
    # rospy.spin()
