import cv2
import numpy as np
from math import ceil
import time
from .cscrollgrid import fast_min_max, inflate_map
np.set_printoptions(suppress=True, threshold=10000)

FLOATMAX = 1000000.0

class ScrollGrid(object):
    def __init__(self, 
                 resolution, 
                 crop_range,
                 ):
        '''
        resoluation: grid size in meter
        range: (xmin, xmax, ymin, ymax) in meter, 
               the cropped grid map covers [(xmin, ymin), (xmax, ymax))
        
        '''

        self.resolution = resolution
        self.xmin, self.xmax, self.ymin, self.ymax  = crop_range
        # the (xmin, ymin) is located at the center of a grid
        # so if the (center_pt - min_pt) can be devided by reso, the center_pt will also be located at the center of a grid
        self.xnum = int(ceil((self.xmax - self.xmin )/self.resolution)) 
        self.ynum = int(ceil((self.ymax - self.ymin )/self.resolution))

        self.emem = np.zeros((self.xnum, self.ynum, 4), dtype=np.float32) # heightmap - min, max, mean, std, mask
        self.cmem = np.zeros((self.xnum, self.ynum, 3), dtype=np.uint8) # rgbmap

        print('Map initialized, resolution {}, range {}, shape {}'.format(self.resolution, crop_range, (self.xnum, self.ynum)))

        self.initialize_elevation_map()

    def initialize_elevation_map(self):
        self.emem[:, :, 0].fill(FLOATMAX) # store min-height
        self.emem[:, :, 1].fill(-FLOATMAX) # store max-height
        self.emem[:, :, 2].fill(0) # store mean-height
        self.emem[:, :, 3].fill(0) # store std-height
        self.cmem.fill(0) # rgbmap

    def pc_xy_to_grid_ind(self, x, y):
        xind = np.round((x - self.xmin)/self.resolution)
        yind = np.round((y - self.ymin)/self.resolution)
        return xind, yind

    def pc_coord_to_grid_ind(self, pc_coord):
        return ((pc_coord-np.array([self.xmin, self.ymin]))/self.resolution).astype(np.int32)
        # return np.round((pc_coord-np.array([self.xmin, self.ymin]))/self.resolution).astype(np.int32)

    def pc_to_map(self, points, colors=None):
        starttime = time.time()
        # import ipdb; ipdb.set_trace()
        grid_inds = self.pc_coord_to_grid_ind(points[:, 0:2])
        self.initialize_elevation_map()

        grid_inds = grid_inds.astype(np.uint16)
        zgrid = points[:, 2].astype(np.float32)
        mean_sq = np.zeros((self.xnum, self.ynum)).astype(np.float32)
        points_count = np.zeros((self.xnum, self.ynum)).astype(np.uint16)
        self.emem, self.cmem = fast_min_max(self.emem, self.cmem, grid_inds, zgrid, colors, mean_sq, points_count, self.xnum, self.ynum)
        self.emem[:, :, 3] = np.sqrt(np.abs(self.emem[:, :, 3]))# due to precision error, the std value can be very small negative value
        # for i, ind in enumerate(grid_inds):
        #     if ind[0]>=0 and ind[0]<self.xnum and ind[1]>=0 and ind[1]<self.ynum:
        #         if points[i, 2] < self.emem[ind[0], ind[1], 0]:
        #             self.emem[ind[0], ind[1], 0] = points[i, 2]
        #         if points[i, 2] > self.emem[ind[0], ind[1], 1]:
        #             self.emem[ind[0], ind[1], 1] = points[i, 2]
        #             if colors is not None:
        #                 self.cmem[ind[0], ind[1], :] = colors[i,:]

        print('Localmap convert time: {}'.format(time.time()-starttime))
        # import ipdb;ipdb.set_trace()
        # self.show_heightmap()
        # import ipdb;ipdb.set_trace()

    def inflate_maps_py(self, neighbor_count=3):
        emem_inf = self.emem.copy()
        cmem_inf = self.cmem.copy()

        neighbor_offsets_x = [-1, -1, -1,  0, 0,  1,  1,  1, 0, 0, 2, -2]
        neighbor_offsets_y = [-1, 0,  1,  -1, 1, -1,  0, 1, -2, 2, 0, 0]

        for i in range(self.xnum):
            for j in range(self.ynum):
                if self.emem[i, j, 0] == FLOATMAX: # if current pixel is empty, check its neighbors
                    validcount = 0 
                    color_ave = np.array([0.,0.,0.])
                    height_ave = np.array([0.,0.,0.,0.])
                    for offx, offy in zip(neighbor_offsets_x, neighbor_offsets_y):
                        neighbor_x = offx + i
                        neighbor_y = offy + j
                        if neighbor_x>=0 and neighbor_x<self.xnum and \
                            neighbor_y>=0 and neighbor_y<self.ynum: #  neighbor position is valid
                            if self.emem[neighbor_x, neighbor_y, 0] < FLOATMAX-1: # neighbor is not empty
                                validcount += 1
                                color_ave += self.cmem[neighbor_x, neighbor_y]
                                height_ave += self.emem[neighbor_x, neighbor_y]
                    if validcount >= neighbor_count: # we have enough valid neighbors
                        emem_inf[i, j] = height_ave/validcount
                        cmem_inf[i, j] = color_ave/validcount
        self.emem = emem_inf
        self.cmem = cmem_inf


# inflate_map(np.ndarray[np.float32_t, ndim=3] emem,
#             np.ndarray[np.uint8_t, ndim=3] cmem,
#             np.ndarray[np.float32_t, ndim=3] emem_inf,
#             np.ndarray[np.uint8_t, ndim=3] cmem_inf,
#             np.ndarray[np.uint32_t, ndim=1] neighbor_offsets_x, 
#             np.ndarray[np.uint32_t, ndim=1] neighbor_offsets_y, 
#             int32_t xnum, int32_t ynum)

    def inflate_maps(self):
        emem_inf = self.emem.copy()
        cmem_inf = self.cmem.copy()

        neighbor_offsets_x = np.array([-1, -1, -1,  0, 0,  1,  1,  1, 0, 0, 2, -2]).astype(np.int32)
        neighbor_offsets_y = np.array([-1, 0,  1,  -1, 1, -1,  0, 1, -2, 2, 0, 0]).astype(np.int32)

        # grid_inds = grid_inds.astype(np.uint16)
        # zgrid = points[:, 2].astype(np.float32)
        emem_inf, cmem_inf = inflate_map(self.emem, self.cmem, emem_inf,cmem_inf, \
                                        neighbor_offsets_x, neighbor_offsets_y, \
                                        self.xnum, self.ynum)
        self.emem = emem_inf
        self.cmem = cmem_inf

    def get_vis_heightmap(self, scale=0.5, hmin=-1, hmax=4):
        mask = self.emem[:,:,0]==FLOATMAX
        disp1 = np.clip((self.emem[:, :, 0] - hmin)*100, 0, 255).astype(np.uint8)
        disp2 = np.clip((self.emem[:, :, 1] - hmin)*100, 0, 255).astype(np.uint8)
        disp3 = np.clip((self.emem[:, :, 2] - hmin)*100, 0, 255).astype(np.uint8)
        disp4 = np.clip(self.emem[:, :, 3]*1000, 0, 255).astype(np.uint8)
        disp1[mask] = 0
        disp2[mask] = 0
        disp3[mask] = 0
        disp4[mask] = 0
        disp_1 = np.concatenate((cv2.flip(disp1, -1), cv2.flip(disp2, -1)) , axis=1)
        disp_2 = np.concatenate((cv2.flip(disp3, -1), cv2.flip(disp4, -1)) , axis=1)
        disp = np.concatenate((disp_1, disp_2) , axis=0)
        disp = cv2.resize(disp, (0, 0), fx=scale, fy=scale)
        disp_color = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        return disp_color

    def show_heightmap(self, hmin=-1, hmax=4):
        disp_color = self.get_vis_heightmap(hmin=-1, hmax=4)

        cv2.imshow('height',disp_color)
        cv2.waitKey(1)

    def get_vis_rgbmap(self, ):
        return cv2.flip(self.cmem, -1)

    def show_colormap(self):
        disp = self.get_vis_rgbmap()
        # disp = cv2.resize(disp, (0, 0), fx=2., fy=2.)

        cv2.imshow('color',disp)
        cv2.waitKey(1)

    def get_height(self, x, y):
        pass


    def get_heightmap(self):
        return self.emem

    def get_rgbmap(self):
        return self.cmem

if __name__ == '__main__':
    localmap = ScrollGrid(0.01, (1., 5., -2., 2.))
    points = np.random.rand(1000000, 3)
    points[:, 0] = points[:, 0] *3
    points[:, 1] = points[:, 1] *4 -2
    colors = np.random.rand(1000000, 3)
    colors = (colors * 256).astype(np.uint8)
    localmap.pc_to_map(points, colors)
    localmap.show_heightmap()
    localmap.show_colormap()
