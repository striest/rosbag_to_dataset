from xmlrpc.client import boolean
import numpy as np
cimport numpy as np
np.import_array()

# from libc.stdint cimport int8_t, int16_t, int32_t, int64_t
from libc.stdint cimport uint8_t, int32_t #, uint16_t, uint32_t, uint64_t

# cdef extern void get_depth_parallel(int32_t* X, int32_t* Y, int32_t* Z, uint32_t height, uint32_t width, int32_t* gids, uint32_t* emap, uint32_t g0, uint32_t g1, uint8_t k_threads)

# cdef extern void c_fast_min_max(
#     int16_t* mem, float32_t m1, uint32_t m2,
#     uint16_t* mem_indices, uint32_t mi1,
#     int32_t* z_gid, float32_t nPoints)

cimport cython

@cython.wraparound(False)
@cython.boundscheck(False)
def fast_min_max(np.ndarray[np.float32_t, ndim=3] emem,
                 np.ndarray[np.uint8_t, ndim=3] cmem,
                 np.ndarray[np.uint16_t, ndim=2] mem_indices,
                 np.ndarray[np.float32_t, ndim=1] z_gid, 
                 np.ndarray[np.uint8_t, ndim=2] c_gid,
                 np.ndarray[np.float32_t, ndim=2] mean_sq,
                 np.ndarray[np.uint16_t, ndim=2] points_count,
                 int32_t xnum, int32_t ynum):

    cdef unsigned int nPoints = mem_indices.shape[0]

    '''
    c_fast_min_max(
        <int16_t*> mem.data, mem.shape[1], mem.shape[2],
        <uint16_t*> mem_indices.data, mem_indices.shape[1],
        <int32_t*> z_gid.data, nPoints
    )
    return
    '''

    cdef unsigned int i
    cdef unsigned int x, y
    cdef float z
    cdef float cnt

    # cdef np.ndarray points_count = np.zeros((xnum, ynum), dtype=np.int)
    # cdef np.ndarray mean_sq = np.zeros((xnum, ynum), dtype=np.float32)

    for i in range(nPoints):
        x = <unsigned int>mem_indices[i, 0]
        y = <unsigned int>mem_indices[i, 1]
        z = <float>z_gid[i]

        if x >= 0 and x<xnum and y>=0 and y<ynum:
            # calculate mean value
            points_count[x, y] = points_count[x, y] + 1
            cnt = float(points_count[x, y]) # number of points in this grid
            emem[x, y, 2] = (cnt-1)/(cnt) * emem[x, y, 2] + z/(cnt) 
            # calculate std value
            mean_sq[x, y] = (cnt-1)/(cnt) * mean_sq[x, y] + z*z/(cnt) 
            emem[x, y, 3] = mean_sq[x, y] - emem[x, y, 2] * emem[x, y, 2]
            
            if z < emem[x, y, 0]:
                emem[x, y, 0] = z

            if z > emem[x, y, 1]:
                emem[x, y, 1] = z
                # cmem[x, y, :] = c_gid[i]
                cmem[x, y, 0] = <uint8_t>((cnt-1)/(cnt) * cmem[x, y, 0] + c_gid[i,0]/(cnt)) #<uint8_t>c_gid[i,0]
                cmem[x, y, 1] = <uint8_t>((cnt-1)/(cnt) * cmem[x, y, 1] + c_gid[i,1]/(cnt)) #<uint8_t>c_gid[i,1]
                cmem[x, y, 2] = <uint8_t>((cnt-1)/(cnt) * cmem[x, y, 2] + c_gid[i,2]/(cnt)) #<uint8_t>c_gid[i,2]

    return emem, cmem

# for k, aaa in enumerate(aa):
#     mean = float(k)/(k+1)*mean + aaa/(k+1)
#     meansq = float(k)/(k+1)*meansq + aaa*aaa/(k+1)
#     var = meansq - mean*mean
#     std = sqrt(var)


@cython.wraparound(False)
@cython.boundscheck(False)
def voxel_filter(np.ndarray[np.uint16_t, ndim=2] pt_indices,
                 np.ndarray[np.float32_t, ndim=1] pt_dist, 
                 np.ndarray[np.float32_t, ndim=3] mindist, 
                 np.ndarray[np.int32_t, ndim=3] mininds, 
                 np.ndarray[np.uint8_t, ndim=1] resmask, 
                 int32_t xnum, int32_t ynum, int32_t znum):

    cdef unsigned int nPoints = pt_indices.shape[0]

    # cdef np.ndarray mindist = np.ones((xnum, ynum, znum), dtype=np.float32) * 10000
    # cdef np.ndarray mininds = np.ones((xnum, ynum, znum), dtype=np.int32)
    # cdef np.ndarray resmask = np.zeros((nPoints), dtype=np.bool)

    cdef unsigned int i
    cdef unsigned int x, y, z

    for i in range(nPoints):
        x = <unsigned int>pt_indices[i, 0]
        y = <unsigned int>pt_indices[i, 1]
        z = <unsigned int>pt_indices[i, 2]

        if x >= 0 and x<xnum and y>=0 and y<ynum and z>=0 and z<znum:

            dist = pt_dist[i]
            if dist < mindist[x, y, z]:
                if mindist[x, y, z] < 9999:
                    resmask[mininds[x, y, z]] = 0
                mindist[x, y, z] = dist
                mininds[x, y, z] = i
                resmask[i] = 1

    return resmask


@cython.wraparound(False)
@cython.boundscheck(False)
def inflate_map(np.ndarray[np.float32_t, ndim=3] emem,
            np.ndarray[np.uint8_t, ndim=3] cmem,
            np.ndarray[np.float32_t, ndim=3] emem_inf,
            np.ndarray[np.uint8_t, ndim=3] cmem_inf,
            np.ndarray[np.int32_t, ndim=1] neighbor_offsets_x, 
            np.ndarray[np.int32_t, ndim=1] neighbor_offsets_y, 
            int32_t xnum, int32_t ynum):

    cdef unsigned int offsetnum = len(neighbor_offsets_x)
    cdef unsigned int i, j
    cdef unsigned int offx, offy, neighbor_x, neighbor_y
    cdef unsigned int validcount
    cdef float color_ave_r, color_ave_g, color_ave_b
    cdef float height_ave_1, height_ave_2, height_ave_3, height_ave_4

    for i in range(xnum):
        for j in range(ynum):
            if emem[i, j, 0] > 10000: # if current pixel is empty, check its neighbors
                validcount = 0 
                color_ave_r = 0.
                color_ave_g = 0.
                color_ave_b = 0.
                height_ave_1 = 0.
                height_ave_2 = 0.
                height_ave_3 = 0.
                height_ave_4 = 0.

                for k in range(offsetnum):
                    offx = neighbor_offsets_x[k]
                    offy = neighbor_offsets_y[k]
                    neighbor_x = offx + i
                    neighbor_y = offy + j
                    if neighbor_x>=0 and neighbor_x<xnum and \
                        neighbor_y>=0 and neighbor_y<ynum: #  neighbor position is valid
                        if emem[neighbor_x, neighbor_y, 0] < 10000: # neighbor is not empty
                            validcount += 1
                            color_ave_r += cmem[neighbor_x, neighbor_y, 0]
                            color_ave_g += cmem[neighbor_x, neighbor_y, 1]
                            color_ave_b += cmem[neighbor_x, neighbor_y, 2]
                            height_ave_1 += emem[neighbor_x, neighbor_y, 0]
                            height_ave_2 += emem[neighbor_x, neighbor_y, 1]
                            height_ave_3 += emem[neighbor_x, neighbor_y, 2]
                            height_ave_4 += emem[neighbor_x, neighbor_y, 3]
                if validcount >= 3: # we have enough valid neighbors
                    emem_inf[i, j, 0] = height_ave_1/validcount
                    emem_inf[i, j, 1] = height_ave_2/validcount
                    emem_inf[i, j, 2] = height_ave_3/validcount
                    emem_inf[i, j, 3] = height_ave_4/validcount
                    cmem_inf[i, j, 0] = <uint8_t>(color_ave_r/validcount)
                    cmem_inf[i, j, 1] = <uint8_t>(color_ave_g/validcount)
                    cmem_inf[i, j, 2] = <uint8_t>(color_ave_b/validcount)

    return emem_inf, cmem_inf