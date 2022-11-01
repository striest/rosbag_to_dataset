from matplotlib import testing
import torch


# from __future__ import division
# import torch
import math
import random
# from PIL import Image, ImageOps
import numpy as np
import numbers
import cv2


from scipy.spatial.transform import Rotation as R

def tensor2img(tensImg,mean,std):
    """
    convert a tensor a numpy array, for visualization
    """
    # undo normalize
    img = tensImg.clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    # undo transpose
    img = (img.numpy().transpose(1,2,0)*float(255)).astype(np.uint8)
    return img


def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def SE2euler(se_data):
    '''
    4 x 4 -> 6
    '''
    res = np.zeros(6)
    SO = se_data[:3,:3]
    euler = R.as_euler(SO, 'XYZ', degrees=False)
    res[:3] = se_data[:3,3] 
    res[3:] = euler
    return res


def se2SE(se_data):
    '''
    6 -> 4 x 4
    '''
    result_mat = np.matrix(np.eye(4))
    result_mat[0:3,0:3] = so2SO(se_data[3:6])
    result_mat[0:3,3]   = np.matrix(se_data[0:3]).T
    return result_mat

def SO2quat(SO_data):
    rr = R.from_matrix(SO_data)
    return rr.as_quat()

def se2quat(se_data):
    '''
    6 -> 7
    '''
    SE_mat = se2SE(se_data)
    pos_quat = np.zeros(7)
    pos_quat[3:] = SO2quat(SE_mat[0:3,0:3])
    pos_quat[:3] = SE_mat[0:3,3].T
    return pos_quat


def calculate_angle_distance_from_du_dv(du, dv, flagDegree=False):
    a = np.arctan2( dv, du )

    angleShift = np.pi

    if ( True == flagDegree ):
        a = a / np.pi * 180
        angleShift = 180
        # print("Convert angle from radian to degree as demanded by the input file.")

    d = np.sqrt( du * du + dv * dv )

    return a, d, angleShift

def visflow(flownp, maxF=500.0, n=8, mask=None, hueMax=179, angShift=0.0): 
    """
    Show a optical flow field as the KITTI dataset does.
    Some parts of this function is the transform of the original MATLAB code flow_to_color.m.
    """

    ang, mag, _ = calculate_angle_distance_from_du_dv( flownp[:, :, 0], flownp[:, :, 1], flagDegree=False )

    # Use Hue, Saturation, Value colour model 
    hsv = np.zeros( ( ang.shape[0], ang.shape[1], 3 ) , dtype=np.float32)

    am = ang < 0
    ang[am] = ang[am] + np.pi * 2

    hsv[ :, :, 0 ] = np.remainder( ( ang + angShift ) / (2*np.pi), 1 )
    hsv[ :, :, 1 ] = mag / maxF * n
    hsv[ :, :, 2 ] = (n - hsv[:, :, 1])/n

    hsv[:, :, 0] = np.clip( hsv[:, :, 0], 0, 1 ) * hueMax
    hsv[:, :, 1:3] = np.clip( hsv[:, :, 1:3], 0, 1 ) * 255
    hsv = hsv.astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if ( mask is not None ):
        mask = mask != 255
        bgr[mask] = np.array([0, 0 ,0], dtype=np.uint8)

    return bgr

def visdepth(depth, maxthresh = 50):
    depthvis = np.clip(depth,0,maxthresh)
    depthvis = depthvis/maxthresh*255
    depthvis = depthvis.astype(np.uint8)
    depthvis = np.tile(depthvis.reshape(depthvis.shape+(1,)), (1,1,3))

    return depthvis

def disp2vis(disp, scale=10,):
    '''
    disp: h x w float32 numpy array
    return h x w x 3 uint8 numpy array
    '''
    disp = np.clip(disp * scale, 0, 255).astype(np.uint8)
    disp = np.tile(disp[:,:,np.newaxis], (1, 1, 3))
    # disp = cv2.resize(disp,(640,480))

    return disp
