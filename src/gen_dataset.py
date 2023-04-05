"""
https://github.com/striest/rosbag_to_dataset/blob/1b2246b28d6ca63d6b3fbcd72f59c74ed516b839/src/rosbag_to_dataset/post_processing/mapping/LocalMappingRegister.py
"""
import sys
import os, os.path as osp
import shutil

import argparse

from tqdm import tqdm
from pprint import pprint
from easydict import EasyDict as edict

from typing import Sequence, Optional, Any

import math
import numpy as np
import cv2
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Build maps registering all the scans.');
parser.add_argument('--data_dir', type=str, 
                    help='data directory containing the trajectories.');
parser.add_argument('--pts_subdir', type=str, 
                    help='Sub-directory containing point clouds.');
parser.add_argument('--motion_subpath', type=str, 
                    help='Sub-path containing motion information.');
parser.add_argument('--pose_subpath', type=str, 
                    help='Sub-path containing pose information.');
parser.add_argument('--max_n_frames', type=int, default=-1,
                    help='Maximum number of frames to process in each trajectory to avoid OOM error.');
parser.add_argument('--out_dir', type=str, 
                    help='directory to save the maps.');                                                            

args = parser.parse_args();
pprint(args);

class TemporalDataGenerator(object) :
    def __init__(self, 
        data_dir: str,
        pts_subdir: str,
        motion_subpath: str,        
        pose_subpath: str,
        out_dir: str,
        pts_ext: str = '.npy',
        pose_ext: str = '.npy',
        downsample_bev_res: float = 0.02, # 2cm
        x_minmax: Sequence[float] = [0., 12.],
        y_minmax: Sequence[float] = [-6., 6.],
        res: float = 0.02, # 2cm,
        image_ext: str = '.png',
        pc_subdir: str = 'pc',
        image_subdir: str = 'images',
        max_n_frames: int = -1,
    ) -> None :

        super().__init__();

        assert osp.isdir(data_dir), f"Data directory (trajectories) not found {data_dir}";

        self.dtype = 'float32';
        self.style_l = ['avg', 'idw-avg', 'mindist'];

        self.data_dir = data_dir;
        self.pts_subdir = pts_subdir;
        self.motion_subpath = motion_subpath;
        self.pose_subpath = pose_subpath;
        self.out_dir = out_dir;
        self.image_ext = image_ext;
        self.pc_subdir = pc_subdir;
        self.image_subdir = image_subdir;

        # if osp.isdir(self.out_dir) :
        #     shutil.rmtree(self.out_dir);
        # # os.makedirs(self.out_image_dir);
        # # os.makedirs(self.out_label_dir);

        self.pts_ext = pts_ext;
        self.pose_ext = pose_ext;

        self.downsample_bev_res = downsample_bev_res;
        self.xmin, self.xmax = x_minmax;
        self.ymin, self.ymax = y_minmax;
        self.int_max = np.iinfo(np.int64).max;
        self.res = res;

        self.max_n_frames = max_n_frames;
        
        self.dense_traj_map_file_name = 'dense_map_wrt_first_frame.bin';
        # gravity aligned
        self.dense_traj_map_g_aligned_file_name = 'dense_map_wrt_first_frame_g_aligned.bin';

        self.pts_loader = self.get_pts_loader();
        self.pose_loader = self.get_pose_loader();

        # for gravity alignment
        # source: https://github.com/striest/rosbag_to_dataset/blob/1b2246b28d6ca63d6b3fbcd72f59c74ed516b839/src/rosbag_to_dataset/post_processing/mapping/LocalMappingRegister.py#L45 
        self.novatel_to_body = np.array([[0,-1 ,0],
                                      [1, 0, 0],
                                      [0, 0, 1]]);
        self.body_to_novatel = np.array([[0, 1 ,0],
                                      [-1, 0, 0],
                                      [ 0, 0, 1]]);

        self.R_ground = np.array([[ 0.9692535 ,  0.        , -0.24606434],
                                [ 0.00610748, -0.99969192,  0.02405752],
                                [-0.24598853, -0.02482067, -0.96895489]] );
        self.T_ground = np.array([[-5.55111512e-17, -6.93889390e-18, 1.77348523e+00]]);

        self.novatel_to_body = self.novatel_to_body.astype(self.dtype);
        self.body_to_novatel = self.body_to_novatel.astype(self.dtype);
        self.R_ground = self.R_ground.astype(self.dtype);
        self.T_ground = self.T_ground.astype(self.dtype);
    
        self.count_oob = 0;


    def get_pts_loader(self) :
        if self.pts_ext == '.npy' :
            return np.load;
        else :
            raise NotImplementedError;

    def get_pose_loader(self) :
        if self.pose_ext == '.npy' :
            return np.load;
        else :
            raise NotImplementedError;


    def is_valid_traj_dir_v1(self, dir_: str) -> bool :
        """
        Check if the trajectory directory is a valid one.
        Remove all the directories w/ interventions, bigpuddle, trial, reverse, etc.
        """
        vals = dir_.split('_');
        for x in vals :
            if not x.isnumeric() :
                return False;
        
        return True;

    def is_valid_traj_dir(self, dir_: str) -> bool :
        """
        Check if the trajectory directory is a valid one.
        """
        vals = dir_.split('_');
        ymd = vals[0];
        if ymd.isnumeric() and len(ymd)==8 :
            return True;
        
        return False;


    def get_image_name_from_pc_name(self, fpath: str) -> str :
        return osp.splitext(fpath)[0] + self.image_ext;


    def list_traj_dirs(self) :
        traj_list = [];
        for traj_subdir in sorted(os.listdir(self.data_dir)) :
            if not self.is_valid_traj_dir(traj_subdir) :
                # print(traj_subdir);
                continue;
            
            # print(traj_subdir);
            traj_list.append(traj_subdir);

        traj_list.sort();
        return traj_list;


    def get_pts_file_list(self, 
        pts_dir: str,
    ) -> Sequence[str] :

        assert osp.isdir(pts_dir), f"Points directory not found = {pts_dir}";
        pts_file_list = [];
        for fname in sorted(os.listdir(pts_dir)) :
            if not fname.endswith(self.pts_ext) :
                continue;
            pts_file_list.append(osp.join(pts_dir, fname));

        return pts_file_list;


    def get_motion_filepath(self, 
        traj_dir: str,
    ) -> str :

        motion_filepath = osp.join(traj_dir, self.motion_subpath);
        assert not osp.isdir(motion_filepath), \
            f"pose file not found = {motion_filepath}";
        return motion_filepath;


    def get_pose_filepath(self, 
        traj_dir: str,
    ) -> str :

        pose_filepath = osp.join(traj_dir, self.pose_subpath);
        assert not osp.isdir(pose_filepath), \
            f"pose file not found = {pose_filepath}";
        return pose_filepath;


    def get_traj_file_list_single(self, traj_dir) :
        pts_file_list = self.get_pts_file_list(osp.join(traj_dir, self.pts_subdir));
        motion_filepath = self.get_motion_filepath(traj_dir);
        pose_filepath = self.get_pose_filepath(traj_dir);
        return pts_file_list, motion_filepath, pose_filepath;

    def get_rel_extrinsics(self, pose_filepath) :
        """Get extrinsics (R, T) w.r.t. the initial time stamp."""
        poses = self.pose_loader(pose_filepath);
        pos = poses[:, :3];
        quat = poses[:, 3:7];

        rot = Rotation.from_quat(quat);
        ypr = rot.as_euler('zyx', degrees=False); # yaw, pitch, roll
        # print(ypr.shape, quat.shape); sys.exit();

        rel_pos = pos - pos[:1, :];
        rel_ypr = ypr - ypr[:1, :];

        rot = Rotation.from_euler('zyx', rel_ypr, degrees=False);
        R_0x = rot.as_matrix();
        T_x = rel_pos;
        # print(R.shape, T.shape, quat.shape, ); sys.exit();
        return R_0x, T_x;


    def transform_x_to_0(self, pts, R_0x, T_x) :
        R_0x = R_0x.astype(pts.dtype);
        T_x = T_x.astype(pts.dtype);
        # x_0 = R_10 @ x_1 + T_0->1
        # x_0.T = x_1.T @ R_10.T + T_0->1.T
        pts = pts @ R_0x.T + T_x[None, :];
        return pts;


    def get_out_filepath_from_traj_dir(self, traj_dir) :
        return osp.join(self.out_dir, osp.basename(traj_dir) + '.bin');

    def save_single(self, map_pts, fpath) :
        map_pts.tofile(fpath);

    def load_single(self, fpath) :
        assert osp.isfile(fpath), f"File not found = {fpath}";
        map_pts = np.fromfile(fpath, dtype=self.dtype).reshape(-1, 7);
        return map_pts;        

    def save_image(self, fpath: str, im: np.ndarray, ) :
        assert osp.isdir(osp.dirname(fpath));
        assert fpath.endswith(self.image_ext);
        cv2.imwrite(fpath, cv2.cvtColor(im, cv2.COLOR_RGB2BGR));
        # imsave(fpath, im);


    def gravity_align2(self, 
        pts: np.ndarray, 
        odom: np.ndarray,
    ) -> np.ndarray :
        """
        Source: https://github.com/striest/rosbag_to_dataset/blob/1b2246b28d6ca63d6b3fbcd72f59c74ed516b839/src/rosbag_to_dataset/post_processing/mapping/LocalMappingRegister.py#L118 
        """

        orientation = odom[3:7]
        rotations = Rotation.from_quat(orientation)
        rot_mat = rotations.as_matrix()

        yy = np.array([rot_mat[0,1], rot_mat[1,1], 0.0]) # project the y axis to the ground
        yy = yy/np.linalg.norm(yy)
        zz = np.array([0.,0, 1])
        xx = np.cross(yy, zz)
        rot_mat2 = np.stack((xx,yy,zz),axis=1)
        rot_mat3 = rot_mat.transpose() @ rot_mat2
        rot_mat3 = rot_mat3.astype(self.dtype); 

        R = self.R_ground @ self.body_to_novatel @ rot_mat3 @ self.novatel_to_body
        T = self.T_ground @ self.body_to_novatel @ rot_mat3 @ self.novatel_to_body
        
        pts_aligned = pts @ R + T
        return pts_aligned


    def transform_prev_to_this(self, pts, motion) :
        # T = motion[:3].reshape(3, 1).astype(self.dtype);
        T = motion[:3].reshape(1, 3).astype(self.dtype);
        quat = motion[3:];
        R = Rotation.from_quat(quat).as_matrix().astype(self.dtype);

        # pts = R.T @ pts.T - T;        
        # pts = pts.T;

        # R = R_{(t-1)->(t)}
        pts = pts @ R - T;
        return pts;

    def transform_post_to_this(self, pts, motion) :
        # T = motion[:3].reshape(3, 1).astype(self.dtype);
        T = motion[:3].reshape(1, 3).astype(self.dtype);
        quat = motion[3:];
        R = Rotation.from_quat(quat).as_matrix().astype(self.dtype);

        # pts = R.T @ pts.T - T;        
        # pts = pts.T;

        pts = (pts + T) @ R.T;
        return pts;

    def transform_prev_to_this_mmat_homo(self, RT_prev, motion) :
        """Transforms previous model matrix to current frame in homo coordinate."""
        T = motion[:3].reshape(1, 3).astype(self.dtype);
        quat = motion[3:];
        R = Rotation.from_quat(quat).as_matrix().astype(self.dtype);

        RT = np.zeros((4, 4), dtype=self.dtype);
        RT[:3, :3] = R;
        RT[3:, :3] = -T;
        RT[3, 3] = 1;

        RT = RT_prev @ RT;
        return RT;                


    def filter_xy_minmax(self, 
        pts: np.ndarray
    ) -> np.ndarray :

        x, y = pts[:, 0], pts[:, 1];
        mask = (x > self.xmin) & (x < self.xmax) & (y > self.ymin) & (y < self.ymax);
        pts = pts[mask];
        return pts;


    def gen_data_single_pair(self, 
        pts_file_list: Sequence[str], 
        motions: np.ndarray, 
        odoms: np.ndarray,
        start_frame_frac: float, # X% from the beginning
        end_frame_frac: float, # X% before the end
        stride: int, # #frames between consecutive data capture 
        downsample_stride: int, # #frames between consecutive downsampling
        full_map_dir: str,
        pc_out_dir: str,   
        image_out_dir: str,
    ) -> None :

        assert 0 <= start_frame_frac < 1, \
            f"Start frame fraction must be in (0, 1), got {start_frame_frac}";
        assert 0 < end_frame_frac <= 1, \
            f"End frame fraction must be in (0, 1), got {end_frame_frac}";
        assert start_frame_frac < end_frame_frac, \
            f"Start frame fraction {start_frame_frac} must be less than end frame fraction {end_frame_frac}";

        if osp.isdir(pc_out_dir) :
            shutil.rmtree(pc_out_dir);
        os.makedirs(pc_out_dir);

        if osp.isdir(image_out_dir) :
            shutil.rmtree(image_out_dir);
        os.makedirs(image_out_dir);        

        # load full map
        # map_fpath = osp.join(out_dir, self.sparse_traj_map_file_name); # sparse map path
        map_fpath = osp.join(full_map_dir, self.dense_traj_map_file_name); # dense map path
        map_pts = self.load_single( map_fpath );            
        xyz_map, rgb_map, depth_map = map_pts[:, :3], map_pts[:, 3:6], map_pts[:, 6:];
        # # homogeneous
        # xyz_map_h = np.concatenate(
        #             (
        #                 xyz_map, 
        #                 np.ones((xyz_map.shape[0], 1), dtype=xyz_map.dtype)
        #             ), axis=1
        # );
        xyz_map_h = xyz_map;
        xyz_map = None;
        map_pts = None;        

        n_frames = len(pts_file_list);
        start_frame = int(start_frame_frac * n_frames);
        end_frame = int(end_frame_frac * n_frames);  
        snap_frame_l = range(start_frame, end_frame+1, stride);

        xyz_all, rgb_all, depth_all = None, None, None;
        RT = None;
        
        snap_list_i = 0;
        snap_frame_i = snap_frame_l[snap_list_i];
        for i, pts_path in enumerate(tqdm(pts_file_list)) :
            # if i%40 > 0 :
            #     continue;
            
            xyzrgb = self.pts_loader(pts_path);
            xyz, rgb = xyzrgb[:, :3], xyzrgb[:, 3:];
            depth = np.linalg.norm(xyz, axis=1, keepdims=True);

            if xyz_all is None :
                xyz_all = xyz;
                rgb_all = rgb;
                depth_all = depth;
                RT = np.eye(4, dtype=self.dtype);
            else :
            
                # register all the previous frames to the current frame
                xyz_all = self.transform_prev_to_this(xyz_all, motions[i-1]);
                # RT = self.transform_prev_to_this_mmat_homo(RT, motions[i-1]);
                xyz_map_h = self.transform_prev_to_this(xyz_map_h, motions[i-1]);

                xyz_all = np.concatenate((xyz_all, xyz), axis=0);
                rgb_all = np.concatenate((rgb_all, rgb), axis=0);
                depth_all = np.concatenate((depth_all, depth), axis=0);
                # xyz_all, rgb_all, depth_all = xyz, rgb, depth;
                assert xyz_all.dtype == xyz.dtype;

            # if (i % downsample_stride == 0) and (i > 0) :
            #     map_pts = np.concatenate((xyz_all, rgb_all, depth_all), axis=1);
            #     map_pts, _ = self.downscale_map_w_mindist(map_pts);
            #     xyz_all, rgb_all, depth_all = map_pts[:, :3], map_pts[:, 3:6], map_pts[:, 6:];

            if i != snap_frame_i :
                continue;

            # # also align for current time stamp (input image)
            xyz_snap = self.gravity_align2(xyz_all, odoms[snap_frame_i]);
            # # xyz_map_snap = (xyz_map_h @ RT)[:, :3]; # remove homo coord
            # # xyz_map_snap = self.gravity_align2(xyz_map_snap, odoms[snap_frame_i]); # full map
            xyz_map_snap = self.gravity_align2(xyz_map_h, odoms[snap_frame_i]); # full map
            # assert xyz_snap.dtype == xyz.dtype;
            assert xyz_map_snap.dtype == xyz.dtype;        

            # save input points
            map_pts = np.concatenate((xyz_snap, rgb_all, depth_all), axis=1);
            # filter w/ xy minmax range and save for memory efficiency
            map_pts = self.filter_xy_minmax(map_pts);
            map_pts, keep_ids = self.downscale_map_w_mindist(map_pts);
            input_fname = 'input_' + str(snap_frame_i).zfill(6) + '.bin';
            self.save_single(map_pts, osp.join(pc_out_dir, input_fname) ); 
            # get rgb image from pc
            input_img_fname = self.get_image_name_from_pc_name(input_fname);
            self.gen_map_2d_single(
                osp.join(pc_out_dir, input_fname), 
                osp.join(image_out_dir, input_img_fname),
                'mindist',
                xmin=self.xmin,
                xmax=self.xmax,
                ymin=self.ymin,
                ymax=self.ymax,
                res=self.res,
            );

            # save full map (label) points
            map_pts = np.concatenate((xyz_map_snap, rgb_map, depth_map), axis=1);
            # filter w/ xy minmax range and save for memory efficiency
            map_pts = self.filter_xy_minmax(map_pts);
            label_fname = 'label_' + str(snap_frame_i).zfill(6) + '.bin';
            self.save_single(map_pts, osp.join(pc_out_dir, label_fname) );             
            # get rgb image from pc
            label_img_fname = self.get_image_name_from_pc_name(label_fname);
            self.gen_map_2d_single(
                osp.join(pc_out_dir, label_fname), 
                osp.join(image_out_dir, label_img_fname),
                'mindist',
                xmin=self.xmin,
                xmax=self.xmax,
                ymin=self.ymin,
                ymax=self.ymax,
                res=self.res,                
            );

            # xyz_all = xyz_all[keep_ids, :];
            # rgb_all = rgb_all[keep_ids, :];
            # depth_all = depth_all[keep_ids, :];

            snap_list_i += 1;
            if snap_list_i == len(snap_frame_l) :
                break;   
            snap_frame_i = snap_frame_l[snap_list_i];


    def gen_data_single_traj(self, 
        traj_i : int,
        n_traj: int,
        traj_dir: str, 
    ) :

        traj_name = osp.basename(traj_dir);
        tmp_out_dir = osp.join(self.out_dir, traj_name);
        os.makedirs(tmp_out_dir, exist_ok=True);
        print(f"[{traj_i} / {n_traj}] Processing = {traj_name} --> {tmp_out_dir}");

        pts_file_list, motion_filepath, pose_filepath = \
                self.get_traj_file_list_single(traj_dir);
        motions = self.pose_loader(motion_filepath);                
        odoms = self.pose_loader(pose_filepath);                

        if (self.max_n_frames>0) and len(pts_file_list) > self.max_n_frames :
            print(f"Total #frames ({len(pts_file_list)}) > max #frames allowed ({self.max_n_frames}), so not processing");
            return;

            print(f"Total #frames ({len(pts_file_list)}) > max #frames allowed ({self.max_n_frames}), so clamping ...");

            pts_file_list = pts_file_list[:self.max_n_frames];
            motions = motions[:self.max_n_frames-1]
            odoms = odoms[:self.max_n_frames]
            sys.exit();              

        self.gen_data_single_pair(
            pts_file_list, 
            motions, 
            odoms,
            start_frame_frac=0.0,
            end_frame_frac=1.0, 
            stride=1, 
            downsample_stride=30, # unused for now
            full_map_dir=tmp_out_dir,
            pc_out_dir=osp.join(tmp_out_dir, self.pc_subdir),
            image_out_dir=osp.join(tmp_out_dir, self.image_subdir),
        );  

        print(f"#OOB points in this trajectory = {self.count_oob}");
        self.count_oob = 0;


            
    def load_pts(self, fpath: str, dim: int = 7) :
        assert osp.isfile(fpath);
        # assert fpath.endswith(self.pts_ext);
        pts = np.fromfile(fpath, dtype=self.dtype).reshape(-1, 7);
        return pts;        

    def downscale_map_w_mindist(self, 
        pts: np.ndarray,
        verbose: bool = False,
    ) -> np.ndarray :

        res = self.downsample_bev_res;

        if verbose :
            print(f"Downscaling map with res = {res}");
        n_pts_org = pts.shape[0];

        xyz, rgb, depth = pts[:, :3], pts[:, 3:6], pts[:, 6];
        x, y = pts[:, 0], pts[:, 1];
        xmin, xmax = x.min(), x.max();
        ymin, ymax = y.min(), y.max();

        w = int(math.ceil((xmax - xmin) / res));
        h = int(math.ceil((ymax - ymin) / res));

        x = x - xmin;
        y = y - ymin;
        x = np.floor(x / res).astype(np.int32);
        y = np.floor(y / res).astype(np.int32);

        # assign points to cells
        lin_id = w * y + x;
        cell_ids = {};
        max_lin_id = lin_id.max();
        if verbose :
            print(f"Linear index maximum = {max_lin_id}");
        assert max_lin_id < self.int_max;

        for i in range(x.size) :
            lid = lin_id[i];
            if lid not in cell_ids :
                cell_ids[lid] = [i];
                continue;
            
            cell_ids[lid].append(i);


        # mindist strategy        
        # update cell id assignments 
        # by selecting the nearest point per cell
        for ci in cell_ids :
            pids = cell_ids[ci];
            i_closest = pids[np.argmin(depth[pids])];
            cell_ids[ci] = i_closest;

        keep_ids = list(cell_ids.values());
        pts = pts[keep_ids, :];
        n_pts_filtered = pts.shape[0];
    
        if verbose :
            perc_pts_removed = (1 - n_pts_filtered / n_pts_org) * 100;
            print(f"Number of raw points = {n_pts_org}");        
            print(f"Number of filtered points = {n_pts_filtered}");
            print(f"Percentage of points removed = {perc_pts_removed:.0f} %");
        return pts, keep_ids;


    def gray_to_rgb(self,
        im: np.ndarray, 
        min_: float = None, 
        max_: float = None, 
        dtype: str = 'uint8', 
        cmap: Any = plt.cm.jet,
    ) -> np.ndarray :

        if min_ is None:
            min_ = np.min(im);
        if max_ is None:
            max_ = np.max(im);
        im = (im - min_) / (max_ - min_);
        im = (255 * cmap(im)[:, :, :3]).astype(dtype);  # H, W, C
        return im;


    def gen_map_2d_single(self, 
        in_fpath: str, 
        out_fpath: str,
        style: str,
        xmin: Optional[float],
        xmax: Optional[float],
        ymin: Optional[float],
        ymax: Optional[float],
        res: Optional[float],
        concat_axis: str='horizontal',
    ) :

        assert style in self.style_l;

        pts = self.load_pts(in_fpath);
        # print(f"Number of raw points = {pts.shape[0]}");

        x, y = pts[:, 0], pts[:, 1];
        if xmin is not None :
            mask = (x > xmin) & (x < xmax) & (y > ymin) & (y < ymax);
            pts = pts[mask];
        else :
            xmin, xmax = x.min(), x.max();
            ymin, ymax = y.min(), y.max();

        # print(xmin, xmax, ymin, ymax, res);
        xyz, rgb, depth = pts[:, :3], pts[:, 3:6], pts[:, 6];
        # print(f"Number of filtered points = {pts.shape[0]}");

        w = int(math.ceil((xmax - xmin) / res));
        h = int(math.ceil((ymax - ymin) / res));
        # print(f"Target (h, w) = ({h}, {w})");

        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2];
        x = x - xmin;
        y = y - ymin;
        x = np.floor(x / res).astype(np.int32);
        y = np.floor(y / res).astype(np.int32);
        # print(x.min(), x.max());
        # print(y.min(), y.max());

        # assign points to cells
        lin_id = w * y + x;
        # print(lin_id.max())
        cell_ids = {};
        for i in range(x.size) :
            lid = lin_id[i];
            if lid not in cell_ids :
                cell_ids[lid] = [i];
                continue;
            
            cell_ids[lid].append(i);

        
        if style == 'mindist' :
            # update cell id assignments 
            # by selecting the nearest point per cell
            for ci in cell_ids :
                pids = cell_ids[ci];
                i_closest = pids[np.argmin(depth[pids])];
                cell_ids[ci] = i_closest;

            # print(len(cell_ids))
            rgb = rgb.astype(np.uint8);
            im_rgb = np.zeros((h, w, 3), dtype=rgb.dtype);
            im_h = np.zeros((h, w), dtype=pts.dtype);
            im_mask = np.zeros((h, w), dtype=bool);
            hmin, hmax = 1e6, -1e6;
            for ci, i in cell_ids.items() :
                xi = ci % w;
                yi = (ci - xi) // w;
                if (xi >= w) or (yi >= h) :
                    self.count_oob += 1;
                    continue;
                im_rgb[yi, xi] = rgb[i];
                im_h[yi, xi] = z[i];
                hmin = min(hmin, z[i]);
                hmax = max(hmax, z[i]);
                im_mask[yi, xi] = True;

            # normalize just for visualization
            im_h[im_mask] = (im_h[im_mask] - hmin) / (hmax - hmin);
            im_h = self.gray_to_rgb(im_h);
            # im_mask = self.gray_to_rgb(im_mask.astype(np.float32));
            im_mask = 255 * im_mask.astype(np.uint8);
            im_mask = np.stack(( im_mask, im_mask, im_mask ), axis=-1);
            # print(im_mask.shape); sys.exit();

        else :
            raise NotImplementedError;

        concat_axis = 1 if concat_axis=='horizontal' else 0;
        im = np.concatenate((im_rgb, im_h, im_mask), axis=concat_axis);
        self.save_image(out_fpath, im);


    def save_map_wrt_first_frame(self, 
        traj_i : int,
        n_traj: int,
        traj_dir: str, 
        downsample_stride: int,
    ) :

        traj_name = osp.basename(traj_dir);
        tmp_out_dir = osp.join(self.out_dir, traj_name);
        os.makedirs(tmp_out_dir, exist_ok=True);
        print(f"[{traj_i} / {n_traj}] Building full map = {traj_name} --> {tmp_out_dir}");

        pts_file_list, motion_filepath, pose_filepath = \
                self.get_traj_file_list_single(traj_dir);
        motions = self.pose_loader(motion_filepath);                
        odoms = self.pose_loader(pose_filepath);  

        if (self.max_n_frames>0) and len(pts_file_list) > self.max_n_frames :
            print(f"Total #frames ({len(pts_file_list)}) > max #frames allowed ({self.max_n_frames}), so not processing");
            return;
        
            print(f"Total #frames ({len(pts_file_list)}) > max #frames allowed ({self.max_n_frames}), so clamping ...");

            pts_file_list = pts_file_list[:self.max_n_frames];
            motions = motions[:self.max_n_frames-1];
            odoms = odoms[:self.max_n_frames];            

        if downsample_stride is None :
            downsample_stride = len(pts_file_list);

        xyz_all, rgb_all, depth_all = None, None, None;
        for i in tqdm(range(len(pts_file_list)-1, -1, -1)) :
            pts_path = pts_file_list[i];

            xyzrgb = self.pts_loader(pts_path);
            xyz, rgb = xyzrgb[:, :3], xyzrgb[:, 3:];
            depth = np.linalg.norm(xyz, axis=1, keepdims=True);

            if xyz_all is None :
                xyz_all = xyz;
                rgb_all = rgb;
                depth_all = depth;
            else :
            
                # if i > 0 :
                #     # # register the current frame to the previous frames
                #     # xyz_all = self.transform_post_to_this(xyz_all, motions[i]);

                # register the current frame to the previous frames
                xyz_all = self.transform_post_to_this(xyz_all, motions[i]);                    

                xyz_all = np.concatenate((xyz_all, xyz), axis=0);
                rgb_all = np.concatenate((rgb_all, rgb), axis=0);
                depth_all = np.concatenate((depth_all, depth), axis=0);
                assert xyz_all.dtype == xyz.dtype;

            # if (i % downsample_stride == 0) and (i != len(pts_file_list)-1) :
            #     map_pts = np.concatenate((xyz_all, rgb_all, depth_all), axis=1);
            #     map_pts, _ = self.downscale_map_w_mindist(map_pts);
            #     xyz_all, rgb_all, depth_all = map_pts[:, :3], map_pts[:, 3:6], map_pts[:, 6:];
           
        assert xyz_all.dtype == xyz.dtype;        

        dense_map_fpath = osp.join(tmp_out_dir, self.dense_traj_map_file_name);
        map_pts = np.concatenate((xyz_all, rgb_all, depth_all), axis=1);
        print(f"(Before alignment) Shape of complete map point cloud w/ color = {map_pts.shape}");
        self.save_single(map_pts,  dense_map_fpath);       

        # get rgb image from pc
        dense_img_path = self.get_image_name_from_pc_name(dense_map_fpath);
        self.gen_map_2d_single(
            dense_map_fpath, 
            dense_img_path, 
            'mindist',
            None, None, None, None, 0.1,
            'vertical',
        );

        
        # gravity alignment
        xyz_all = self.gravity_align2(xyz_all, odoms[0]);
        assert xyz_all.dtype == xyz.dtype;        

        dense_map_fpath = osp.join(tmp_out_dir, self.dense_traj_map_g_aligned_file_name);
        map_pts = np.concatenate((xyz_all, rgb_all, depth_all), axis=1);
        print(f"(After alignment) Shape of complete map point cloud w/ color = {map_pts.shape}");
        self.save_single(map_pts,  dense_map_fpath);
    
        # get rgb image from pc
        dense_img_path = self.get_image_name_from_pc_name(dense_map_fpath);
        self.gen_map_2d_single(
            dense_map_fpath, 
            dense_img_path, 
            'mindist',
            None, None, None, None, 0.1,
            'vertical',            
        );



    def run(self) -> None :

        traj_list = self.list_traj_dirs();
        n_traj = len(traj_list);

        print('=' * 40);
        print("Building complete maps first ...");
        print('=' * 40);
        for traj_i, traj_subdir in enumerate(traj_list, 1) :
            traj_dir = osp.join(self.data_dir, traj_subdir);
            print(traj_i, traj_subdir, traj_dir);

            self.save_map_wrt_first_frame(
                traj_i, 
                n_traj, 
                traj_dir,
                downsample_stride=None,
            );   

        print();
        print('=' * 40);
        print("Generating per frame image/labels ...");
        print('=' * 40);
        for traj_i, traj_subdir in enumerate(traj_list, 1) :
            traj_dir = osp.join(self.data_dir, traj_subdir);
            print(traj_i, traj_subdir, traj_dir);

            self.gen_data_single_traj(
                traj_i,
                n_traj,
                traj_dir, 
            );

            # break;

        #     if traj_i == 50 :
        #         sys.exit();




def main() :
    global args;

    data_gen = TemporalDataGenerator(
                    args.data_dir,
                    args.pts_subdir,
                    args.motion_subpath,
                    args.pose_subpath,
                    args.out_dir,
                    max_n_frames=args.max_n_frames,
    );

    data_gen.run();


if __name__ == "__main__" :
    main();
    sys.exit();