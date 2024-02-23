# terrain_map_tartandrive
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from .utils import quat_to_yaw, SO2quat, quat2SE, quats2SEs, SOs2Eulers, pose2motion
np.set_printoptions(suppress=True,precision=4)

def get_local_path(odom, ref_ind=0):
    pos = odom[:,:2]
    quat = odom[:,3:7]
    yaw = quat_to_yaw(quat)

    rel_pos = pos[:,:] - pos[ref_ind,:]
    rel_yaw = yaw[:] - yaw[ref_ind]

    theta = -yaw[ref_ind]
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # rel_pos = np.matmul(rot, rel_pos.transpose((1,0))).transpose((1,0))
    rel_pos = np.matmul(rot, rel_pos[...,np.newaxis]).squeeze(axis=-1)
    rel_path = np.concatenate((rel_pos, rel_yaw[...,np.newaxis]), axis=1)
    return rel_path
    
class TerrainMap:
    def __init__(self, map_metadata = {}, device='cpu'):
        '''
        A simpler version of the original TerrainMap developed in ss_costmap
        This one only looks at the crop coordinates, but won't returns the patches
        '''
        self.map_metadata = map_metadata
        self.device = device 

        self.resolution = self.map_metadata['resolution']
        self.origin = self.map_metadata['origin']

        self.map_width = int(map_metadata['width']/self.resolution)
        self.map_height = int(map_metadata['height']/self.resolution)

    def get_crop_batch_and_masks(self, poses, crop_params, maskonly=False):
        '''Obtain an NxCxHxW tensor of crops for a given path.

        Procedure:
        1. Get initial meshgrid for crop in metric space centered around the origin
        2. Apply affine transform to all the crop positions (to the entire meshgrid) using batch multiplication:
            - The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]
        3. Center around the right origin and rescale to obtain pixel coordinates
        4. Obtain map values at those pixel coordinates and handle invalid coordinates using a mask

        Note: 
        - All tensors will obey the following axis convention: [batch x crop_x x crop_y x transform_x x transform_y]
        - Input is in metric coordinates, so we flip the terrain map axes where necessary to match robot-centric coordinates
        
        Args:
            - path:
                Nx3 tensor of poses, where N is the number of poses to evaluate and each pose is represented as [x, y, yaw].
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crops:
                Tensor of NxCxHxW of crops at poses on the path, where C is the number of channels in self.maps and N is the number of points in the path
        '''
        ## Create initial crop template in metric space centered around (0,0) to generate all pixel values
        # import ipdb;ipdb.set_trace()
        crop_height = crop_params['crop_size'][0] # In meters
        crop_width = crop_params['crop_size'][1] # In meters
        output_height = crop_params['output_size'][0] # In pixels
        output_width = crop_params['output_size'][1] # In pixels

        crop_xs = torch.linspace((-crop_height+self.resolution)/2., (crop_height-self.resolution)/2., output_height).to(self.device)
        crop_ys = torch.linspace((-crop_width+self.resolution)/2., (crop_width-self.resolution)/2., output_width).to(self.device)
        crop_positions = torch.stack(torch.meshgrid(crop_xs, crop_ys, indexing="ij"), dim=-1) # HxWx2 tensor

        ## Obtain translations and rotations for 2D rigid body transformation
        translations = poses[:, :2]  # Nx2 tensor, [x, y] in metric space
        yaws = poses[:,2]
        rotations = torch.stack([torch.cos(yaws), -torch.sin(yaws), torch.sin(yaws), torch.cos(yaws)], dim=-1)  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]

        ## Reshape tensors to perform batch tensor multiplication. 
        rotations = rotations.view(-1, 1, 1, 2, 2).float() #[B x 1 x 1 x 2 x 2]
        crop_positions = crop_positions.view(1, *crop_params['output_size'], 2, 1).float() #[1 x H x W x 2 x 1]
        translations = translations.view(-1, 1, 1, 2, 1).float() #[B x 1 x 1 x 2 x 1]

        # Apply each transform to all crop positions (res = [B x H x W x 2])
        crop_positions_transformed = (torch.matmul(rotations, crop_positions) + translations).squeeze()

        # Obtain actual pixel coordinates
        map_origin = torch.Tensor(self.origin).view(1, 1, 1, 2).to(self.device)
        pixel_coordinates = ((crop_positions_transformed - map_origin) / self.resolution).long()  # .long() is needed so that we can use these as indices

        # Obtain maximum and minimum values of map to later filter out pixel locations that are out of bounds
        map_p_low = torch.tensor([0, 0]).to(self.device).view(1, 1, 1, 2)
        map_p_high = torch.tensor([self.map_height, self.map_width]).to(self.device).view(1, 1, 1, 2)
        valid_mask = (pixel_coordinates >= map_p_low).all(dim=-1) & (pixel_coordinates < map_p_high).all(dim=-1)  # If map is not square we might need to swap these axes as well

        return pixel_coordinates, valid_mask

    def get_centural_points(self, poses):
        '''Obtain an NxCxHxW tensor of crops for a given path.

        Procedure:
        1. Get initial meshgrid for crop in metric space centered around the origin
        2. Apply affine transform to all the crop positions (to the entire meshgrid) using batch multiplication:
            - The goal is to obtain a tensor of size [B, H, W, 2], where B is the batch size, H and W are the dimensions fo the image, and 2 corresponds to the actual x,y positions. To do this, we need to rotate and then translate every pair of points in the meshgrid. In batch multiplication, only the last two dimensions matter. That is, usually we need the following dimensions to do matrix multiplication: (m,n) x (n,p) -> (m,p). In batch multiplication, the last two dimensions of each array need to line up as mentioned above, and the earlier dimensions get broadcasted (more details in the torch matmul page). Here, we will reshape rotations to have shape [B,1,1,2,2] where B corresponds to batch size, the two dimensions with size 1 are there so that we can broadcast with the [H,W] dimensions in crop_positions, and the last two dimensions with size 2 reshape the each row in rotations into a rotation matrix that can left multiply a position to transform it. The output of torch.matmul(rotations, crop_positions) will be a [B,H,W,2,1] tensor. We will reshape translations to be a [B,1,1,2,1] vector so that we can add it to this output and obtain a tensor of size [B,H,W,2,1], which we will then squeeze to obtain our final tensor of size [B,H,W,2]
        3. Center around the right origin and rescale to obtain pixel coordinates
        4. Obtain map values at those pixel coordinates and handle invalid coordinates using a mask

        Note: 
        - All tensors will obey the following axis convention: [batch x crop_x x crop_y x transform_x x transform_y]
        - Input is in metric coordinates, so we flip the terrain map axes where necessary to match robot-centric coordinates
        
        Args:
            - path:
                Nx3 tensor of poses, where N is the number of poses to evaluate and each pose is represented as [x, y, yaw].
            - crop_params:
                Dictionary of params for crop:

                {'crop_size': [crop_size_x, crop_size_y],
                 'output_size': [output_size_x, output_size_y]}

                crop_size is in meters, output_size is in pixels

        Returns:
            - crops:
                Tensor of NxCxHxW of crops at poses on the path, where C is the number of channels in self.maps and N is the number of points in the path
        '''
        ## Create initial crop template in metric space centered around (0,0) to generate all pixel values
        # import ipdb;ipdb.set_trace()

        crop_positions = torch.tensor([0.0, 0.0]) # HxWx2 tensor

        ## Obtain translations and rotations for 2D rigid body transformation
        translations = poses[:, :2]  # Nx2 tensor, [x, y] in metric space
        yaws = poses[:,2]
        rotations = torch.stack([torch.cos(yaws), -torch.sin(yaws), torch.sin(yaws), torch.cos(yaws)], dim=-1)  # Nx4 tensor where each row corresponds to [cos(theta), -sin(theta), sin(theta), cos(theta)]

        ## Reshape tensors to perform batch tensor multiplication. 
        rotations = rotations.view(-1, 2, 2).float() #[B x 2 x 2]
        crop_positions = crop_positions.view(1, 2, 1).float() #[1 x 2 x 1]
        translations = translations.view(-1, 2, 1).float() #[B x 2 x 1]

        # Apply each transform to all crop positions (res = [B x 2])
        crop_positions_transformed = (torch.matmul(rotations, crop_positions) + translations).squeeze()

        # Obtain actual pixel coordinates
        map_origin = torch.Tensor(self.origin).view(1, 2).to(self.device)
        pixel_coordinates = ((crop_positions_transformed - map_origin) / self.resolution).long()  # .long() is needed so that we can use these as indices

        # Obtain maximum and minimum values of map to later filter out pixel locations that are out of bounds
        map_p_low = torch.tensor([0, 0]).to(self.device).view( 1, 2)
        map_p_high = torch.tensor([self.map_height, self.map_width]).to(self.device).view( 1, 2)
        valid_mask = (pixel_coordinates >= map_p_low).all(dim=-1) & (pixel_coordinates < map_p_high).all(dim=-1)  # If map is not square we might need to swap these axes as well

        return pixel_coordinates, valid_mask

def integrate_motion(motion):
    '''
    Integrate the motion 
    Output a list of poses in the matrix format
    '''
    pose = np.matrix(np.eye(4,4))
    poselist = [pose,]
    for k in range(motion.shape[0]):
        # np.savetxt(outdir + '/' + filestr +'_motion.txt', motion_quat)
        if len(motion[k].shape) == 1: # in the quaternion
            motion_mat = quat2SE(motion[k])
        else:
            motion_mat = motion[k]
        pose = pose * motion_mat
        # quat = SO2quat(pose[0:3,0:3])
        # pose_quat = [pose[0,3], pose[1,3], pose[2,3], quat[0], quat[1], quat[2], quat[3]]
        poselist.append(pose)
        # np.savetxt(outdir + '/' + filestr +'_pose.txt', pose_quat)
    return poselist

def transform_ground(poses):
    # the R and T come from ground calibration
    # this is used for Yamaha ATV! 
    # After transformation, the z axis will be pointing up, x pointing forward, y pointing left
    # SO2quat(R) = array([ 0.99220717,  0.00153886, -0.12397937,  0.01231552])
    # starttime = time.time()
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    # T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    T = np.eye(4)
    T[:3,:3] = R
    poses_trans = np.matmul(T, np.array(poses)) #+ T.transpose(1,0)
    return poses_trans

def transform_ground_motion(motions):
    R = np.array([[ 0.9692535,   0.00610748, -0.24598853],
                  [ 0.,         -0.99969192, -0.02482067],
                  [-0.24606434, 0.02405752,  -0.96895489]] )
    # T = np.array([[-5.55111512e-17], [-6.93889390e-18], [1.77348523e+00]])
    T = np.eye(4)
    T[:3,:3] = R
    T_inv = np.linalg.inv(T)
    motions_trans = T @ np.array(motions) @ T_inv #+ T.transpose(1,0)
    return motions_trans

def tartanvo_motion_to_odom(motion):
    '''
    motion is in the [x,y,z,qx,qy,qz,qw] format
    Integrate the motion from the first frame
    Transform the poses to the vehile frame, according to the ground calibration
    '''
    poselist = integrate_motion(motion)
    pose_trans = transform_ground(poselist)
    pose_transformed_quat = SO2quat(pose_trans[:,:3,:3]) # n x 4
    pose_transformed = np.concatenate((pose_trans[:,:3,3], pose_transformed_quat),axis=1) # n x 7
    return np.array(pose_transformed, dtype=np.float32)

def tartanmotion2vel(motions_mat, dt=0.1):
    # motions_mat = quats2SEs(motions)
    motions_trans = transform_ground_motion(motions_mat)
    motions_rot = SOs2Eulers(motions_trans[:,:3,:3])
    vels = np.concatenate((motions_trans[:,0:1,3], motions_rot[:,2:3]), axis=1)
    return vels.astype(np.float32)/dt # vx, vy, v_yaw

def gpsodom2vel(odom):
    '''
    odom: N x 13 numpy array (x, y, z, rx, ry, rz, rw, vx, vy, xz, vrx, vry, vrz) in the global frame
    res: (forward vel, yaw)
    '''
    # import ipdb;ipdb.set_trace()
    ori_T = Rotation.from_quat(odom[:,3:7]).as_matrix()
    ori_T_inv = np.linalg.inv(ori_T)

    vel_trans = np.matmul(ori_T_inv, odom[:,7:10,np.newaxis]).squeeze(axis=-1) # N x 3
    vel_rot = np.matmul(ori_T_inv, odom[:,10:13,np.newaxis]).squeeze(axis=-1) # N x 3

    vel = vel_trans[:,0:1] # for tartandrive data (2021), the x and y axis are different
    yaw = vel_rot[:,2:3]
    # print(vel_trans)
    # print(vel_rot)
    return np.concatenate((vel,yaw),axis=1).astype(np.float32)

#     velx = motion[:,0:1] / 0.1
#     rot = Rotation.from_quat(motion[:,3:7]).as_euler("XYZ", degrees=False)
#     yaw = rot[:,2:3]
#     yaw = yaw / 0.1
#     return np.concatenate((velx,yaw),axis=1).astype(np.float32)

if __name__ == "__main__":
    from os.path import join
    from os import listdir
    import cv2
    from .utils import add_text
    
    base_dir = '/home/wenshan/tmp/arl_data/full_trajs/20220531_lowvel_0'
    
    odom_folder = 'tartanvo_odom' #'odom' #
    rgbmap_folder = 'rgb_map'
    heightmap_folder ='height_map'

    # base_dir = '/cairo/arl_bag_files/SARA/sara_obs_traj_old/sara_obs_0'

    map_height = 12.0
    map_width = 12.0
    resolution = 0.02

    crop_width = 4  # in meters
    crop_size = [crop_width, crop_width]
    output_size = [224, 224]


    map_metadata = {
        'height': map_height,
        'width': map_width,
        'resolution': resolution,
        'origin': [-2.0, -6.0],
    }

    crop_params ={
        'crop_size': crop_size,
        'output_size': output_size
    }

    # using novatel
    # odom = np.load(join(base_dir, odom_folder, 'odometry.npy'))
    # using tartanvo
    motions = np.load(join(base_dir, odom_folder, 'motions.npy'))
    # print(motions.shape)
    # tartanvo_motion_to_odom(motions)
    # tartanodom2vel(motions)
    maplist = listdir(join(base_dir, rgbmap_folder))
    maplist = [mm for mm in maplist if mm.endswith('.npy')]
    maplist.sort()

    startframe = 0
    # rgblist = []
    cropnum = 26
    # # for startframe in range(startframe, len(maplist), 1):
    for startframe in range(startframe, len(maplist)-cropnum, 1):
        print(startframe)
        rgbmap = np.load(join(base_dir, rgbmap_folder, maplist[startframe]))
        heightmap = np.load(join(base_dir, heightmap_folder, maplist[startframe]))

        rgb_map_tensor = torch.from_numpy(rgbmap).permute(2,0,1) # (C,H,W)
        height_map_tensor = torch.from_numpy(heightmap).permute(2,0,1) # (C,H,W)

        tm = TerrainMap(map_metadata=map_metadata, device="cpu")

        # for tartanvo 
        pose_transfromed = tartanvo_motion_to_odom(motions[startframe:startframe+cropnum,:])
        local_path = get_local_path(pose_transfromed[:cropnum,:])
        local_path = torch.from_numpy(local_path)

        # visualize the map on the top-down view map
        map_path, valid_mask = tm.get_centural_points(local_path)
        map_path = map_path.numpy()
        # import ipdb;ipdb.set_trace()
        for k in range(len(map_path)-1):
            if not valid_mask[k] or not valid_mask[k+1]:
                continue
            cv2.line(rgbmap, (map_path[k,1], map_path[k,0]), (map_path[k+1,1], map_path[k+1,0]),color=(0,0,255), thickness=2)
        cv2.imshow('img', rgbmap)
        cv2.waitKey(0)