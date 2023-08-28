import numpy as np
import torch
import cv2

# ['heightmap', 'rgbmap', 'odom', 'odom_tartanvo', 'cost', 'patches', 'masks']
# to tensor, normalization, speed
# training script
# overfitting

# filter wrt the unknown percentage
# normalize the input
# random shift z height?
class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def quat_to_yaw(quat):
    """
    Convert a quaternion (as [x, y, z, w]) to yaw
    """
    if isinstance(quat, torch.Tensor):
        if len(quat.shape) < 2:
            return quat_to_yaw(quat.unsqueeze(0)).squeeze()

        return torch.atan2(2 * (quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]), 1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))

    elif isinstance(quat, np.ndarray):
        if len(quat.shape) < 2:
            return quat_to_yaw(quat[np.newaxis,...])[0]

        return np.arctan2(2 * (quat[:, 3]*quat[:, 2] + quat[:, 0]*quat[:, 1]), 1 - 2 * (quat[:, 1]**2 + quat[:, 2]**2))

def add_text( img, text, offset_height = 0):
    text_bg_color = (230, 130, 10) # BGR
    text_color = (70, 200, 230)
    textlen = len(text)
    bg_width = textlen * 12
    bg_height = 30
    x, y = 10 + offset_height , 0
    
    img[x:x+bg_height, y:y+bg_width, :] = img[x:x+bg_height, y:y+bg_width, :] * 0.5 + np.array(text_bg_color) * 0.5
    cv2.putText(img,text,(y+10, x + 5 + bg_height//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, thickness=1)

    return img

from scipy.spatial.transform import Rotation as R

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

def so2SO(so_data):
    return R.from_rotvec(so_data).as_matrix()

def quat2SE(quat_data):
    '''
    quat_data: 7
    SE: 4 x 4
    '''
    SO = R.from_quat(quat_data[3:7]).as_matrix()
    SE = np.matrix(np.eye(4))
    SE[0:3,0:3] = np.matrix(SO)
    SE[0:3,3]   = np.matrix(quat_data[0:3]).T
    return SE

def quats2SEs(quat_datas):
    '''
    pos_quats: N x 7
    SEs: N x 4 x 4
    '''
    data_len = quat_datas.shape[0]
    SEs = np.zeros((data_len,4,4))
    for i_data in range(0,data_len):
        SE = quat2SE(quat_datas[i_data,:])
        SEs[i_data,:,:] = SE
    return SEs

def SOs2Eulers(SOs):
    rot = R.as_euler(R.from_matrix(SOs), 'XYZ', degrees=False)
    return rot

# def data_transform(sample, augment_data=False):
#     # Transform left_img=img0, right_img=img1, color_img=imgc, disparity image=disp0
#     # Convert to Tensor
#     # Transform to pytorch tensors, make sure they are all in CxHxW configuration
#     if "img0" in sample:
#         sample["img0"] = torch.unsqueeze(torch.stack([torch.from_numpy(img) for img in sample["img0"]],0), 0)/255.0
#     if "img1" in sample:
#         sample["img1"] = torch.unsqueeze(torch.stack([torch.from_numpy(img) for img in sample["img1"]],0), 0)/255.0
#     if "imgc" in sample:
#         img_transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#         ])
#         imgs = []
#         stacked_np = np.stack([img for img in sample["imgc"]],0)
#         for img in stacked_np:
#             img_torch = img_transform(img.astype(np.uint8))
#             imgs.append(img_torch)
#         sample["imgc"] = torch.stack(imgs, 0)
#     if "disp0" in sample:
#         sample["disp0"] = torch.unsqueeze(torch.stack([torch.from_numpy(img) for img in sample["disp0"]],0), 0)/255.0

#     # Transform heightmap:
#     # Convert to Tensor
#     # Clamp at [-2,2]
#     # Normalize so that it is between 0 and 1
#     # Make sure channels go first
#     if "heightmap" in sample:
#         hm = sample["heightmap"]
#         hm = torch.stack([torch.from_numpy(img) for img in hm],0)
#         hm_nan = torch.isnan(hm).any(dim=-1, keepdim=True) | (hm > 1e5).any(dim=-1, keepdim=True) | (hm < -1e5).any(dim=-1, keepdim=True)
#         hm = torch.nan_to_num(hm, nan=0.0, posinf=2, neginf=-2)
#         hm = torch.clamp(hm, min=-2, max=2)
#         hm = (hm - (-2))/(2 - (-2))
#         hm = torch.cat([hm, hm_nan], dim=-1)
#         hm = hm.permute(0,3,1,2)
#         sample["heightmap"] = hm

#     # Transform rgbmap:
#     # Convert to Tensor
#     # Normalize using ImageNet normalization
#     # Make sure channels go first
#     if "rgbmap" in sample:
#         img_transform = T.Compose([
#             T.ToTensor(),
#             T.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])
#         ])
#         imgs = []
#         stacked_np = np.stack([img for img in sample["rgbmap"]],0)
#         for img in stacked_np:
#             img_torch = img_transform(img.astype(np.uint8))
#             imgs.append(img_torch)
#         sample["rgbmap"] = torch.stack(imgs, 0)

#     # Transform cmd, odom, cost, imu to be tensors 
#     if "cmd" in sample:
#         sample["cmd"] = torch.from_numpy(sample["cmd"])

#     if "odom" in sample:
#         sample["odom"] = torch.from_numpy(sample["odom"])

#     if "cost" in sample:
#         sample["cost"] = torch.from_numpy(sample["cost"])

#     if "imu" in sample:
#         sample["imu"] = torch.from_numpy(sample["imu"])


#     # Transform patches:
#     # Convert to Tensor
#     # Clamp last 4 dimensions at [-2,2]
#     # import pdb;pdb.set_trace() 
#     if "patches" in sample:
#         patches = sample["patches"]
#         stacked_np = np.stack([img for img in patches],0)
#         # Process heightmaps
#         patches = torch.stack([torch.from_numpy(img) for img in patches],0)
#         patches_hm = patches[...,3:]
#         patches_rgb = stacked_np[...,:3] #patches[...,:3]

#         patches_hm_nan = torch.isnan(patches_hm).any(dim=-1, keepdim=True) | (patches_hm > 1e5).any(dim=-1, keepdim=True) | (patches_hm < -1e5).any(dim=-1, keepdim=True)
#         patches_hm = torch.nan_to_num(patches_hm, nan=0.0, posinf=2, neginf=-2)
#         patches_hm = torch.clamp(patches_hm, min=-2, max=2)
#         patches_hm = (patches_hm - (-2))/(2 - (-2))
#         patches_hm = torch.cat([patches_hm, patches_hm_nan], dim=-1)

#         # Process rgb maps
#         if augment_data:
#             img_transform = T.Compose([
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#                 T.RandomApply(torch.nn.ModuleList([
#                     T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
#                 ]), p=0.5)
#             ])
#         else:
#             img_transform = T.Compose([
#                 T.ToTensor(),
#                 T.Normalize(mean=[0.485, 0.456, 0.406],
#                             std=[0.229, 0.224, 0.225]),
#             ])

#         imgs = []
#         for img in patches_rgb:
#             img_torch = img_transform(img.astype(np.uint8))
#             imgs.append(img_torch)
#         patches_rgb = torch.stack(imgs,0)

#         patches_hm = patches_hm.permute(0,3,1,2)
#         patches = torch.cat([patches_rgb, patches_hm], dim=-3)
        

#         # # Add data augmentation 
#         if augment_data:
#             augment_transform = T.Compose([
#                 T.RandomVerticalFlip(p=0.5),
#                 T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1))
#             ])

#             patches = augment_transform(patches)

#         sample["patches"] = patches
#     return sample
