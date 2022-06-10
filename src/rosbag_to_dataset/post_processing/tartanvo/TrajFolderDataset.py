import numpy as np
import cv2
from torch.utils.data import Dataset
from os import listdir
import torch

def crop_imgs( sample, crop_w, crop_h_low, crop_h_high):
    for imgstr in {'img0', 'img1', 'imgc', 'img0n', 'intrinsic'}: 
        if imgstr in sample:
            img = sample[imgstr]
            img = img[crop_h_low:img.shape[0]-crop_h_high, crop_w:img.shape[1]-crop_w, :]
            sample[imgstr] = img
    return sample


def scale_imgs(sample, w, h):
    for imgstr in {'img0', 'img1', 'imgc', 'img0n', 'intrinsic'}: 
        if imgstr in sample:
            img = sample[imgstr]
            ori_h, ori_w = img.shape[0], img.shape[1]
            img = cv2.resize(img,(w, h))
            sample[imgstr] = img
    if 'blxfx' in sample: 
        sample['blxfx'] = sample['blxfx']  * w / ori_w
    return sample

def to_tensor(sample):
    for imgstr in {'img0', 'img1', 'img0n', 'intrinsic'}:
        if imgstr in sample: 
            img = sample[imgstr]
            img = img.transpose(2,0,1)
            imgTensor = torch.from_numpy(img).float()
            sample[imgstr] = imgTensor
    return sample

def normalize(sample, mean, std, keep_old=False):
    for imgstr in {'img0', 'img1', 'img0n'}:
        if imgstr in sample: 
            img = sample[imgstr]/float(255)
            for t, m, s in zip(img, mean, std):
                t.sub_(m).div_(s)
            if keep_old:
                sample[imgstr+'_norm'] = img
                sample[imgstr] = sample[imgstr]/float(255)
            else:
                sample[imgstr] = img
    return sample

def make_intrinsics_layer(w, h, fx, fy, ox, oy, ):
    ww, hh = np.meshgrid(range(w), range(h))
    ww = (ww.astype(np.float32) - ox + 0.5 )/fx
    hh = (hh.astype(np.float32) - oy + 0.5 )/fy
    intrinsicLayer = np.stack((ww,hh)).transpose(1,2,0)
    return intrinsicLayer

class TrajFolderDataset(Dataset):
    """Load images from a folder. """

    def __init__(self, rootfolder, leftfolder='image_left', rightfolder='image_right', colorfolder=None, forvo=False, \
                    imgw=1024, imgh=544, crop_w=64, crop_h_high=32, crop_h_low=32, resize_w=640, resize_h=448, \
                    focalx=320, focaly=320, centerx=320, centery=160, blxfx=80, stereomaps=None):
        '''
        forvo: true return two consequtive left images
               false return two corresponding left/right images
        '''
        
        imgleftfolder = rootfolder + '/' + leftfolder
        imgrightfolder = rootfolder + '/' + rightfolder
        leftfiles = listdir(imgleftfolder)
        rightfiles = listdir(imgrightfolder)
        self.leftfiles = [(imgleftfolder +'/'+ ff) for ff in leftfiles if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rightfiles = [(imgrightfolder +'/'+ ff) for ff in rightfiles if (ff.endswith('.png') or ff.endswith('.jpg'))]
        assert len(self.leftfiles)==len(self.rightfiles), "Left and right images are not consistent! "
        self.stereomaps = stereomaps
        
        self.leftfiles.sort()
        self.rightfiles.sort()

        # load colored image
        if colorfolder is not None:
            imgcolorfolder = rootfolder + '/' + colorfolder
            colorfiles = listdir(imgcolorfolder)
            self.colorfiles = [(imgcolorfolder +'/'+ ff) for ff in colorfiles if (ff.endswith('.png') or ff.endswith('.jpg'))]
            self.colorfiles.sort()
            assert len(self.leftfiles)==len(self.colorfiles), "Left and color images are not consistent! "
        else:
            self.colorfiles = None

        self.N = len(self.leftfiles)
        if forvo:
            self.N = self.N -1
            self.intrinsics = make_intrinsics_layer(imgw, imgh, focalx, focaly, centerx, centery)

        print('Find {} image files in {}'.format(self.N, imgleftfolder))

        self.forvo = forvo
        self.imgw, self.imgh = imgw, imgh
        self.crop_w, self.crop_h_high, self.crop_h_low = crop_w, crop_h_high, crop_h_low
        self.resize_w, self.resize_h = resize_w, resize_h
        self.focalx, self.focaly, self.centerx, self.centery = focalx, focaly, centerx, centery
        self.blxfx = blxfx

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        imgfile1 = self.leftfiles[idx].strip()
        imgfile2 = self.rightfiles[idx].strip() # for stereo matching
            
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        if self.stereomaps != '': # rectify stereo images (for warthog)
            img1 = cv2.remap( img1, self.stereomaps[0], self.stereomaps[1], cv2.INTER_LINEAR )
            img2 = cv2.remap( img2, self.stereomaps[2], self.stereomaps[3], cv2.INTER_LINEAR )

        sample = {'img0': img1, 
                'img1': img2, 
                'filename0': imgfile1.split('/')[-1], 
                'filename1': imgfile2.split('/')[-1]}

        if self.forvo:
            imgfilenext = self.leftfiles[idx+1].strip()
            img1n = cv2.imread(imgfilenext)
            if self.stereomaps != '': # rectify stereo images (for warthog)
                img1n = cv2.remap( img1n, self.stereomaps[0], self.stereomaps[1], cv2.INTER_LINEAR )
            sample['img0n'] = img1n
            sample['filename0n'] = imgfilenext.split('/')[-1]
            sample['intrinsic'] = self.intrinsics
            sample['blxfx'] = np.array([self.blxfx])

        if self.colorfiles is not None:
            imgcolor = cv2.imread(self.colorfiles[idx].strip())
            sample['imgc'] = imgcolor
        else:
            sample['imgc'] = img1.copy()

        # # image processing
        # hard code the preprocessing for stereo and vo
        if self.forvo: # for vo, resize to (896 x 448), crop to (640 x 448)
            sample = scale_imgs(sample, self.resize_w, self.resize_h)
            sample = crop_imgs(sample, self.crop_w, self.crop_h_low, self.crop_h_high)
        else:
            sample = crop_imgs(sample, self.crop_w, self.crop_h_low, self.crop_h_high)
            sample = scale_imgs(sample, self.resize_w, self.resize_h)
        # debug vo
        sample = to_tensor(sample)
        sample = normalize(sample,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], keep_old=self.forvo) 

        return sample


if __name__ == '__main__':
    kittidataset = TrajFolderDataset('/cairo/arl_bag_files/TartanCost/Trajectories/000150', \
                                    forvo=True)

    # import ipdb;ipdb.set_trace()
    for k in range(0,50):
        sample = kittidataset[k]
        img1 = sample['img0']
        img2 = sample['img1']
        filename1 = sample['filename0']
        filename2 = sample['filename1']
        print (filename1, filename2)

        img = np.concatenate((img1,img2),axis=0)

        cv2.imshow('img', img)
        cv2.waitKey(0)
        # import ipdb;ipdb.set_trace()
