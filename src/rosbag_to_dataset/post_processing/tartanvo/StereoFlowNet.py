import torch
from torch import nn
import torch.nn.functional as F
from .modules import FeatureExtraction, HourglassEncoder, HourglassDecoder

class StereoNet(nn.Module):
    def __init__(self, act_fun='relu'):
        super(StereoNet, self).__init__()
        self.feature_extraction = FeatureExtraction(last_planes=64, bigger=True, middleblock=9) # return 1/2 size feature map
        self.encoder = HourglassEncoder(in_channels=64*2+6, act_fun=act_fun)
        self.decoder = HourglassDecoder(out_channels=1, act_fun=act_fun)

        self.decoder_unc = HourglassDecoder(out_channels=1, act_fun=act_fun, hourglass=False)

    def forward(self, x ):
        x1 = x.reshape(x.shape[0]*2, x.shape[1]//2, x.shape[2], x.shape[3])
        x1 = self.feature_extraction(x1)
        x1 = x1.view(x1.shape[0]//2, x1.shape[1]*2, x1.shape[2], x1.shape[3])
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = torch.cat((x1,x2),dim=1)

        x, cat0, cat1, cat2, cat3, cat4 = self.encoder(x)
        out0 = self.decoder(x, cat0, cat1, cat2, cat3, cat4)
        out_unc = self.decoder_unc(x, cat0, cat1, cat2, cat3, cat4)
        return out0, out_unc

    def calc_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        '''
        Note: criterion is not used when uncertainty is included
        '''
        if mask is not None: 
            output_ = output[mask]
            target_ = target[mask]
            if unc is not None:
                unc = unc[mask]
        else:
            output_ = output
            target_ = target

        if unc is None:
            return criterion(output_, target_), None
        else: # if using uncertainty, then no mask 
            
            diff = torch.abs( output_ - target_) # hard code L1 loss
            loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
            loss = torch.mean(diff)
            return  loss_unc/(1.0+lamb), loss

class FlowNet(nn.Module):
    def __init__(self, act_fun='relu', uncertainty=True):
        super(FlowNet, self).__init__()
        self.feature_extraction = FeatureExtraction(last_planes=64, bigger=True, middleblock=9) # return 1/2 size feature map
        self.encoder = HourglassEncoder(in_channels=64*2+6, act_fun=act_fun)
        self.decoder = HourglassDecoder(out_channels=2, act_fun=act_fun)

        self.decoder_unc = HourglassDecoder(out_channels=1, act_fun=act_fun, hourglass=False)

    def forward(self, x ):
        x1 = x.reshape(x.shape[0]*2, x.shape[1]//2, x.shape[2], x.shape[3])
        x1 = self.feature_extraction(x1)
        x1 = x1.view(x1.shape[0]//2, x1.shape[1]*2, x1.shape[2], x1.shape[3])
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        x = torch.cat((x1,x2),dim=1)

        x, cat0, cat1, cat2, cat3, cat4 = self.encoder(x)
        out0 = self.decoder(x, cat0, cat1, cat2, cat3, cat4)
        out_unc = self.decoder_unc(x, cat0, cat1, cat2, cat3, cat4)
        return out0, out_unc

    def calc_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        '''
        Note: criterion is not used when uncertainty is included
        '''
        if mask is not None: 
            output_ = output[mask]
            target_ = target[mask]
            if unc is not None:
                unc = unc[mask]
        else:
            output_ = output
            target_ = target

        if unc is None:
            return criterion(output_, target_)
        else: # if using uncertainty, then no mask 
            diff = torch.abs( output_ - target_) # hard code L1 loss
            loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
            loss = torch.mean(diff)
            return  loss_unc/(1.0+lamb), loss

class StereoFlowNet(nn.Module):
    def __init__(self, act_fun='relu'):
        super(StereoFlowNet, self).__init__()
        self.feature_extraction = FeatureExtraction(last_planes=64, bigger=True, middleblock=9) # return 1/2 size feature map
        self.s_encoder = HourglassEncoder(in_channels=64*2+6, act_fun=act_fun)
        self.s_decoder = HourglassDecoder(out_channels=1, act_fun=act_fun)
        self.s_decoder_unc = HourglassDecoder(out_channels=1, act_fun=act_fun, hourglass=False)

        self.f_encoder = HourglassEncoder(in_channels=64*2+6, act_fun=act_fun)
        self.f_decoder = HourglassDecoder(out_channels=2, act_fun=act_fun)
        self.f_decoder_unc = HourglassDecoder(out_channels=1, act_fun=act_fun, hourglass=False)

    def forward(self, x, stereo=True, flow=True ):
        '''
        x[:,0:3,:,:] right image: N x 3 x h x w
        x[:,3:6,:,:] left image:  N x 3 x h x w
        x[:,6:9,:,:] left image2: N x 3 x h x w
        '''
        # import ipdb;ipdb.set_trace()
        x1 = x.reshape(x.shape[0]*3, x.shape[1]//3, x.shape[2], x.shape[3])
        x1 = self.feature_extraction(x1)
        x1 = x1.view(x1.shape[0]//3, x1.shape[1]*3, x1.shape[2], x1.shape[3]) # N x 9 x H x W
        x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=True)
        s_x = torch.cat((x1[:, :128, :, :],x2[:, :6, :, :]),dim=1)
        f_x = torch.cat((x1[:, 64:, :, :],x2[:, 3:, :, :]),dim=1)

        if stereo:
            x, cat0, cat1, cat2, cat3, cat4 = self.s_encoder(s_x)
            s_out = self.s_decoder(x, cat0, cat1, cat2, cat3, cat4)
            s_out_unc = self.s_decoder_unc(x, cat0, cat1, cat2, cat3, cat4)
        else:
            s_out, s_out_unc = None, None

        if flow:
            x, cat0, cat1, cat2, cat3, cat4 = self.f_encoder(f_x)
            f_out = self.f_decoder(x, cat0, cat1, cat2, cat3, cat4)
            f_out_unc = self.f_decoder_unc(x, cat0, cat1, cat2, cat3, cat4)
        else:
            f_out, f_out_unc = None, None

        return s_out, s_out_unc, f_out, f_out_unc

    def calc_loss(self, output, target, criterion, mask=None, unc=None, lamb=1.0):
        '''
        Note: criterion is not used when uncertainty is included
        '''
        if mask is not None: 
            output_ = output[mask]
            target_ = target[mask]
            if unc is not None:
                unc = unc[mask]
        else:
            output_ = output
            target_ = target

        if unc is None:
            return criterion(output_, target_)
        else: # if using uncertainty, then no mask 
            diff = torch.abs( output_ - target_) # hard code L1 loss
            loss_unc = torch.mean(torch.exp(-unc) * diff + unc * lamb)
            loss = torch.mean(diff)
            return  loss_unc/(1.0+lamb), loss

if __name__ == '__main__':
    
    stereonet = StereoFlowNet()
    stereonet.cuda()
    # print (stereonet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    np.set_printoptions(precision=4, threshold=100000)
    x, y = np.ogrid[:512, :256]
    # print (x, y, (x+y))
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print (img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    # imgInput = np.concatenate((imgInput,imgInput),axis=0)
    imgInput = np.concatenate((imgInput,imgInput,imgInput),axis=1)

    target = torch.zeros((imgInput.shape[0], 1, 512, 256), dtype=torch.float32).cuda()
    target2 = torch.zeros((imgInput.shape[0], 2, 512, 256), dtype=torch.float32).cuda()

    starttime = time.time()
    ftime, edtime = 0., 0.
    for k in range(10):
        imgTensor = torch.from_numpy(imgInput)
        z, unc, f, func = stereonet(imgTensor.cuda())
        print (z.data.cpu().numpy().shape, unc.data.cpu().numpy().shape, f.data.cpu().numpy().shape, func.data.cpu().numpy().shape)
        print(stereonet.calc_loss(z, target, None, unc=unc))
        print(stereonet.calc_loss(f, target2, None, unc=unc))
    print (time.time() - starttime, ftime, edtime)