'''
From StereoNet5
Fewer blocks in feature extractor
SSP in ED
'''
import torch 
import torch.nn as nn
import torch.nn.functional as F
import math
from .PSM import feature_extraction, Hourglass 


def predict_layer(in_planes):
    return nn.Conv2d(in_planes,1,kernel_size=3,stride=1,padding=1,bias=True)

class SSP(nn.Module):
    def __init__(self, in_planes):
        super(SSP, self).__init__()
        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     nn.Conv2d(in_planes, int(in_planes/4), 1, 1, 0),
                                     nn.ReLU(inplace=True))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     nn.Conv2d(in_planes, int(in_planes/4), 1, 1, 0),
                                     nn.ReLU(inplace=True))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     nn.Conv2d(in_planes, int(in_planes/4), 1, 1, 0),
                                     nn.ReLU(inplace=True))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     nn.Conv2d(in_planes, int(in_planes/4), 1, 1, 0),
                                     nn.ReLU(inplace=True))

    def forward(self, x):
        target_h, target_w = x.size()[2],x.size()[3]

        output_branch1 = self.branch1(x)
        output_branch1 = F.interpolate(output_branch1, [target_h, target_w], mode='bilinear')

        output_branch2 = self.branch2(x)
        output_branch2 = F.interpolate(output_branch2, [target_h, target_w], mode='bilinear')

        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, [target_h, target_w], mode='bilinear')

        output_branch4 = self.branch4(x)
        output_branch4 = F.interpolate(output_branch4, [target_h, target_w], mode='bilinear')

        output_feature = torch.cat((x, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        return output_feature


class StereoNet7(nn.Module):
    def __init__(self, version=0, group_norm=False, act_fun='relu'):
        super(StereoNet7, self).__init__()
        self.version = version
        # feature extraction layers
        self.feature_extraction = feature_extraction(last_planes=64, bigger=True, middleblock=3) # return 1/2 size feature map

        if act_fun == 'relu':
            self.actfun = F.relu
        elif act_fun == 'selu':
            self.actfun = F.selu
        else:
            print ('Unknown activate function {}'.format( act_fun))
            self.actfun = F.relu

        # depth regression layers
        self.conv_c0 = nn.Conv2d(134,64, kernel_size=3, padding=1) # 1/2
        self.conv_c1 = Hourglass(2, 64, 0) # 1/2
        self.conv_c2 = Hourglass(2, 64, 0) # 1/4 #nn.Conv2d(128,128, kernel_size=3, padding=1)
        self.conv_c2_SSP = SSP(64) # 1/4
        self.conv_c3 = Hourglass(2, 128, 64) # 1/8 #nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv_c4 = Hourglass(2, 192, 64) # 1/16 #nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.conv_c5 = nn.Conv2d(256,384, kernel_size=3, padding=1) # 1/32
        self.conv_c6 = nn.Conv2d(384,512, kernel_size=3, padding=1) # 1/64
        self.deconv_c7 = nn.ConvTranspose2d(896, 320,kernel_size=4,stride=2,padding=1) # 1/16
        self.deconv_c8 = nn.ConvTranspose2d(576, 192,kernel_size=4,stride=2,padding=1) # 1/8
        self.conv_c8 = Hourglass(2, 192, 0) # 1/8
        self.deconv_c9 = nn.ConvTranspose2d(384, 128,kernel_size=4,stride=2,padding=1) # 1/4
        self.conv_c9 = Hourglass(2, 128, 0) # 1/4
        self.deconv_c10 = nn.ConvTranspose2d(256, 64,kernel_size=4,stride=2,padding=1) # 1/2
        self.conv_c10 = Hourglass(2, 64, 0) # 1/2
        self.deconv_c11 = nn.ConvTranspose2d(128, 64,kernel_size=4,stride=2,padding=1) # 1/1
        self.conv_c12 = nn.Conv2d(64, 16,kernel_size=1,padding=0)
        self.conv_c13 = nn.Conv2d(16, 1, kernel_size=1,padding=0)

        self.conv_c6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.deconv_c7_2 = nn.ConvTranspose2d(512, 512,kernel_size=4,stride=2,padding=1) # 1/32


    def forward(self, x):
        left = self.feature_extraction(x[:, 0:3, :, :])
        right = self.feature_extraction(x[:, 3:6, :, :])
        x2 = F.interpolate(x[:, 0:3, :, :], scale_factor=0.5, mode='bilinear')
        x3 = F.interpolate(x[:, 3:6, :, :], scale_factor=0.5, mode='bilinear')
        x = torch.cat((left,right,x2,x3),dim=1)

        # depth regression layers
        x = self.conv_c0(x) # 1/2
        cat0 = self.conv_c1(x) # 1/2 - 64
        x = self.conv_c2(cat0) # 1/2
        x = F.max_pool2d(x, kernel_size=2) # 1/4 - 64
        cat1 = self.conv_c2_SSP(x) # 1/4 - 128
        x = self.conv_c3(cat1) # 1/8
        cat2 = F.max_pool2d(x, kernel_size=2) # 1/8 - 192
        x = self.conv_c4(cat2)
        cat3 = F.max_pool2d(x, kernel_size=2) # 1/16 - 256
        x = self.conv_c5(cat3)
        x = self.actfun(x, inplace=True)
        cat4 = F.max_pool2d(x, kernel_size=2) # 1/32 - 384
        x = self.conv_c6(cat4)
        x = self.actfun(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2) # 1/64 - 512
        x = self.conv_c6_2(x)
        x = self.actfun(x, inplace=True)

        x = self.deconv_c7_2(x) # 1/32 - 512
        x = self.actfun(x, inplace=True)
        x = torch.cat((x,cat4),dim=1) #  - 896
        x = self.deconv_c7(x) # 1/16 - 320
        x = self.actfun(x, inplace=True)
        x = torch.cat((x,cat3),dim=1) # - 576
        x = self.deconv_c8(x) # 1/8 - 192 
        x = self.actfun(x, inplace=True)
        x = self.conv_c8(x)
        x = torch.cat((x,cat2),dim=1) # - 384
        x = self.deconv_c9(x) # 1/4 - 128
        x = self.actfun(x, inplace=True)
        x = self.conv_c9(x)
        x = torch.cat((x,cat1),dim=1) # - 256
        x = self.deconv_c10(x) # 1/2 - 64
        x = self.actfun(x, inplace=True)
        x = self.conv_c10(x)
        x = torch.cat((x,cat0),dim=1) # - 128
        x = self.deconv_c11(x) # 1/1 - 64
        x = self.actfun(x, inplace=True)

        x = self.conv_c12(x)
        x = self.actfun(x, inplace=True)
        out0 = self.conv_c13(x)
        # x = F.relu(x, inplace=True)

        return out0, None

if __name__ == '__main__':
    
    stereonet = StereoNet7()
    stereonet.cuda()
    # print stereonet
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    np.set_printoptions(precision=4, threshold=100000)
    x, y = np.ogrid[:480, :896]
    # print x, y, (x+y)
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print(img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    # imgInput = np.concatenate((imgInput,imgInput),axis=0)
    imgInput = np.concatenate((imgInput,imgInput),axis=1)

    starttime = time.time()
    for k in range(10):
        imgTensor = torch.from_numpy(imgInput)
        z = stereonet(imgTensor.cuda())
        print(z.data.cpu().numpy().shape)
    print(time.time() - starttime)
    # print(z.data.numpy())

    for name,param in stereonet.named_parameters():
      print(name,param.requires_grad)

