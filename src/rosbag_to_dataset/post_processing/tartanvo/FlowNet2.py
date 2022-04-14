import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
from PSM import feature_extraction, Hourglass

def predict_layer(in_planes, middle_planes, out_planes):
    return nn.Sequential( nn.Conv2d(in_planes,middle_planes,kernel_size=3,stride=1,padding=1,bias=True),
                          nn.ReLU(),
                          nn.Conv2d(middle_planes,out_planes,kernel_size=3,stride=1,padding=1,bias=True)  )

class FlowNet2(nn.Module):
    """
    In the current setting, image width and height should be multipliers of 64
    """
    def __init__(self, version=0, act_fun='relu', scale=1, middleblock=16, uncertainty=False):
        super(FlowNet2, self).__init__()
        self.version = version
        # feature extraction layers
        self.feature_extraction = feature_extraction(last_planes=int(64/scale), bigger=True,middleblock=middleblock) # return 1/2 size feature map

        if act_fun == 'relu':
            self.actfun = F.relu
        elif act_fun == 'selu':
            self.actfun = F.selu
        else:
            print ('Unknown activate function', act_fun)
            self.actfun = F.relu

        # depth regression layers
        self.conv_c0 = nn.Conv2d(int(128/scale)+6,int(64/scale), kernel_size=3, padding=1) # 1/2
        self.conv_c1 = Hourglass(2, int(64/scale), 0) # 1/2
        self.conv_c2 = Hourglass(2, int(64/scale), int(64/scale)) # 1/4 #nn.Conv2d(128,128, kernel_size=3, padding=1)
        self.conv_c3 = Hourglass(2, int(128/scale), int(64/scale)) # 1/8 #nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv_c4 = Hourglass(2, int(192/scale), int(64/scale)) # 1/16 #nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.conv_c5 = nn.Conv2d(int(256/scale),int(384/scale), kernel_size=3, padding=1) # 1/32
        self.conv_c6 = nn.Conv2d(int(384/scale),int(512/scale), kernel_size=3, padding=1) # 1/64

        self.conv_c7 = nn.Conv2d(int(576/scale), int(384/scale),kernel_size=3, padding=1) # 1/32
        self.deconv_c7 = nn.ConvTranspose2d(int(384/scale), int(128/scale),kernel_size=4,stride=2,padding=1) # 1/16
        self.conv_c8 = nn.Conv2d(int(384/scale), int(256/scale),kernel_size=3, padding=1) # 1/16
        self.deconv_c8 = nn.ConvTranspose2d(int(256/scale), int(96/scale),kernel_size=4,stride=2,padding=1) # 1/8
        self.conv_c9 = nn.Conv2d(int(288/scale), int(192/scale),kernel_size=3, padding=1) # 1/8
        self.deconv_c9 = nn.ConvTranspose2d(int(192/scale), int(64/scale),kernel_size=4,stride=2,padding=1) # 1/4
        self.conv_c10 = nn.Conv2d(int(192/scale), int(96/scale),kernel_size=3, padding=1) # 1/4
        self.deconv_c10 = nn.ConvTranspose2d(int(96/scale), int(32/scale),kernel_size=4,stride=2,padding=1) # 1/2
        self.conv_c11 = nn.Conv2d(int(96/scale), int(64/scale),kernel_size=3, padding=1) # 1/2
        self.deconv_c11 = nn.ConvTranspose2d(int(64/scale), int(48/scale),kernel_size=4,stride=2,padding=1) # 1/1
        # self.conv_c12 = nn.Conv2d(48, 16,kernel_size=1,padding=0)
        # self.conv_c13 = nn.Conv2d(16, 1, kernel_size=1,padding=0)

        self.conv_c6_2 = nn.Conv2d(int(512/scale), int(384/scale), kernel_size=3, padding=1)
        self.deconv_c7_2 = nn.ConvTranspose2d(int(384/scale), int(192/scale),kernel_size=4,stride=2,padding=1) # 1/32

        self.predout1 = predict_layer(int(48/scale), 16, 2)


    def forward(self, x, combinelr=False):
        if combinelr:
            assert x.shape[1]%2 == 0
            x1 = x.reshape(x.shape[0]*2, int(x.shape[1]/2), x.shape[2], x.shape[3])
            x1 = self.feature_extraction(x1)
            x1 = x1.view(int(x1.shape[0]/2), x1.shape[1]*2, x1.shape[2], x1.shape[3])
            x2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
            x = torch.cat((x1,x2),dim=1)
        else:
            left = self.feature_extraction(x[0])
            right = self.feature_extraction(x[1])
            x2 = F.interpolate(x[0], scale_factor=0.5, mode='bilinear')
            x3 = F.interpolate(x[1], scale_factor=0.5, mode='bilinear')
            x = torch.cat((left,right,x2,x3),dim=1)

        # depth regression layers
        x = self.conv_c0(x) # 1/2
        cat0 = self.conv_c1(x) # 1/2 - 64
        x = self.conv_c2(cat0) # 1/2
        cat1 = F.max_pool2d(x, kernel_size=2) # 1/4 - 128
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

        x = self.deconv_c7_2(x) # 1/32 - 192
        out6 = self.actfun(x, inplace=True)
        x = torch.cat((out6,cat4),dim=1) #  - 576
        x = self.conv_c7(x) # 1/32 - 384
        x = self.actfun(x, inplace=True)
        x = self.deconv_c7(x) # 1/16 - 128
        out5 = self.actfun(x, inplace=True)
        x = torch.cat((out5,cat3),dim=1) # - 384
        x = self.conv_c8(x) # 1/16 - 256
        x = self.actfun(x, inplace=True)
        x = self.deconv_c8(x) # 1/8 - 96 
        out4 = self.actfun(x, inplace=True)
        x = torch.cat((out4,cat2),dim=1) # - 288
        x = self.conv_c9(x) # 1/8 - 192
        x = self.actfun(x, inplace=True)
        x = self.deconv_c9(x) # 1/4 - 64
        out3 = self.actfun(x, inplace=True)
        x = torch.cat((out3,cat1),dim=1) # - 192
        x = self.conv_c10(x) # 1/4 - 96
        x = self.actfun(x, inplace=True)
        x = self.deconv_c10(x) # 1/2 - 32
        out2 = self.actfun(x, inplace=True)
        x = torch.cat((out2,cat0),dim=1) # - 96
        x = self.conv_c11(x) # 1/2 - 64
        x = self.actfun(x, inplace=True)
        x = self.deconv_c11(x) # 1/1 - 48
        out1 = self.actfun(x, inplace=True)
        out1 = self.predout1(out1) # 1/1

        return out1

    def get_loss(self, output, target, criterion, mask=None, small_scale=False):
        if mask is None:
            return criterion(output, targetflow)
        else:
            valid_mask = np.logical_or(mask<0.1, mask>1) #mask<128
            valid_mask = valid_mask.expand(target.shape)
            return criterion(output[valid_mask], target[valid_mask])

if __name__ == '__main__':
    
    stereonet = FlowNet2()
    stereonet.cuda()
    # print (stereonet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    np.set_printoptions(precision=4, threshold=100000)
    x, y = np.ogrid[:256, :256]
    # print (x, y, (x+y))
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print (img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    # imgInput = np.concatenate((imgInput,imgInput),axis=0)
    imgInput = np.concatenate((imgInput,imgInput),axis=1)

    starttime = time.time()
    for k in range(10):
        imgTensor = torch.from_numpy(imgInput)
        z = stereonet(imgTensor.cuda() ,combinelr=True)
        print (z.data.cpu().numpy().shape)
    print (time.time() - starttime)
