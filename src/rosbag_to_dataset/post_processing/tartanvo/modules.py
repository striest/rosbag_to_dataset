import torch 
import torch.nn as nn
import torch.nn.functional as F

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride = 1, bn = False, relu = True):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=True)
        self.relu = nn.ReLU() if relu else None
        self.bn = nn.BatchNorm2d(out_dim) if bn else None

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Residual(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Residual, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = Conv(inp_dim, int(out_dim/2), 1, relu=False)
        self.conv2 = Conv(int(out_dim/2), int(out_dim/2), 3, relu=False)
        self.conv3 = Conv(int(out_dim/2), out_dim, 1, relu=False)
        self.skip_layer = Conv(inp_dim, out_dim, 1, relu=False)
        if inp_dim == out_dim:
            self.need_skip = False
        else:
            self.need_skip = True
        
    def forward(self, x):
        if self.need_skip:
            residual = self.skip_layer(x)
        else:
            residual = x
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out += residual
        return out 

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv(inplanes, planes, 3, stride)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, padding=1, bias=True) #Conv(planes, planes, 3, 1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

class FeatureExtraction(nn.Module):
    def __init__(self,last_planes=32,bigger=False,middleblock=16):
        super(FeatureExtraction, self).__init__()
        self.inplanes = 32
        self.bigger = bigger # output 1/2 size instead of 1/4 size
        if bigger:
          extra_channel = 32
        else:
          extra_channel = 0
        self.firstconv = nn.Sequential(Conv(3, 32, 3, 2),
                                       Conv(32, 32, 3, 1),
                                       Conv(32, 32, 3, 1))

        self.layer1 = self._make_layer(BasicBlock, 32, 3, 1,1,1)
        self.layer2 = self._make_layer(BasicBlock, 64, middleblock, 2,1,1) 
        self.layer3 = self._make_layer(BasicBlock, 128, 3, 1,1,1)
        self.layer4 = self._make_layer(BasicBlock, 128, 3, 1,1,1)

        self.branch1 = nn.Sequential(nn.AvgPool2d((64, 64), stride=(64,64)),
                                     Conv(128, 32, 1, 1))

        self.branch2 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                     Conv(128, 32, 1, 1))

        self.branch3 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                     Conv(128, 32, 1, 1))

        self.branch4 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                     Conv(128, 32, 1, 1))

        self.lastconv = nn.Sequential(Conv(320+extra_channel, 128, 3, 1),
                                      nn.Conv2d(128, last_planes, kernel_size=1, padding=0, stride = 1, bias=False))

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None))

        return nn.Sequential(*layers)

    def forward(self, x):
        output      = self.firstconv(x)
        output_0    = self.layer1(output)
        output_raw  = self.layer2(output_0)
        output      = self.layer3(output_raw)
        output_skip = self.layer4(output)

        target_h, target_w = output_skip.size()[2],output_skip.size()[3]

        output_branch1 = self.branch1(output_skip)
        output_branch1 = F.interpolate(output_branch1, [target_h, target_w], mode='bilinear', align_corners=True)
        # upsample(output_branch1, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch2 = self.branch2(output_skip)
        output_branch2 = F.interpolate(output_branch2, [target_h, target_w], mode='bilinear', align_corners=True)
        # upsample(output_branch2, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch3 = self.branch3(output_skip)
        output_branch3 = F.interpolate(output_branch3, [target_h, target_w], mode='bilinear', align_corners=True)
        # upsample(output_branch3, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_branch4 = self.branch4(output_skip)
        output_branch4 = F.interpolate(output_branch4, [target_h, target_w], mode='bilinear', align_corners=True)
        # upsample(output_branch4, (output_skip.size()[2],output_skip.size()[3]),mode='bilinear')

        output_feature = torch.cat((output_raw, output_skip, output_branch4, output_branch3, output_branch2, output_branch1), 1)

        if self.bigger:
          output_feature = torch.cat((F.interpolate(output_feature, [target_h*2, target_w*2], mode='bilinear', align_corners=True),output_0), 1)

        output_feature = self.lastconv(output_feature)

        return output_feature



class Hourglass(nn.Module):
    def __init__(self, n, f, increase=0):
        super(Hourglass, self).__init__()
        nf = f + increase
        self.up1 = Residual(f, nf)
        # Lower branch
        self.pool1 = nn.MaxPool2d(2, 2)
        # self.low1 = Residual(f, nf)
        self.n = n
        # Recursive hourglass
        if self.n > 1:
            self.low2 = Hourglass(n-1, nf, increase=0)
        else:
            self.low2 = Residual(nf, nf)
        self.low3 = Residual(nf, nf)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        up1  = self.up1(x)
        pool1 = self.pool1(up1)
        # low1 = self.low1(pool1)
        low2 = self.low2(pool1)
        low3 = self.low3(low2)
        up2  = self.up2(low3)
        return up1 + up2

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
        output_branch1 = F.interpolate(output_branch1, [target_h, target_w], mode='bilinear', align_corners=True)

        output_branch2 = self.branch2(x)
        output_branch2 = F.interpolate(output_branch2, [target_h, target_w], mode='bilinear', align_corners=True)

        output_branch3 = self.branch3(x)
        output_branch3 = F.interpolate(output_branch3, [target_h, target_w], mode='bilinear', align_corners=True)

        output_branch4 = self.branch4(x)
        output_branch4 = F.interpolate(output_branch4, [target_h, target_w], mode='bilinear', align_corners=True)

        output_feature = torch.cat((x, output_branch4, output_branch3, output_branch2, output_branch1), 1)
        return output_feature

class HourglassEncoder(nn.Module):
    def __init__(self, in_channels=134, act_fun='relu'):
        super(HourglassEncoder, self).__init__()
        # depth regression layers
        if act_fun == 'relu':
            self.actfun = F.relu
        elif act_fun == 'selu':
            self.actfun = F.selu
        else:
            print ('Unknown activate function', act_fun)
            self.actfun = F.relu

        self.conv_c0 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1) # 1/2
        self.conv_c1 = Hourglass(4, 64, 0) # 1/2
        self.conv_c2 = Hourglass(3, 64, 0) # 1/4 #nn.Conv2d(128,128, kernel_size=3, padding=1)
        self.conv_c2_SSP = SSP(64) # 1/4
        self.conv_c3 = Hourglass(3, 128, 64) # 1/8 #nn.Conv2d(128,256, kernel_size=3, padding=1)
        self.conv_c4 = Hourglass(2, 192, 64) # 1/16 #nn.Conv2d(256,256, kernel_size=3, padding=1)
        self.conv_c5 = nn.Conv2d(256, 384, kernel_size=3, padding=1) # 1/32
        self.conv_c6 = nn.Conv2d(384, 512, kernel_size=3, padding=1) # 1/64
        self.conv_c6_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

    def forward(self, x ):
        # import ipdb;ipdb.set_trace()        
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
    
        return x, cat0, cat1, cat2, cat3, cat4

class HourglassDecoder(nn.Module):
    def __init__(self, out_channels=1, act_fun='relu', hourglass=True):
        super(HourglassDecoder, self).__init__()
        if act_fun == 'relu':
            self.actfun = F.relu
        elif act_fun == 'selu':
            self.actfun = F.selu
        else:
            print ('Unknown activate function', act_fun)
            self.actfun = F.relu

        self.deconv_c7_2 = nn.ConvTranspose2d(512, 512,kernel_size=4,stride=2,padding=1) # 1/32
        self.deconv_c7 = nn.ConvTranspose2d(896, 320,kernel_size=4,stride=2,padding=1) # 1/16
        self.deconv_c8 = nn.ConvTranspose2d(576, 192,kernel_size=4,stride=2,padding=1) # 1/8
        self.deconv_c9 = nn.ConvTranspose2d(384, 128,kernel_size=4,stride=2,padding=1) # 1/4
        self.deconv_c10 = nn.ConvTranspose2d(256, 64,kernel_size=4,stride=2,padding=1) # 1/2
        self.deconv_c11 = nn.ConvTranspose2d(128, 64,kernel_size=4,stride=2,padding=1) # 1/1
        self.conv_c12 = nn.Conv2d(64, 16,kernel_size=1,padding=0)
        self.conv_c13 = nn.Conv2d(16, out_channels, kernel_size=1,padding=0)

        if hourglass:
            self.conv_c8 = Hourglass(2, 192, 0) # 1/8
            self.conv_c9 = Hourglass(3, 128, 0) # 1/4
            self.conv_c10 = Hourglass(4, 64, 0) # 1/2
        self.hourglass = hourglass

    def forward(self, x, cat0, cat1, cat2, cat3, cat4 ):

        x = self.deconv_c7_2(x) # 1/32 - 512
        x = self.actfun(x, inplace=True)
        x = torch.cat((x,cat4),dim=1) #  - 896
        x = self.deconv_c7(x) # 1/16 - 320
        x = self.actfun(x, inplace=True)
        x = torch.cat((x,cat3),dim=1) # - 576
        x = self.deconv_c8(x) # 1/8 - 192 
        x = self.actfun(x, inplace=True)
        if self.hourglass:
            x = self.conv_c8(x)
        x = torch.cat((x,cat2),dim=1) # - 384
        x = self.deconv_c9(x) # 1/4 - 128
        x = self.actfun(x, inplace=True)
        if self.hourglass:
            x = self.conv_c9(x)
        x = torch.cat((x,cat1),dim=1) # - 256
        x = self.deconv_c10(x) # 1/2 - 64
        x = self.actfun(x, inplace=True)
        if self.hourglass:
            x = self.conv_c10(x)
        x = torch.cat((x,cat0),dim=1) # - 128
        x = self.deconv_c11(x) # 1/1 - 64
        x = self.actfun(x, inplace=True)

        x = self.conv_c12(x)
        x = self.actfun(x, inplace=True)
        out0 = self.conv_c13(x)
        # x = F.relu(x, inplace=True)
        return out0


