import torch 
import torch.nn as nn
import torch.nn.functional as F
import math

def conv(in_planes, out_planes, kernel_size=3, stride=2, padding=1, dilation=1, bn_layer=False, bias=True):
    if bn_layer:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True)
        )
    else: 
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation),
            nn.ReLU(inplace=True)
        )

def linear(in_planes, out_planes):
    return nn.Sequential(
        nn.Linear(in_planes, out_planes), 
        nn.ReLU(inplace=True)
        )

def feature_extract(bn_layer, intrinsic):

    if intrinsic:
        inputnum = 4
    else:
        inputnum = 2
    conv1 = conv(inputnum, 16, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 320 x 240
    conv2 = conv(16,32, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 160 x 120
    conv3 = conv(32,64, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 80 x 60
    conv4 = conv(64,128, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 40 x 30
    conv5 = conv(128,256, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 20 x 15

    conv6 = conv(256,512, kernel_size=5, stride=5, padding=0, bn_layer=bn_layer) # 4 x 3
    conv7 = conv(512,1024, kernel_size=(3, 4), stride=1, padding=0, bn_layer=bn_layer) # 1 x 1

    return nn.Sequential(conv1, conv2, conv3, conv4,
                                conv5, conv6, conv7,  
        )


class VOFlowNet(nn.Module):
    """
    Input flow, output VO

    """

    def __init__(self, bn_layer=False, intrinsic=False):
        super(VOFlowNet, self).__init__()
        fc1 = linear(1024,256)
        fc2 = linear(256,32)
        fc3 = nn.Linear(32,6)

        self.voflow1 = feature_extract(bn_layer=bn_layer, intrinsic=intrinsic)
        self.voflow2 = nn.Sequential(fc1, fc2, fc3)
        print('Initialize VOFlowNet...')

    def forward(self, x):
        x = self.voflow1(x)
        x = x.view(x.shape[0], -1)
        return self.voflow2(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

class VOFlowNetTwoHeads(nn.Module):
    """
    Input flow, output VO

    """

    def __init__(self, bn_layer=False, intrinsic=False):
        super(VOFlowNetTwoHeads, self).__init__()

        fc1_trans = linear(1024,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(1024,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow1 = feature_extract(bn_layer=bn_layer, intrinsic=intrinsic)
        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        print('Initialize VOFlowNetTwoHeads...')

    def forward(self, x):
        x = self.voflow1(x)
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

class VOFlowNetTwoHeads2(nn.Module):
    """
    Input flow, output VO

    """

    def __init__(self, bn_layer=False, intrinsic=False):
        super(VOFlowNetTwoHeads2, self).__init__()
        self.fc1 = linear(1024,256)
        fc2_trans = linear(256,20)
        fc3_trans = nn.Linear(20,3)

        fc2_rot = linear(256,20)
        fc3_rot = nn.Linear(20,3)

        self.voflow1 = feature_extract(bn_layer=bn_layer, intrinsic=intrinsic)
        self.voflow_trans = nn.Sequential(fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc2_rot, fc3_rot)
        print('Initialize VOFlowNetTwoHeads2...')

    def forward(self, x):
        x = self.voflow1(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

class VOFlowFCN(nn.Module):
    """
    Fully Convolutional Net
    Input flow, output VO

    """

    def __init__(self, bn_layer=False, intrinsic=False, spatial_pooling=False):
        super(VOFlowFCN, self).__init__()
        self.spatial_pooling = spatial_pooling
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        conv1 = conv(inputnum, 16, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 320 x 240
        conv2 = conv(16,32, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 160 x 120
        conv3 = conv(32,64, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 80 x 60

        conv4 = conv(64,128, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 40 x 30
        conv5 = conv(128,256, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 20 x 15

        conv6 = conv(256,512, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 10 x 8
        conv7 = conv(512,1024, kernel_size=3, stride=2, padding=1, bn_layer=bn_layer) # 5 x 4

        self.voflow1 = nn.Sequential(conv1, conv2, conv3, conv4,
                                    conv5, conv6, conv7,  
        )

        fc1_trans = linear(1024,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(1024,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)

        if self.spatial_pooling:
            self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                         conv(64, 17, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                         conv(64, 22, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                         conv(64, 32, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch4 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2,2)),
                                         conv(64, 57, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            # after the last conv 
            self.branch5 = nn.Sequential(nn.AdaptiveAvgPool2d(4),
                                         conv(512, 32, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch6 = nn.Sequential(nn.AdaptiveAvgPool2d(2),
                                         conv(512, 64, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch7 = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                         conv(512, 256, 1, 1, 0),
                                         nn.ReLU(inplace=False))

            self.voflow1 = nn.Sequential(conv1, conv2, conv3)  
            self.voflow2 = nn.Sequential(conv5, conv6)  

        print('Initialize VOFlowFCN...')

    def forward(self, x):
        batch = x.shape[0]
        if self.spatial_pooling:
            x = self.voflow1(x)
            b0 = self.branch4(x)
            size_out = (b0.shape[2], b0.shape[3])
            b1 = self.branch1(x)
            b1 = F.interpolate(b1, size_out, mode='bilinear', align_corners=True)
            b2 = self.branch2(x)
            b2 = F.interpolate(b2, size_out, mode='bilinear', align_corners=True)
            b3 = self.branch3(x)
            b3 = F.interpolate(b3, size_out, mode='bilinear', align_corners=True)
            x = torch.cat((b0, b1, b2, b3), 1)
            x = self.voflow2(x)
            b0 = self.branch5(x).view(batch, -1)
            b1 = self.branch6(x).view(batch, -1)
            b2 = self.branch7(x).view(batch, -1)
            x = torch.cat((b0, b1, b2), dim=1)
        else:
            x = self.voflow1(x)
            x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
            x = x.view(batch, -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

class VOFlowFCN2(nn.Module):
    """
    Fully Convolutional Net
    Input flow, output VO
    No padding

    """

    def __init__(self, bn_layer=False, intrinsic=False):
        super(VOFlowFCN2, self).__init__()
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        conv1 = conv(inputnum, 64, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 319 x 239
        conv2 = conv(64,128, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 159 x 119
        conv3 = conv(128,256, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 79 x 59
        conv4 = conv(256,256, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 39 x 29
        conv5 = conv(256,512, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 19 x 14

        conv6 = conv(512,512, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 9 x 6
        conv7 = conv(512,1024, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 4 x 2

        self.voflow1 = nn.Sequential(conv1, conv2, conv3, conv4,
                                    conv5, conv6, conv7,  
        )

        fc1_trans = linear(1024,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(1024,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        print('Initialize VOFlowFCN2...')

    def forward(self, x):
        x = self.voflow1(x)
        x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

class VOFlowFCN3(nn.Module):
    """
    Fully Convolutional Net
    Input flow, output VO
    Dialated

    """

    def __init__(self, bn_layer=False, intrinsic=False, dilation=False):
        super(VOFlowFCN3, self).__init__()
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        if dilation:
            dialist = [1,2,2,4,4]
        else:
            dialist = [1,] * 5   

        conv1 = conv(inputnum, 64, kernel_size=3, stride=1, padding=0, dilation=dialist[0], bn_layer=bn_layer) # 319 x 239
        conv1_2 = conv(64, 64, kernel_size=3, stride=2, padding=0, dilation=1, bn_layer=bn_layer) # 319 x 239
        conv2 = conv(64,64, kernel_size=3, stride=1, padding=0, dilation=dialist[1], bn_layer=bn_layer) # 159 x 119
        conv2_2 = conv(64,128, kernel_size=3, stride=2, padding=0, dilation=1, bn_layer=bn_layer) # 159 x 119
        conv3 = conv(128,128, kernel_size=3, stride=1, padding=0, dilation=dialist[2], bn_layer=bn_layer) # 79 x 59
        conv3_2 = conv(128,256, kernel_size=3, stride=2, padding=0, dilation=1, bn_layer=bn_layer) # 79 x 59
        conv4 = conv(256,256, kernel_size=3, stride=1, padding=0, dilation=dialist[3], bn_layer=bn_layer) # 39 x 29
        conv4_2 = conv(256,256, kernel_size=3, stride=2, padding=0, dilation=1, bn_layer=bn_layer) # 39 x 29
        conv5 = conv(256,256, kernel_size=3, stride=1, padding=0, dilation=dialist[4], bn_layer=bn_layer) # 19 x 14
        conv5_2 = conv(256,512, kernel_size=3, stride=2, padding=0, dilation=1, bn_layer=bn_layer) # 19 x 14

        conv6 = conv(512,512, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 9 x 6
        conv7 = conv(512,1024, kernel_size=3, stride=2, padding=0, bn_layer=bn_layer) # 4 x 2

        self.voflow1 = nn.Sequential(conv1, conv1_2, conv2, conv2_2, conv3, conv3_2, conv4, conv4_2,
                                    conv5, conv5_2, conv6, conv7, 
        )

        fc1_trans = linear(1024,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(1024,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        print('Initialize VOFlowFCN3...')

    def forward(self, x):
        x = self.voflow1(x)
        x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)

class VOFlowFCN4(nn.Module):
    """
    Fully Convolutional Net
    Input flow, output VO

    """

    def __init__(self, bn_layer=False, intrinsic=False, spatial_pooling=False):
        super(VOFlowFCN4, self).__init__()
        self.spatial_pooling = spatial_pooling
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        conv1_1 = conv(inputnum, 16, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 640 x 480
        conv1_2 = conv(16, 32, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 320 x 240
        conv2_1 = conv(32,32, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 320 x 240
        conv2_2 = conv(32,32, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 320 x 240
        conv2_3 = conv(32,64, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 160 x 120
        conv3_1 = conv(64,64, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 160 x 120
        conv3_2 = conv(64,64, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 160 x 120
        conv3_3 = conv(64,128, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 80 x 60

        if spatial_pooling:
            self.branch1 = nn.Sequential(nn.AvgPool2d((32, 32), stride=(32,32)),
                                         conv(64, 17, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch2 = nn.Sequential(nn.AvgPool2d((16, 16), stride=(16,16)),
                                         conv(64, 22, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch3 = nn.Sequential(nn.AvgPool2d((8, 8), stride=(8,8)),
                                         conv(64, 32, 1, 1, 0),
                                         nn.ReLU(inplace=False))
            self.branch4 = nn.Sequential(nn.AvgPool2d((2, 2), stride=(2,2)),
                                         conv(64, 57, 1, 1, 0),
                                         nn.ReLU(inplace=False))

        conv4_1 = conv(128,128, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 80 x 60
        conv4_2 = conv(128,128, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 80 x 60
        conv4_3 = conv(128,256, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 40 x 30

        conv5_1 = conv(256,256, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 40 x 30
        conv5_2 = conv(256,256, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 40 x 30
        conv5_3 = conv(256,384, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 20 x 15

        conv6_1 = conv(384,384, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 20 x 15
        conv6_2 = conv(384,384, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 20 x 15
        conv6_3 = conv(384,512, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 10 x 7

        conv7_1 = conv(512,512, kernel_size=3, stride=1, padding=1, bn_layer=bn_layer) # 10 x 7
        conv7_2 = conv(512,768, kernel_size=4, stride=2, padding=1, bn_layer=bn_layer) # 5 x 3

        if self.spatial_pooling:
            self.voflow1 = nn.Sequential(conv1_1, conv1_2, 
                                        conv2_1, conv2_2, conv2_3, 
                                        conv3_1, conv3_2,
            )
        else:
            self.voflow1 = nn.Sequential(conv1_1, conv1_2, 
                                        conv2_1, conv2_2, conv2_3, 
                                        conv3_1, conv3_2, conv3_3, 
            )

        self.voflow2 = nn.Sequential(conv4_1, conv4_2, conv4_3,
                                    conv5_1, conv5_2, conv5_3, 
                                    conv6_1, conv6_2, conv6_3, 
                                    conv7_1, conv7_2,   
        )

        fc1_trans = linear(768,128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(768,128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)

        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)
        print('Initialize VOFlowFCN4...')

    def forward(self, x):
        x = self.voflow1(x)
        if self.spatial_pooling:
            b0 = self.branch4(x)
            size_out = (b0.shape[2], b0.shape[3])
            b1 = self.branch1(x)
            b1 = F.interpolate(b1, size_out, mode='bilinear', align_corners=True)
            b2 = self.branch2(x)
            b2 = F.interpolate(b2, size_out, mode='bilinear', align_corners=True)
            b3 = self.branch3(x)
            b3 = F.interpolate(b3, size_out, mode='bilinear', align_corners=True)
            x = torch.cat((b0, b1, b2, b3), 1)
        x = self.voflow2(x)
        x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)
        return torch.cat((x_trans, x_rot), dim=1)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = conv(inplanes, planes, 3, stride, pad, dilation)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return F.relu(out, inplace=True)

class VOFlowRes(nn.Module):
    def __init__(self, intrinsic=False, down_scale=False, config=0, stereo=False, autoDistTarget=0., uncertainty=0):
        super(VOFlowRes, self).__init__()
        if intrinsic:
            inputnum = 4
        else:
            inputnum = 2
        if stereo:
            inputnum += 1
        inputnum += uncertainty # mono-uncertainty: +1, stereo-uncertainty: +2
        
        self.down_scale = down_scale
        self.config = config
        self.stereo = stereo
        self.autoDistTarget = autoDistTarget # scale the distance wrt the mean value 

        if config==0:
            blocknums = [2,2,3,3,3,3,3]
            outputnums = [32,64,64,64,128,128,128]
        elif config==1:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif config==2:
            blocknums = [2,2,3,4,6,7,3]
            outputnums = [32,64,64,128,128,256,256]
        elif config==3:
            blocknums = [3,4,7,9,9,5,3]
            outputnums = [32,64,128,128,256,256,512]


        self.firstconv = nn.Sequential(conv(inputnum, 32, 3, 2, 1, 1, False),
                                       conv(32, 32, 3, 1, 1, 1),
                                       conv(32, 32, 3, 1, 1, 1))

        self.inplanes = 32
        if not down_scale:
            self.layer0 = self._make_layer(BasicBlock, outputnums[0], blocknums[0], 2, 1, 1) # (160 x 112)
            self.layer0_2 = self._make_layer(BasicBlock, outputnums[1], blocknums[1], 2, 1, 1) # (80 x 56)

        self.layer1 = self._make_layer(BasicBlock, outputnums[2], blocknums[2], 2, 1, 1) # 40 x 28
        self.layer2 = self._make_layer(BasicBlock, outputnums[3], blocknums[3], 2, 1, 1) # 20 x 14
        self.layer3 = self._make_layer(BasicBlock, outputnums[4], blocknums[4], 2, 1, 1) # 10 x 7
        self.layer4 = self._make_layer(BasicBlock, outputnums[5], blocknums[5], 2, 1, 1) # 5 x 4
        self.layer5 = self._make_layer(BasicBlock, outputnums[6], blocknums[6], 2, 1, 1) # 3 x 2
        fcnum = outputnums[6] * 6
        if config==2:
            self.layer6 = conv(outputnums[6],outputnums[6]*2, kernel_size=(2, 3), stride=1, padding=0) # 1 x 1
            fcnum = outputnums[6]*2
        if config==3:
            fcnum = outputnums[6]


        fc1_trans = linear(fcnum, 128)
        fc2_trans = linear(128,32)
        fc3_trans = nn.Linear(32,3)

        fc1_rot = linear(fcnum, 128)
        fc2_rot = linear(128,32)
        fc3_rot = nn.Linear(32,3)


        self.voflow_trans = nn.Sequential(fc1_trans, fc2_trans, fc3_trans)
        self.voflow_rot = nn.Sequential(fc1_rot, fc2_rot, fc3_rot)


    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
           downsample = nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, pad, dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,1,None,pad,dilation))

        return nn.Sequential(*layers)

    def forward(self, x, scale_disp=1.0):
        # import ipdb;ipdb.set_trace()
        if self.stereo:
            if self.autoDistTarget > 0:
                distTarget = 1.0/(self.autoDistTarget * 0.25) # normalize the target by 0.25 -- hard code
                depth_mean = torch.mean(x[:,2,:,:], (1,2))
                scale_disp = distTarget / depth_mean
                print(scale_disp)
                x[:,2,:,:] = x[:,2,:,:] * scale_disp.view(scale_disp.shape+(1,1)) # tensor: (n, 1, 1)
            else:
                x[:,2,:,:] = x[:,2,:,:] * scale_disp

        x = self.firstconv(x)
        if not self.down_scale:
            x  = self.layer0(x)
            x  = self.layer0_2(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        if self.config==2:
            x = self.layer6(x)
        if self.config==3:
            x = F.avg_pool2d(x, kernel_size = x.shape[-2:])
        
        x = x.view(x.shape[0], -1)
        x_trans = self.voflow_trans(x)
        x_rot = self.voflow_rot(x)

        if self.stereo:
            if self.autoDistTarget > 0:
                x_trans = x_trans * scale_disp.view(scale_disp.shape+(1,))
            else:
                x_trans = x_trans * scale_disp

        return torch.cat((x_trans, x_rot), dim=1)

if __name__ == '__main__':
    
    voflownet = VOFlowRes(down_scale=True, config=1) # 
    voflownet.cuda()
    print (voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    np.set_printoptions(precision=4, threshold=10000)
    imsize1 = 112
    imsize2 = 160
    x, y = np.ogrid[:imsize1, :imsize2]
    # print x, y, (x+y)
    img = np.repeat((x + y)[..., np.newaxis], 2, 2) / float(imsize1 + imsize2)
    img = img.astype(np.float32)
    print(img.dtype, img.shape)
    # print(img)

    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    imgTensor = torch.from_numpy(imgInput)
    print(imgTensor.shape)
    z = voflownet(imgTensor.cuda())
    print(z.data.shape)
    print(z.data.cpu().numpy())

    # for name,param in voflownet.named_parameters():
    #   print name,param.requires_grad

