
import torch 
import torch.nn as nn
import torch.nn.functional as F

class StereoVONet(nn.Module):
    def __init__(self, network=0, intrinsic=True, flowNormFactor=1.0, stereoNormFactor=1.0, poseDepthNormFactor=0.25, down_scale=True, 
                        config=1, fixflow=True, fixstereo=True, autoDistTarget=0.,blxfx=80.):
        '''
        flowNormFactor: difference between flownet and posenet
        stereoNormFactor: norm value used in stereo training
        poseDepthNormFactor: distance is normalized in posenet training ()
        autoDistTarget: 0.  : no auto scale
                        > 0.: scale the distance wrt the mean value 
        '''
        super(StereoVONet, self).__init__()

        if network==0: # PSM + PWC
            from .PWC import PWCDCNet as FlowNet
            from .PSM import stackhourglass as StereoNet
            self.flowNet   = FlowNet()
        if network==1: # 5_5 + PWC
            from .PWC import PWCDCNet as FlowNet
            from .StereoNet7 import StereoNet7 as StereoNet
            self.flowNet   = FlowNet()
        else:
            from .FlowNet2 import FlowNet2 as FlowNet
            from .StereoNet7 import StereoNet7 as StereoNet
            self.flowNet   = FlowNet(middleblock=3)

        self.stereoNet = StereoNet()

        from .VOFlowNet import VOFlowRes as FlowPoseNet
        self.flowPoseNet = FlowPoseNet(intrinsic=intrinsic, down_scale=down_scale, config=config, stereo=True, autoDistTarget=autoDistTarget)

        self.network = network
        self.intrinsic = intrinsic
        self.flowNormFactor = flowNormFactor
        self.stereoNormFactor = stereoNormFactor
        self.poseDepthNormFactor = poseDepthNormFactor
        self.down_scale = down_scale
        self.blxfx = blxfx

        if fixflow:
            for param in self.flowNet.parameters():
                param.requires_grad = False

        if fixstereo:
            for param in self.stereoNet.parameters():
                param.requires_grad = False

        self.autoDistTarget = autoDistTarget

    def forward(self, x): 
                # x0_flow, x0n_flow, x0_stereo, x1_stereo, intrin=None, 
                # scale_w=None, only_flow=False, only_stereo=False, gt_flow=None, gt_disp=None, 
                # scale_disp=1.0, blxfx=80.):
        '''
        flow_out: pwcnet: 5 scale predictions up to 1/4 size
                  flownet: 1/1 size prediction
        stereo_out: psmnet: 3 scale predictions up to 1/1 size
                    stereonet: 1/1 size prediction 
        scale_w: the x-direction scale factor in data augmentation
        scale_depth: scale input depth and output motion to shift the data distribution
        '''
        # import ipdb;ipdb.set_trace()

        x0_flow = x[:,:3,:,:]
        x0n_flow = x[:,3:6,:,:]
        x0_stereo = x[:,6:9,:,:]
        x1_stereo = x[:,9:12,:,:]
        intrin = x[:,12:,:,:]
        intrin = F.interpolate(intrin, size=(112,160), mode='bilinear', align_corners=True) # for input wo downsampleflow
        # scale the flow size
        x0_flow = F.interpolate(x0_flow, size=(256,256), mode='bilinear', align_corners=True)
        x0n_flow = F.interpolate(x0n_flow, size=(256,256), mode='bilinear', align_corners=True)
        flow_out = self.flowNet([x0_flow, x0n_flow])
        if self.network==0 or self.network==1: # pwcnet, 1/4 size
            if self.down_scale:
                flow_out = F.interpolate(flow_out[0], size=(112,160), mode='bilinear', align_corners=True) 
                flow_out[:,0,:,:] = flow_out[:,0,:,:] * 2.5 # 640/256
                flow_out[:,1,:,:] = flow_out[:,1,:,:] * 1.75 # 448/256
                flow = flow_out
            else: # TODO: use pwcnet with down_scale=False will have problems
                flow = F.interpolate(flow_out[0], scale_factor=4, mode='bilinear', align_corners=True) # TODO: this is not used, flow_out is still in smaller size
        else: # flownet, 1/1 size
            if self.down_scale:
                flow_out = F.interpolate(flow_out, size=(112,160), mode='bilinear', align_corners=True)
                flow_out[:,0,:,:] = flow_out[:,0,:,:] * 2.5 # 640/256
                flow_out[:,1,:,:] = flow_out[:,1,:,:] * 1.75 # 448/256
            flow = flow_out

        # downscale the stereo input 
        x0_stereo = F.interpolate(x0_stereo, size=(256,256), mode='bilinear', align_corners=True)
        x1_stereo = F.interpolate(x1_stereo, size=(256,256), mode='bilinear', align_corners=True)
        stereo_out = self.stereoNet(torch.cat((x0_stereo, x1_stereo),dim=1))

        # scale the disparity size
        if self.network == 0: # psmnet, 3 outputs # deprecated since manually downscaled the input size
            if self.down_scale: 
                stereo_out[0] = F.interpolate(stereo_out[0], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo_out[1] = F.interpolate(stereo_out[1], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo_out[2] = F.interpolate(stereo_out[2], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo = stereo_out[2]
            else:
                stereo = stereo_out[2]
        else:
            if self.down_scale:
                stereo_out = F.interpolate(stereo_out, size=(112,160), mode='bilinear', align_corners=True)
                stereo_out = stereo_out * 2.5 # compensate the resize - 2.5 = 640/256
            stereo = stereo_out

        flow_input    = flow * self.flowNormFactor
        # disp_input    = stereo / self.stereoNormFactor # disparity network norm value, 0.02 or 0.05
        # depth_input    = blxfx / disp_input # from disparity to distance
        # depth_input    = depth_input * self.poseDepthNormFactor # from disparity to distance
        # depth_input    = 1 / depth_input
        # depth_input = 1 / ( (blxfx / (stereo / self.stereoNormFactor)) * self.poseDepthNormFactor)
        depth_input = stereo / self.blxfx / (self.stereoNormFactor * self.poseDepthNormFactor )

        # scale the disp back, because we treat it as depth
        # depth_input = depth_input / scale_w
        # scale the disp for better motion distribution
        # import ipdb;ipdb.set_trace()
        
        # if self.autoDistTarget == 0:
        #     depth_input = depth_input * scale_disp
        # else: 
        #     distTarget = 1.0/(self.autoDistTarget * self.poseDepthNormFactor) # normalize the target by 0.25
        #     depth_mean = torch.mean(depth_input, (1,2,3))
        #     scale_disp = distTarget / depth_mean
        #     depth_input = depth_input * scale_disp.view(scale_disp.shape+(1,1))
        #     print (scale_disp)


        if self.intrinsic:
            inputTensor = torch.cat( ( flow_input, depth_input, intrin ), dim=1 )
        else:
            inputTensor = torch.cat( ( flow_input, depth_input ), dim=1 )
        
        pose = self.flowPoseNet( inputTensor, scale_disp=1.0 )
        # scale the translation back
        # if self.autoDistTarget == 0:
        #     pose[:, :3] = pose[:, :3] * scale_disp
        # else:
        #     pose[:, :3] = pose[:, :3] * scale_disp.view(scale_disp.shape+(1,))

        # return pose
        return flow, stereo, pose

    def forward_sep(self, x0_flow, x0n_flow, x0_stereo, x1_stereo, intrin=None, 
                scale_w=None, only_flow=False, only_stereo=False, gt_flow=None, gt_disp=None, 
                scale_disp=1.0, blxfx=80.):
        '''
        flow_out: pwcnet: 5 scale predictions up to 1/4 size
                  flownet: 1/1 size prediction
        stereo_out: psmnet: 3 scale predictions up to 1/1 size
                    stereonet: 1/1 size prediction 
        scale_w: the x-direction scale factor in data augmentation
        scale_depth: scale input depth and output motion to shift the data distribution
        '''
        # import ipdb;ipdb.set_trace()
        if only_flow:
            return self.flowNet([x0_flow, x0n_flow])

        if only_stereo:
            return self.stereoNet([x0_stereo, x1_stereo])

        # scale the flow size
        x0_flow = F.interpolate(x0_flow, size=(256,256), mode='bilinear', align_corners=True)
        x0n_flow = F.interpolate(x0n_flow, size=(256,256), mode='bilinear', align_corners=True)
        flow_out = self.flowNet([x0_flow, x0n_flow])
        if self.network==0 or self.network==1: # pwcnet, 1/4 size
            if self.down_scale:
                flow_out = F.interpolate(flow_out[0], size=(112,160), mode='bilinear', align_corners=True) 
                flow_out[:,0,:,:] = flow_out[:,0,:,:] * 2.5 # 448/256
                flow_out[:,1,:,:] = flow_out[:,1,:,:] * 1.75 # 640/256
                flow = flow_out
            else: # TODO: use pwcnet with down_scale=False will have problems
                flow = F.interpolate(flow_out[0], scale_factor=4, mode='bilinear', align_corners=True) # TODO: this is not used, flow_out is still in smaller size
        else: # flownet, 1/1 size
            if self.down_scale:
                flow_out = F.interpolate(flow_out, scale_factor=0.25, mode='bilinear', align_corners=True)
            flow = flow_out

        # downscale the stereo input 
        x0_stereo = F.interpolate(x0_stereo, size=(256,256), mode='bilinear', align_corners=True)
        x1_stereo = F.interpolate(x1_stereo, size=(256,256), mode='bilinear', align_corners=True)
        stereo_out = self.stereoNet([x0_stereo, x1_stereo])

        # scale the disparity size
        if self.network == 0: # psmnet, 3 outputs # deprecated since manually downscaled the input size
            if self.down_scale: 
                stereo_out[0] = F.interpolate(stereo_out[0], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo_out[1] = F.interpolate(stereo_out[1], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo_out[2] = F.interpolate(stereo_out[2], scale_factor=0.25, mode='bilinear', align_corners=True)
                stereo = stereo_out[2]
            else:
                stereo = stereo_out[2]
        else:
            if self.down_scale:
                stereo_out = F.interpolate(stereo_out, size=(112,160), mode='bilinear', align_corners=True)
                stereo_out = stereo_out * 2.5 # compensate the resize - 2.5 = 640/256
            stereo = stereo_out

        flow_input    = flow * self.flowNormFactor
        # disp_input    = stereo / self.stereoNormFactor # disparity network norm value, 0.02 or 0.05
        # depth_input    = blxfx / disp_input # from disparity to distance
        # depth_input    = depth_input * self.poseDepthNormFactor # from disparity to distance
        # depth_input    = 1 / depth_input
        # depth_input = 1 / ( (blxfx / (stereo / self.stereoNormFactor)) * self.poseDepthNormFactor)
        depth_input = stereo / blxfx / (self.stereoNormFactor * self.poseDepthNormFactor )

        # scale the disp back, because we treat it as depth
        depth_input = depth_input / scale_w
        # scale the disp for better motion distribution
        # import ipdb;ipdb.set_trace()
        
        # if self.autoDistTarget == 0:
        #     depth_input = depth_input * scale_disp
        # else: 
        #     distTarget = 1.0/(self.autoDistTarget * self.poseDepthNormFactor) # normalize the target by 0.25
        #     depth_mean = torch.mean(depth_input, (1,2,3))
        #     scale_disp = distTarget / depth_mean
        #     depth_input = depth_input * scale_disp.view(scale_disp.shape+(1,1))
        #     print (scale_disp)


        if self.intrinsic:
            inputTensor = torch.cat( ( flow_input, depth_input, intrin ), dim=1 )
        else:
            inputTensor = torch.cat( ( flow_input, depth_input ), dim=1 )
        
        pose = self.flowPoseNet( inputTensor, scale_disp=scale_disp )
        # scale the translation back
        # if self.autoDistTarget == 0:
        #     pose[:, :3] = pose[:, :3] * scale_disp
        # else:
        #     pose[:, :3] = pose[:, :3] * scale_disp.view(scale_disp.shape+(1,))

        return flow_out, stereo_out, pose

    def get_flow_loss(self, netoutput, target, criterion, mask=None, small_scale=False):
        if self.network == 0 or self.network==1: # pwc net
            # netoutput 1/4, 1/8, ..., 1/32 size flow
            # if mask is not None:
            return self.flowNet.get_loss(netoutput, target, criterion, mask, small_scale=small_scale) # To be tested
            # else:
            #     return self.flowNet.get_loss(netoutput, target, criterion, small_scale=small_scale)
        else: 
            if mask is not None:
                valid_mask = mask<128
                valid_mask = valid_mask.expand(target.shape)
                return criterion(netoutput[valid_mask], target[valid_mask])
            else:
                return criterion(netoutput, target)

    def get_stereo_loss(self, netoutput, target, criterion, mask=None):
        if self.network == 0: # psm net
            return self.stereoNet.get_loss(netoutput, target, criterion, mask)
        else: 
            return criterion(netoutput, target)

if __name__ == '__main__':
    
    voflownet = StereoVONet(network=0, intrinsic=True, flowNormFactor=1.0, down_scale=True, config=1, fixflow=True) # 
    voflownet.cuda()
    voflownet.eval()
    print(voflownet)
    import numpy as np
    import matplotlib.pyplot as plt
    import time

    x, y = np.ogrid[:448, :640]
    # print(x, y, (x+y))
    img = np.repeat((x + y)[..., np.newaxis], 3, 2) / float(512 + 384)
    img = img.astype(np.float32)
    print(img.dtype)
    imgInput = img[np.newaxis,...].transpose(0, 3, 1, 2)
    intrin = imgInput[:,:2,:112,:160].copy()

    imgTensor = torch.from_numpy(imgInput)
    intrinTensor = torch.from_numpy(intrin)
    print(imgTensor.shape)
    stime = time.time()
    for k in range(100):
        flow, pose = voflownet((imgTensor.cuda(), imgTensor.cuda(), intrinTensor.cuda()))
        print(flow.data.shape, pose.data.shape)
        print(pose.data.cpu().numpy())
        print(time.time()-stime)
    print (time.time()-stime)/100
