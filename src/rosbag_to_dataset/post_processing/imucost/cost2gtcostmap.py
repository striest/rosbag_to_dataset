import numpy as np
from os.path import join, isdir
from os import listdir, mkdir
import cv2
from .utils import add_text
from .terrain_map_tartandrive import TerrainMap, tartanvo_motion_to_odom, get_local_path
import torch

class GTCostMapNode(object):
    def __init__(self, crop_size=[2,2]) -> None:
        map_height = 12.0
        map_width = 12.0
        resolution = 0.02
        output_size = [int(crop_size[0]/resolution), int(crop_size[1]/resolution)]

        self.map_metadata = {
            'height': map_height,
            'width': map_width,
            'resolution': resolution,
            'origin': [-2.0, -6.0],
        }

        self.crop_params ={
            'crop_size': crop_size,
            'output_size': output_size
        }

    def get_crops(self, ):
        '''
        input: poses, a starting frame
        output:  
        '''
    
    def process(self, traj_root_folder, costmap_output_folder, cost_input_folder='cost2', odom_folder='tartanvo_odom', new_odom=True):
        '''
        Output the costmap in the costmap folder
        '''
        print('Working on traj ', traj_root_folder)

        if costmap_output_folder is not None:
            outdir = join(traj_root_folder, costmap_output_folder)
            if not isdir(outdir):
                mkdir(outdir)
                print('Create folder: {}'.format(outdir))

        rgbmap_folder = 'rgb_map'
        heightmap_folder ='height_map'

        maplist = listdir(join(traj_root_folder, rgbmap_folder))
        maplist = [mm for mm in maplist if mm.endswith('.npy')]
        maplist.sort()

        framenum = len(maplist)

        # using tartanvo
        if odom_folder.startswith('tartanvo'):
            motions = np.load(join(traj_root_folder, odom_folder, 'motions.npy'))
            assert framenum==motions.shape[0]+1, "Error: map number {} and motion number {} mismatch".format(framenum, motions.shape[0])
        else: # using novatel
            odom = np.load(join(traj_root_folder, odom_folder, 'odometry.npy'))
            assert framenum==odom.shape[0], "Error: map number {} and odom number {} mismatch".format(framenum, odom.shape[0])

        cost = np.load(join(traj_root_folder, cost_input_folder, 'cost.npy'))

        cropnum = 50
        preframenum = 10
        # for startframe in range(startframe, framenum, 1):
        for currentframe in range(0, framenum-cropnum, 1):
            if currentframe%100 == 0:
                print('  processing', currentframe)
            startframe = max(0, currentframe-preframenum)
            endframe = min(framenum, currentframe + cropnum)
            seqcropnum = endframe - startframe
            rgbmap = np.load(join(traj_root_folder, rgbmap_folder, maplist[currentframe]))
            # heightmap = np.load(join(traj_root_folder, heightmap_folder, maplist[currentframe]))

            tm = TerrainMap(map_metadata=self.map_metadata, device="cpu")

            if odom_folder.startswith('tartanvo'): # for tartanvo 
                pose_transfromed = tartanvo_motion_to_odom(motions[startframe:endframe-1,:]) # motion has 1 less frame than other modalities
                assert seqcropnum==len(pose_transfromed), 'Error! Pose transformed length {}, mismatch the number {}'.format(len(pose_transfromed), seqcropnum)
                local_path = get_local_path(pose_transfromed, ref_ind=(currentframe-startframe))
            else:
                local_path = get_local_path(odom[startframe:endframe,:], ref_ind=(currentframe-startframe))
                if not new_odom:
                    local_path = local_path[:, [1,0,2]] # swith x and y
                    local_path[:,1] = -local_path[:,1]

            local_path = torch.from_numpy(local_path)
            # crop_coordinates: [B x H x W x 2], valid_masks: [B x H x W ]
            crop_coordinates, valid_masks = tm.get_crop_batch_and_masks(local_path, self.crop_params)
            crop_coordinates = crop_coordinates.numpy()
            valid_masks = valid_masks.numpy()
            # import ipdb;ipdb.set_trace()
            print('  valid mask', np.sum(valid_masks))
            assert len(crop_coordinates) == seqcropnum, "Error! Patch number doesn't match {} - {}".format(len(crop_coordinates), seqcropnum)
            # import ipdb;ipdb.set_trace()

            costmap = np.zeros((seqcropnum, rgbmap.shape[0], rgbmap.shape[1]),dtype=np.float32) 
            costcount = np.zeros((rgbmap.shape[0], rgbmap.shape[1]),dtype=np.uint8)

            for k in range(seqcropnum):
                costind = int(startframe+k)
                assert costind>=0 and costind<len(cost), "Error! Cost ind out of limit {}/{}".format(costind, len(cost))
                cc = crop_coordinates[k][valid_masks[k]] # -1 x 2
                dd = np.unique(cc, axis=0)
                print('  cordinates shape', cc.shape, dd.shape)
                costmap[k, dd[:,0],dd[:,1]] = cost[costind] # TODO: test if the coordinate has duplicate
                costcount[dd[:,0], dd[:,1]] = costcount[dd[:,0], dd[:,1]] + 1
            costmap = np.sum(costmap, axis=0) / (costcount+1e-6)
            costmask = costcount > 0
            costmap = (costmap*255).astype(np.uint8)

            costmap_vis = costmap.copy()
            costmap_vis[costcount == 0] = 128
            cv2.imshow('img',costmap_vis)
            cv2.waitKey(1)
            cv2.imshow('count',np.clip(costcount*10,0,255))
            cv2.waitKey(0)
            # import ipdb;ipdb.set_trace()

        #     rgblist.append(patches_numpy_list[rgbind])
        #     rgbind -= 1

        # patchvis = np.concatenate(rgblist, axis=1)
        # cv2.imshow("img", patchvis)
        # cv2.waitKey(0)
        import ipdb;ipdb.set_trace()

if __name__=="__main__":
    base_dir = '/home/wenshan/tmp/arl_data/full_trajs/20220531_lowvel_0'
    costmap = GTCostMapNode()
    costmap.process(base_dir, 'costmap')