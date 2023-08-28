import numpy as np
from os.path import join, isdir, isfile
from os import listdir, mkdir
import cv2
from .utils import add_text
from .terrain_map_tartandrive import TerrainMap, tartanvo_motion_to_odom, get_local_path, tartanodom2vel, gpsodom2vel
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

    def crop_contains_enough_known(self, valid_mask, crop_coordinates, crop_valid_mask, threshold = 0.3):
        '''
        valid_mask: h x w, indicating if the pixel is known or unknown
        crop_coordinates: b x h x w x 2
        crop_valid_mask: if the coordinates are in or out of the mapping range
        output: a boolean vector of whether each crop contrains enough known area or not
        '''
        map_pixels = crop_coordinates.shape[0] * crop_coordinates.shape[1]
        enough_known = []
        known_counts = []
        for k in range(len(crop_coordinates)): 
            crop_coord = crop_coordinates[k]
            crop_mask = crop_valid_mask[k]
            valid_coord = crop_coord[crop_mask, :] # valid_num x 2
            known_mask_valid = valid_mask[valid_coord[:,0], valid_coord[:,1]]
            known_count = np.sum(known_mask_valid)
            known_counts.append(known_count)
            known_percent = known_count / float(map_pixels) 
            if known_percent > threshold:
                enough_known.append(True)
            else:
                enough_known.append(False)
        # print(enough_known, known_counts)
        return enough_known

    def find_intevention(self, control, velx):
        '''
        intevention: 1. look at the brake signal
                     2. look at the velx if it come to and stay zero after the brake
        we assume that the intervention can only be at the last part of the trajectory
        '''
        # print(control.shape)
        seqlen = len(velx)
        res = np.zeros(seqlen)
        intervene = control[:,1] > 200
        vel0 = np.abs(velx) < 0.5
        k = seqlen-1
        if vel0[k]: # the final vel is 0
            while k>=0: # find in the inverse order where the vel = 1m/s
                if velx[k] > 1.0:
                    break
                k -= 1
            if k>0 and (True in intervene[max(0,k-10):]): # if the brake is pushed
                res[k:] = 1
                print("  intevention {}/{}".format(k, seqlen))
        return res

    def modify_vel_cost_for_intevention(self, control, motion, velx, cost, dt=0.1):
        intervention = self.find_intevention(control, velx)
        seqlen = len(velx)
        if intervention[-1]: # intervention at the end of this sequence
            interlen = int(np.sum(intervention))
            # generate a random vel
            inter_start = seqlen - interlen
            last_vel = velx[max(0,inter_start-1)]
            randvel = np.linspace(last_vel, np.random.randint(1,10), interlen).astype(np.float32)
            velx[inter_start:] = randvel
            cost[inter_start:] = 1.0
            motion[inter_start:,0] = randvel[:-1] * dt
        return motion, velx, cost
    
    def process(self, traj_root_folder, out_root_folder, costmap_output_folder, cost_input_folder='cost2', odom_folder='tartanvo_odom', new_odom=True, vis='file'):
        '''
        Output the costmap in the costmap folder
        '''
        print('Working on traj ', traj_root_folder)

        if costmap_output_folder is not None:
            outdir = join(out_root_folder, costmap_output_folder)
            if not isdir(outdir):
                mkdir(outdir)
                print('Create folder: {}'.format(outdir))

        rgbmap_folder = 'rgb_map'
        heightmap_folder ='height_map'
        intervention_folder='intervention'

        maplist = listdir(join(traj_root_folder, rgbmap_folder))
        maplist = [mm for mm in maplist if mm.endswith('.npy')]
        maplist.sort()

        framenum = len(maplist)

        # using tartanvo
        if odom_folder.startswith('tartanvo'):
            motions = np.load(join(traj_root_folder, odom_folder, 'motions.npy'))
            vels = tartanodom2vel(motions, dt=0.1)
            assert framenum==motions.shape[0]+1, "Error: map number {} and motion number {} mismatch".format(framenum, motions.shape[0])
        else: # using novatel
            odom = np.load(join(traj_root_folder, odom_folder, 'odometry.npy'))
            vels = gpsodom2vel(odom) # TODO: test
            assert framenum==odom.shape[0], "Error: map number {} and odom number {} mismatch".format(framenum, odom.shape[0])

        # append one frame at the end just to align the number of frames
        vels = np.concatenate((vels, vels[-1:]), axis=0)

        # load cost
        cost = np.load(join(traj_root_folder, cost_input_folder, 'cost.npy'))
        ## Load intervention data
        intervention_dir = join(traj_root_folder, intervention_folder)
        intervention_fp = join(intervention_dir, "control.npy")

        if cost.shape[0] < framenum:
            print("*** Error: cost and map number mismatch {}/{}".format(cost.shape[0], framenum))
            cost = np.concatenate((cost, [0,]*(framenum-cost.shape[0]))) # padding zero
        
        if isfile(intervention_fp):
            intervention_data = np.load(intervention_fp)
            if not odom_folder.startswith('tartanvo'):
                raise NotImplementedError
            motions, vels[:,0], cost = self.modify_vel_cost_for_intevention(intervention_data, motions, vels[:,0], cost, dt=0.1)
        else: 
            print('Missing intervention file', intervention_fp)
            intervention_data = np.zeros((framenum,2))


        cropnum = 50
        preframenum = 10
        # for startframe in range(startframe, framenum, 1):
        for currentframe in range(0, framenum, 1):
            if currentframe%100 == 0:
                print('  processing', currentframe)
            startframe = max(0, currentframe-preframenum)
            endframe = min(framenum, currentframe + cropnum)
            seqcropnum = endframe - startframe
            rgbmap = np.load(join(traj_root_folder, rgbmap_folder, maplist[currentframe]))
            heightmap = np.load(join(traj_root_folder, heightmap_folder, maplist[currentframe]))

            map_valid_mask = heightmap[:,:,0] < 1000 # there are unknow areas in the currentmap

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
            valid_masks = valid_masks.numpy() # [B x H x W ]
            # print('  valid mask', np.sum(valid_masks))
            assert len(crop_coordinates) == seqcropnum, "Error! Patch number doesn't match {} - {}".format(len(crop_coordinates), seqcropnum)
            # import ipdb;ipdb.set_trace()

            # filter out patches that covers too much unknown area
            enough_known = self.crop_contains_enough_known(map_valid_mask, crop_coordinates, valid_masks)

            # Start to aggregate the cost
            costmap = np.zeros((seqcropnum, rgbmap.shape[0], rgbmap.shape[1]),dtype=np.float32) 
            costcount = np.zeros((rgbmap.shape[0], rgbmap.shape[1]),dtype=np.uint32)
            costmap_ave = np.zeros((rgbmap.shape[0], rgbmap.shape[1]),dtype=np.float32) 

            for k in range(seqcropnum):
                # this crop covers mostly unknown or out-of-boundary region
                if not enough_known[k]:
                    continue

                costind = int(startframe+k)
                assert costind>=0 and costind<len(cost), "Error! Cost ind out of limit {}/{}".format(costind, len(cost))
                cc = crop_coordinates[k][valid_masks[k]] # -1 x 2
                dd = np.unique(cc, axis=0)
                # print('  cordinates shape', cc.shape, dd.shape)
                costmap[k, dd[:,0],dd[:,1]] = cost[costind] # TODO: test if the coordinate has duplicate
                costcount[dd[:,0], dd[:,1]] = costcount[dd[:,0], dd[:,1]] + 1

            costmask = costcount > 0
            costmap_ave[costmask] = np.sum(costmap[:,costmask], axis=0) / costcount[costmask]
            costmap_ave = (costmap_ave*255).astype(np.uint8) # scale and save the value in uint8 to save space
            costmap_res = np.stack((costmap_ave, costmask), axis=-1)

            # save the velocities
            vels_crop = vels[startframe:endframe, :]
            assert len(vels_crop)==len(enough_known), "Error, the number of vel {} is different from the number of mask {} ".format(len(vels_crop), len(enough_known))
            vels_crop_valid = vels_crop[enough_known]
            # print(len(vels_crop_valid), vels_crop_valid)

            costmap_vis = costmap_ave.copy()
            costmap_vis[costcount == 0] = 128
            costmap_vis = cv2.applyColorMap(costmap_vis, cv2.COLORMAP_JET)
            disp_overlay = rgbmap.copy()
            disp_overlay[costcount > 0] = (rgbmap[costcount > 0] * 0.8 + costmap_vis[costcount > 0] * 0.2).astype(np.uint8)
            disp = cv2.hconcat((disp_overlay, costmap_vis))
            # cv2.imshow('img',disp)
            # cv2.waitKey(1)
            if costmap_output_folder is not None:
                # save the result
                np.save(join(outdir, maplist[currentframe]), costmap_res)
                np.savetxt(join(outdir, maplist[currentframe].replace('.npy', '_vel.txt')), vels_crop_valid)
                # save visualization
                cv2.imwrite(join(outdir, maplist[currentframe].replace('.npy', '.png')), cv2.resize(disp, (0,0), fx=0.5, fy=0.5))
            # cv2.imshow('count',np.clip(costcount*20,0,255).astype(np.uint8))
            # cv2.waitKey(1)
            # import ipdb;ipdb.set_trace()

if __name__=="__main__":
    base_dir = '/home/amigo/workspace/ros_atv/src/rosbag_to_dataset/test_output/20210903_10' #'/home/wenshan/tmp/arl_data/full_trajs/20210826_61' # 20210826_61 # 20220531_lowvel_0
    costmap = GTCostMapNode()
    costmap.process(base_dir, 'costmap')