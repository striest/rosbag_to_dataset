from __future__ import print_function

import numpy as np
import cv2

from torch.utils.data import Dataset

class DatasetBase(Dataset):
    '''
    Loader for multi-modal data
    -----
    framelistfile: 
    TRAJNAME FRAMENUM
    FRAMESTR0
    FRAMESTR1
    ...
    -----
    Requirements: 
    The actural data path consists three parts: DATAROOT+TRAJNAME+CONVERT(FRAMESTR)
    The frames under the same TRAJNAME should be consequtive. So when the next frame is requested, it should return the next one in the same sequence. 
    The frames should exists on the harddrive. 
    Sequential data: 
    When a sequence of data is required, the code will automatically adjust the length of the dataset, to make sure the longest modality queried exists. 
    The IMU has a higher frequency than the other modalities. The frequency is imu_freq x other_freq. 
    '''
    def __init__(self, \
        framelistfile, \
        dataroot = "", \
        datatypes = "img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,imu", \
        modalitylens = [1,1,1,1,1,1,1,1,10], \
        transform=None, \
        imu_freq = 10, \
        frame_skip = 0, frame_stride = 1):

        super(DatasetBase, self).__init__()
        self.framelistfile = framelistfile
        self.dataroot = dataroot
        self.transform = transform
        self.imu_freq = imu_freq
        self.frame_skip = frame_skip # sample not consequtively, skip a few frames within a sequences
        self.frame_stride = frame_stride # sample less sequence, skip a few frames between two sequences 

        self.datatypelist = datatypes.split(',')
        self.modalitylenlist = modalitylens
        assert len(self.datatypelist)==len(modalitylens), "Error: datatype len {}, modalitylens len {}".format(len(self.datatypelist),len(modalitylens))
        self.trajlist, self.trajlenlist, self.framelist, self.imulenlist = self.parse_inputfile(framelistfile, frame_skip)
        self.sample_seq_len = self.calc_seq_len(self.datatypelist, modalitylens, imu_freq)
        self.seqnumlist = self.parse_length(self.trajlenlist, frame_skip, frame_stride, self.sample_seq_len)

        self.framenumFromFile = len(self.framelist)
        self.N = sum(self.seqnumlist)
        self.trajnum = len(self.trajlenlist)
        self.acc_trajlen = [0,] + np.cumsum(self.trajlenlist).tolist()
        self.acc_seqlen = [0,] + np.cumsum(self.seqnumlist).tolist() # [0, num[0], num[0]+num[1], ..]
        self.acc_imulen = [0,] + np.cumsum(self.imulenlist).tolist() # [0, num[0], num[0]+num[1], ..]
        print('Loaded {} sequences from {}...'.format(self.N, framelistfile))

        if 'cmd' in self.datatypelist:
            self.cmdlist = self.loadDataFromFile(self.trajlist, 'cmd/twist.npy')
        if 'odom' in self.datatypelist:
            self.odomlist = self.loadDataFromFile(self.trajlist, 'odom/odometry.npy')
        if 'imu' in self.datatypelist:
            self.imulist = self.loadDataFromFile(self.trajlist, 'imu/imu.npy')
        

    def parse_inputfile(self, inputfile, frame_skip):
        '''
        trajlist: [TRAJ0, TRAJ1, ...]
        trajlenlist: [TRAJLEN0, TRAJLEN1, ...]
        framelist: [FRAMESTR0, FRAMESTR1, ...]
        imulenlist: length of imu frames in each trajectory
                       [IMULen0, IMULen1, ...]
                       this is used to calculate the IMU frame index in __item__()                        
        '''
        with open(inputfile,'r') as f:
            lines = f.readlines()
        trajlist, trajlenlist, framelist, imulenlist = [], [], [], []
        ind = 0
        while ind<len(lines):
            line = lines[ind].strip()
            traj, trajlen = line.split(' ')
            trajlen = int(trajlen)
            trajlist.append(traj)
            trajlenlist.append(trajlen)
            imulenlist.append(trajlen*self.imu_freq)
            ind += 1
            for k in range(trajlen):
                if ind>=len(lines):
                    print("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                    raise Exception("Datafile Error: {}, line {}...".format(self.framelistfile, ind))
                line = lines[ind].strip()
                framelist.append(line)
                ind += 1

        print('Read {} trajectories, including {} frames'.format(len(trajlist), len(framelist)))
        return trajlist, trajlenlist, framelist, imulenlist

    def calc_seq_len(self, datatypelist, seqlens, imu_freq):
        '''
        decide what is the sequence length for cutting the data, considering the different length of different modalities
        For now, all the modalities are at the same frequency except for the IMU which is faster by a factor of 'imu_freq'
        seqlens: the length of seq for each modality
        '''
        maxseqlen = 0
        for ttt, seqlen in zip(datatypelist, seqlens):
            if ttt=='imu': # IMU has a higher freqency than other modalities
                seqlen = int((float(seqlen+imu_freq-1)/imu_freq))
            if seqlen > maxseqlen:
                maxseqlen = seqlen
        return maxseqlen

    def parse_length(self, trajlenlist, skip, stride, sample_length): 
        '''
        trajlenlist: the length of each trajectory in the dataset
        skip: skip frames within sequence
        stride: skip frames between sequence
        sample_length: the sequence length 
        Return: 
        seqnumlist: the number of sequences in each trajectory
        the length of the whole dataset is the sum of the seqnumlist
        '''
        seqnumlist = []
        # sequence length with skip frame 
        # e.g. x..x..x (sample_length=3, skip=2, seqlen_w_skip=1+(2+1)*(3-1)=7)
        seqlen_w_skip = (skip + 1) * sample_length - skip
        # import ipdb;ipdb.set_trace()
        for trajlen in trajlenlist:
            # x..x..x---------
            # ----x..x..x-----
            # --------x..x..x-
            # ---------x..x..x <== last possible sequence
            #          ^-------> this starting frame number is (trajlen - seqlen_w_skip + 1)
            # stride = 4, skip = 2, sample_length = 3, seqlen_w_skip = 7, trajlen = 16
            # seqnum = (16 - 7)/4 + 1 = 3
            seqnum = int((trajlen - seqlen_w_skip)/ stride) + 1
            if trajlen<seqlen_w_skip:
                seqnum = 0
            seqnumlist.append(seqnum)
        return seqnumlist


    def getDataPath(self, trajstr, framestrlist, datatype):
        '''
        return the file path name wrt the data type and framestr
        '''
        datapathlist = []

        for framestr in framestrlist: 
            if datatype == 'img0':
                datapathlist.append(trajstr + '/image_left/' + framestr + '.png')
            if datatype == 'img1':
                datapathlist.append(trajstr + '/image_right/' + framestr + '.png')
            if datatype == 'imgc':
                datapathlist.append(trajstr + '/image_left_color/' + framestr + '.png')
            if datatype == 'disp0':
                datapathlist.append(trajstr + '/depth_left/' + framestr + '.npy')
            if datatype == 'heightmap':
                datapathlist.append(trajstr + '/height_map/' + framestr + '.npy')
            if datatype == 'rgbmap':
                datapathlist.append(trajstr + '/rgb_map/' + framestr + '.npy')

        return datapathlist

    def idx2traj(self, idx):
        '''
        handle the stride and the skip
        return: 1. the index of trajectory 
                2. the indexes of all the frames in a sequence
        '''
        # import ipdb;ipdb.set_trace()
        for k in range(self.trajnum):
            if idx < self.acc_seqlen[k+1]:
                break

        remainingframes = (idx-self.acc_seqlen[k]) * self.frame_stride
        frameind = self.acc_trajlen[k] + remainingframes
        imuframeind = self.acc_imulen[k] + remainingframes * self.imu_freq

        # put all the frames in the seq into a list
        frameindlist = []
        for w in range(self.sample_seq_len):
            frameindlist.append(frameind)
            frameind += self.frame_skip + 1
        return self.trajlist[k], frameindlist, imuframeind


    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # parse the idx to trajstr
        trajstr, frameindlist, imuframeind = self.idx2traj(idx)
        framestrlist = [self.framelist[k] for k in frameindlist]

        sample = {}
        for datatype, datalen in zip(self.datatypelist, self.modalitylenlist): 
            datafilelist = self.getDataPath(trajstr, framestrlist[:datalen], datatype)
            if datatype == 'img0' or datatype == 'img1' or datatype == 'imgc':
                imglist = self.load_image(datafilelist)
                if imglist is None:
                    print("!!!READ IMG ERROR {}, {}, {}".format(idx, trajstr, framestrlist, datafilelist))
                sample[datatype] = imglist
            elif datatype == 'disp0' or datatype == 'heightmap' or datatype=='rgbmap':
                datalist = self.load_numpy(datafilelist)
                sample[datatype] = datalist
            elif datatype == 'odom':
                odomlist = self.load_odom(frameindlist, datalen)
                sample[datatype] = odomlist
            elif datatype == 'cmd':
                cmdlist = self.load_cmd(frameindlist, datalen)
                sample[datatype] = cmdlist
            elif datatype == 'imu': 
                imulist = self.load_imu(imuframeind, datalen)
                sample[datatype] = imulist
            else:
                print('Unknow Datatype {}'.format(datatype))

        # Transform.
        if ( self.transform is not None):
            sample = self.transform(sample)

        return sample

    def load_image(self, fns):
        imglist = []
        for fn in fns: 
            img = cv2.imread(self.dataroot + '/' + fn, cv2.IMREAD_UNCHANGED)
            imglist.append(img)
            assert img is not None, "Error loading image {}".format(fn)
        return imglist

    def load_numpy(self, fns):
        displist = []
        for fn in fns:
            disp = np.load(self.dataroot + '/' + fn)
            displist.append(disp)

        return displist

    def load_imu(self, startidx, len):
        return self.imulist[startidx: startidx+(len*(self.frame_skip+1)): self.frame_skip+1]

    def load_odom(self, frameindlist, datalen):
        return self.odomlist[frameindlist[:datalen]]

    def load_cmd(self, frameindlist, datalen):
        return self.cmdlist[frameindlist[:datalen]]

    def loadDataFromFile(self, trajlist, data_folder_and_filename):
        print('Loading data from {}...'.format(data_folder_and_filename))
        datalist = []
        for k, trajdir in enumerate(trajlist): 
            trajpath = self.dataroot + '/' + trajdir
            cmds = np.load(trajpath + '/' + data_folder_and_filename).astype(np.float32) # framenum
            datalist.extend(cmds)
            if k%100==0:
                print('    Processed {} trajectories...'.format(k))
        return np.array(datalist)


if __name__ == '__main__':
    # framelistfile = 'tartandrive_train.txt'
    framelistfile = '/home/mateo/Data/SARA/TartanDriveCost/Splits/train.txt'
    datarootdir = '/home/mateo/Data/SARA/TartanDriveCost/Trajectories'

    dataset = DatasetBase(framelistfile, \
                            dataroot= datarootdir, \
                            datatypes = "img0,img1,imgc,disp0,heightmap,rgbmap,cmd,odom,imu", \
                            modalitylens = [10,10,10,10,10,10,10,10,100], \
                            # modalitylens = [477,477,477,477,477,477,477,477,4770], \
                            transform=None, \
                            imu_freq = 10, \
                            frame_skip = 0, frame_stride=5)
    print('Dataset length: ',len(dataset))
    # import pdb;pdb.set_trace()
    for k in range(0, len(dataset), 10):
        sample = dataset[k]
        print('Sample index: {}'.format(k))
        for k in sample: 
            e = sample[k]
            if isinstance(e, list):
                print('   {}: len {}, shape {}'.format(k, len(e), e[0].shape))
            elif isinstance(e, np.ndarray):
                print('   {}: shape {}'.format(k, e.shape))
        if 'imgc' in sample: # visualize the image
            cv2.imshow('img', sample['rgbmap'][0])
            cv2.waitKey(10)
    print('---')
