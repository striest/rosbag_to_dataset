from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from TartanDriveDataset import DatasetBase
from wheeledsim_rl.util.util import dict_to, DummyEnv
import torch
import random

import numpy as np

import matplotlib.pyplot as plt
import argparse
import os
import yaml
import time
from datetime import datetime


from torch import nn, optim

from tqdm import tqdm

from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_to, DummyEnv
from wheeledsim_rl.util.rl_util import split_trajs
from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.normalizer import ObservationNormalizer
from wheeledsim_rl.util.ouNoise import ouNoise
from wheeledsim_rl.policies.ou_policy import OUPolicy
from wheeledsim_rl.collectors.base_collector import Collector
from wheeledsim_rl.trainers.experiment import Experiment

from wheeledsim_rl.networks.str_to_cls import str_to_cls as network_str_to_cls
from wheeledsim_rl.trainers.str_to_cls import str_to_cls as trainer_str_to_cls
from wheeledsim_rl.losses.str_to_cls import str_to_cls as loss_str_to_cls

import torch.distributed as dist
import argparse
from datetime import datetime
import os
import torch.multiprocessing as mp

def convert_queue(observation, action, queue):
	"""
	Actually convert the queue into numpy.
	"""
	out = {
		'observation':{},
		'action':{},
	}

	for topic in observation:
		data = queue[topic]
		out['observation'][topic] = data
	
	for topic in action:
		data = queue[topic]
		out['action'][topic] = data

	if len(action) > 0:
		out['action'] = np.concatenate([v for v in out['action'].values()], axis=1)
	
	return out

def traj_to_torch(traj,imu_freq):
	torch_traj = {}

	#-1 to account for next_observation
	# trajlen = min(traj['action'].shape[0], min([traj['observation'][k].shape[0] for k in traj['observation'].keys()])) - 1
	trajlen = min(traj['action'].shape[1], min([traj['observation'][k].shape[1] for k in traj['observation'].keys()]))

	#Nan check

	torch_traj['action'] = [0]*traj['action'].shape[0]
	torch_traj['reward'] = [0]*traj['action'].shape[0]
	torch_traj['terminal'] = [0]*traj['action'].shape[0]
	torch_traj['observation'] = [0]*traj['action'].shape[0]
	torch_traj['next_observation'] = [0]*traj['action'].shape[0]
	for i in range(traj['action'].shape[0]):
		max_nan_idx=-1
		for t in range(trajlen):
			temp = [not np.isfinite(traj['observation'][k][i][t]).any() for k in traj['observation'].keys()]
			obs_nan = any(temp)
			act_nan = not np.any(np.isfinite(traj['action'][i][t]))

			if obs_nan or act_nan:
				max_nan_idx = t
		start_idx = max_nan_idx + 1
		torch_traj['action'][i] = torch.tensor(traj['action'][i][start_idx:trajlen]).float()
		torch_traj['reward'][i] = torch.zeros(trajlen - start_idx)
		torch_traj['terminal'][i] = torch.zeros(trajlen - start_idx).bool()
		torch_traj['terminal'][i][-1] = True

		temp = {}
		for k,v in traj['observation'].items():
			if k != 'imu':
				temp[k] = torch.tensor(v[i][start_idx:trajlen]).float()
			else:
				data = torch.tensor(v[i]).float()
				strides = imu_freq
				if strides > 1:
				#If strided, need to reshape data (we also need to (same) pad the end)
					# print(data.shape)
					# pad_t = strides - (data.shape[0] % strides)
					# data = torch.cat([data, torch.stack([data[-1]] * pad_t, axis=0)], axis=0)
					# print(data.shape)
					data = data.reshape(-1, strides, *data.shape[1:])
				temp[k] = data
		torch_traj['observation'][i] = temp

		temp = {}
		for k,v in traj['observation'].items():
			if k != 'imu':
				temp[k] = torch.tensor(v[i][start_idx+1:trajlen+1]).float()
				temp[k] = torch.cat((temp[k],temp[k][[-1]]),dim=0)
			else:
				strides = imu_freq
				data = torch.tensor(v[i][start_idx+1*strides:]).float()
				pad_t = strides - (data.shape[0] % strides)
				data = torch.cat([data, torch.stack([data[-1]] * pad_t, axis=0)], axis=0)
				if strides > 1:
				#If strided, need to reshape data (we also need to (same) pad the end)
					# pad_t = strides - (data.shape[0] % strides)
					# data = torch.cat([data, torch.stack([data[-1]] * pad_t, axis=0)], axis=0)
					data = data.reshape(-1,strides, *data.shape[1:])
				temp[k] = data
				# temp[k] = torch.cat((temp[k],temp[k][[-1]]),dim=0)
		torch_traj['next_observation'][i] = temp

		# temp_dict = {}
		# for k,v in traj['observation'].items():
		#     temp_dict.append(k:torch.stack(torch.tensor(v[i][start_idx+1:trajlen+1]),torch.tensor(v[i][-1])).float())
			
		# torch_traj['next_observation'][i] = {k:torch.cat((torch.tensor(v[i][start_idx+1:trajlen+1]),torch.tensor(v[i][[-1]])),dim=0).float() for k,v in traj['observation'].items()}
		# print(torch_traj['next_observation'][i]['state'].shape )

	torch_traj['action'] = torch.stack(torch_traj['action'])
	torch_traj['reward'] = torch.stack(torch_traj['reward'])
	torch_traj['terminal'] = torch.stack(torch_traj['terminal'])
	torch_traj['observation'] = {key: torch.stack([i[key] for i in torch_traj['observation']]) for key in torch_traj['observation'][0].keys()}
	torch_traj['next_observation'] = {key: torch.stack([i[key] for i in torch_traj['next_observation']]) for key in torch_traj['next_observation'][0].keys()}
	# print("Shape")
	# print(torch_traj['next_observation']['state'].shape)
	return torch_traj

def preprocess_pose(traj, zero_init=True):
	"""
	Do a sliding window to smooth it out a bit
	"""
	N = 2
	T = traj['observation']['state'].shape[1]
	pad_states = torch.cat([traj['observation']['state'][:,[0]]] * N + [traj['observation']['state'][:]] + [traj['observation']['state'][:,[-1]]] * N,dim = 1)    
	smooth_states = torch.stack([pad_states[:,i:T+i] for i in range(N*2 + 1)], dim=-1).mean(dim=-1)[:,:,:3]
	pad_next_states = torch.cat([traj['next_observation']['state'][:,[0]]] * N + [traj['next_observation']['state'][:]] + [traj['next_observation']['state'][:,[-1]]] * N,dim = 1)
	temp = [pad_next_states[:,i:T+i] for i in range(N*2+1)] # Used T-1 instead of T iin indexing as the lentgh of 
	te = torch.stack(temp, dim=-1)
	smooth_next_states = te.mean(dim=-1)[:,:,:3]
	traj['observation']['state'][:,:, :3] = smooth_states
	traj['next_observation']['state'][:,:, :3] = smooth_next_states

	if zero_init:
		init_state = traj['observation']['state'][:,[0],:3]
		traj['next_observation']['state'][:,:,:3] = traj['next_observation']['state'][:,:,:3] - init_state
		traj['observation']['state'][:,:, :3] = traj['observation']['state'][:,:, :3]- init_state

	return traj

def preprocess_observations(res, fill_value=0.):
	"""
	NOTE: These are temporary fixes to get the models to run.
	we are
		1. Just looking at the high value of the map (map should listen to both)
		2. Replacing nans with a fill value (we should add a mask channel)
	"""
	for k in res['observation'].keys():
		# if 'map' in k or 'image_rgb' in k:
		if k not in ['state','imu']:
			map_data = res['observation'][k]
			map_data[~torch.isfinite(map_data)] = fill_value
			res['observation'][k] = map_data.moveaxis(-1, -3)
			map_data = res['next_observation'][k]
			map_data[~torch.isfinite(map_data)] = fill_value
			res['next_observation'][k] = map_data.moveaxis(-1, -3)
	return res

if __name__ == '__main__':
	
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--config_fp', type=str, required=True, help='The yaml file for the experiment')
	args = parser.parse_args()
	

	config = yaml.load(open(args.config_fp, 'r'), Loader=yaml.FullLoader)
	now = datetime.now()
	config["experiment_fp"] = os.path.join(
            config["experiment_fp"], f'{now.strftime("%m - %d - %Y ,  %H:%M:%S")}')
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = config["output_device"]+","+config["device_ids"]	

	observation = ['img0','heightmap','rgbmap','img1','imgc','disp0','odom','imu']
	observation = ['imgc','heightmap','rgbmap','odom','imu']
	remap= {'img0':'image_left','img1':'image_right','disp0':'disp0','imgc':'image_rgb','odom':'state','imu':'imu','heightmap':'heightmap','rgbmap':'rgbmap'}
	action = ['cmd']
	remapped_obs = [remap[i] for i in observation]
	datatypes = "imgc,heightmap,rgbmap,cmd,odom,imu" #observation + cmd
	dt = 0.1
	N_per_step= 10
	modality = 30
	modality_len = [modality]*(len(datatypes.split(","))-1)+[modality*N_per_step]
	print(modality_len)
	dataset_train = DatasetBase(config['train_framelistfile'], \
							dataroot= config['train_fp'], \
							datatypes = datatypes, \
							modalitylens = modality_len, \
							transform=None, \
							imu_freq = N_per_step, \
							frame_skip = 0, frame_stride=5, config = config , remap = remap)
	train_dataloader = DataLoader(dataset_train, batch_size=1, shuffle=True)
	
	train_dataloader = iter(train_dataloader)
	for _ in range(len(train_dataloader)):
		data = next(train_dataloader)
		remapped_data = {}
		for i in observation:
			remapped_data[remap[i]] = data[i]
		for i in action:
			remapped_data[i] = data[i]
		res = convert_queue(remapped_obs, action, remapped_data)
		torch_traj = traj_to_torch(res,N_per_step)
		torch_traj = preprocess_pose(torch_traj)
		torch_traj = preprocess_observations(torch_traj)
		torch_traj['dt'] = torch.ones(torch_traj['action'].shape[1]) * dt
		for k in torch_traj.keys():
			if(type(torch_traj[k]) is dict):
				for k1 in torch_traj[k].keys():
					torch_traj[k][k1] = torch.squeeze(torch_traj[k][k1],dim=0)
			else:
				torch_traj[k] = torch.squeeze(torch_traj[k],dim=0)

		env = DummyEnv(torch_traj)
		break

	noise = ouNoise(lowerlimit=env.action_space.low, upperlimit=env.action_space.high, var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
	policy = OUPolicy(noise)
	
	train_dataloader = DataLoader(dataset_train, batch_size=64,num_workers=4,persistent_workers=True)
	print('\nloading train data...')
	traj_list_train = []
	
	train_dataloader = iter(train_dataloader)
	for _ in tqdm(range(len(train_dataloader))):
	# for _ in tqdm(range(4)):
		data = next(train_dataloader)
		remapped_data = {}
		for i in observation:
			remapped_data[remap[i]] = data[i]
		for i in action:
			remapped_data[i] = data[i]
		res = convert_queue(remapped_obs, action, remapped_data)
		torch_traj = traj_to_torch(res,N_per_step)
		torch_traj = preprocess_pose(torch_traj)
		torch_traj = preprocess_observations(torch_traj)
		torch_traj['dt'] = torch.ones(torch_traj['action'].shape[0:2]) * dt
		for i in range(torch_traj['action'].shape[0]):
			temp_torch_traj = {'observation':{},'next_observation':{}}
			for k in torch_traj.keys():
				if(type(torch_traj[k]) is dict):
					for k1 in torch_traj[k].keys():
						temp_torch_traj[k][k1] = torch.squeeze(torch_traj[k][k1][i],dim=0)
				else:
					temp_torch_traj[k] = torch.squeeze(torch_traj[k][i],dim=0)
			
			traj_list_train.append(temp_torch_traj)
	# train_buf = NStepDictReplayBuffer(env, capacity=3000).to(config['model_device'])
	train_dataloader._shutdown_workers()
	train_buf = NStepDictReplayBuffer(env, capacity=3000)

	for traj in traj_list_train:
		train_buf.insert(traj)
	
	dataset_eval = DatasetBase(config['eval_framelistfile'], \
							dataroot= config['eval_fp'], \
							datatypes = datatypes, \
							modalitylens = modality_len, \
							transform=None, \
							imu_freq = N_per_step, \
							frame_skip = 0, frame_stride=5,config=config , remap = remap)

	eval_dataloader = DataLoader(dataset_eval, batch_size=64, num_workers=4,persistent_workers=True)
	eval_dataloader = iter(eval_dataloader)
	
	print('\nloading Eval data...')
	traj_list_eval = []
	lst = time.time()
	for _ in tqdm(range(len(eval_dataloader))):
	# for _ in tqdm(range(4)):
		data = next(eval_dataloader)
		remapped_data = {}
		for i in observation:
			remapped_data[remap[i]] = data[i]
		for i in action:
			remapped_data[i] = data[i]
		res = convert_queue(remapped_obs, action, remapped_data)
		torch_traj = traj_to_torch(res,N_per_step)
		torch_traj = preprocess_pose(torch_traj)
		torch_traj = preprocess_observations(torch_traj)
		torch_traj['dt'] = torch.ones(torch_traj['action'].shape[0:2]) * dt
		for i in range(torch_traj['action'].shape[0]):
			temp_torch_traj = {'observation':{},'next_observation':{}}
			for k in torch_traj.keys():
				if(type(torch_traj[k]) is dict):
					for k1 in torch_traj[k].keys():
						temp_torch_traj[k][k1] = torch.squeeze(torch_traj[k][k1][i],dim=0)
				else:
					temp_torch_traj[k] = torch.squeeze(torch_traj[k][i],dim=0)
			
			traj_list_eval.append(temp_torch_traj)

	eval_dataloader._shutdown_workers()
	if not isinstance(traj_list_eval, list):
		trajs = split_trajs(traj_list_eval.sample_idxs(torch.arange(len(traj_list_eval))))
	else:
		trajs = traj_list_eval

	print('\nloading Model...')

	inorm = ObservationNormalizer(env=env)
	onorm = ObservationNormalizer(env=env)

	encoder_dim = config['latent_model']['params']['rnn_hidden_size']
	decoder_dim = encoder_dim + config['latent_model']['params'].get('latent_size', 0)
	print("enc = ", encoder_dim)
	print("dec = ", decoder_dim)
	batch = train_buf.sample(2, N=3)
	
	state_dim = batch['observation']['state'].shape[-1]
	act_dim = batch['action'].shape[-1]
	encoders = {}
	decoders = {}
	losses = {}
	#Parse modalitites
	if config['modalities'] is not None:
		for name, info in config['modalities'].items():
			topic = info['topic']
			encoder_params = info['encoder']
			decoder_params = info['decoder']
			loss_params = info['loss']
			
			sample = batch['observation'][topic][:, 0]

			encoder_cls = network_str_to_cls[encoder_params['type']]
			decoder_cls = network_str_to_cls[decoder_params['type']]
			loss_cls = loss_str_to_cls[loss_params['type']]

			#Assumed that the first two net args are insize, outsize
			encoder = encoder_cls(sample.shape[1:], encoder_dim, **encoder_params['params'])
			decoder = decoder_cls(decoder_dim, sample.shape[1:], **decoder_params['params'])
			loss = loss_cls(**loss_params['params'])

			encoders[topic] = encoder
			decoders[topic] = decoder
			losses[topic] = loss

	latent_model_cls = network_str_to_cls[config['latent_model']['type']]
	latent_model = latent_model_cls(encoders, decoders, **config['latent_model']['params'], input_normalizer=inorm, output_normalizer=onorm, state_insize=state_dim, action_insize=act_dim)

	latent_model = latent_model.to(config['model_device'])

	#initialize normalizers
	print('\ninitializing normalizers...')
	tbuf = []

	K=100
	for i in range(K):
		ts = time.time()
		all_data = dict_to(random.choice(traj_list_train),config['model_device'])
		tbuf.append(time.time() - ts)
		targets = latent_model.get_training_targets(all_data)
		latent_model.input_normalizer.update(all_data)
		latent_model.output_normalizer.update({'observation':targets})
		print('{}/{}'.format(i+1, K), end='\r')

	print('Avg collect time = {:.6f}s'.format(sum(tbuf)/len(tbuf)))

	print('MODEL NPARAMS =', sum([np.prod(x.shape) for x in latent_model.parameters()]))

	opt = optim.Adam(latent_model.parameters(), lr=config['lr'])
	
	collector = Collector(env, policy=policy)
	trainer_cls = trainer_str_to_cls[config['trainer']['type']]
	latent_model = nn.DataParallel(latent_model,output_device = 0,device_ids=list(map(int,config["device_ids"].split(","))))

	# latent_model = nn.DataParallel(latent_model,device_ids=[0,1])
	latent_model = latent_model.to(config['model_device'])
	latent_model.module.input_normalizer = latent_model.module.input_normalizer.to("cpu")
	# latent_model.module.output_normalizer = latent_model.module.output_normalizer.to("cpu")

	#change collecter and training buf to model so it can be converted

	trainer = trainer_cls(env, policy, latent_model, opt, train_buf,collector, losses=losses, steps_per_epoch=0, eval_dataset=trajs, **config['trainer']['params'], device="cpu")
	config['experiment_fp'] = config['experiment_fp']+"/epoch10"
	experiment = Experiment(trainer, config['name'], config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], train_data=traj_list_train, buffer_cycle=1)

	maybe_mkdir(os.path.join(config['experiment_fp'], config['name']), force=False)
	with open(os.path.join(config['experiment_fp'], config['name'], '_config.yaml'), 'w') as fp:
		yaml.dump(config, fp)

	if True:
		experiment.run()
		trainer.total_epochs = 5000
		config['experiment_fp'] = config['experiment_fp'][:-2]+"5000"
		experiment = Experiment(trainer, config['name'], config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], train_data=traj_list_train, buffer_cycle=1)

		maybe_mkdir(os.path.join(config['experiment_fp'], config['name']), force=False)
		with open(os.path.join(config['experiment_fp'], config['name'], '_config.yaml'), 'w') as fp:
			yaml.dump(config, fp)
		experiment.run()


	else:
		print("hellooooo")
		best_model_dir = "/project/learningphysics/tartandrive_trajs_parv_test/experiment/reconstruction_atv_all_t50/_best/model.cpt"
		latent_model = torch.load(best_model_dir)
		latent_model = nn.DataParallel(latent_model,device_ids=[0,1])
		latent_model = latent_model.to(config['model_device'])
		trainer = trainer_cls(env, policy, latent_model, opt, collector, losses=losses, steps_per_epoch=0, eval_dataset=trajs, **config['trainer']['params'], device=config['model_device'])

		eval_feature_wise_rmse = trainer.evaluate(train_dataset=False,plot_data=True,data_dir=os.path.join(config['experiment_fp'],"trajs"))
		print("Eval RMSE", eval_feature_wise_rmse.mean().item())
