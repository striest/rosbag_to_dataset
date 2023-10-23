from wheeledsim_rl.util.util import dict_to, DummyEnv
import torch
import random

import numpy as np

import argparse
import os
import yaml
import time
from datetime import datetime


from torch import nn, optim

from wheeledsim_rl.replaybuffers.dict_replaybuffer import NStepDictReplayBuffer
from wheeledsim_rl.util.util import dict_to, DummyEnv
from wheeledsim_rl.util.rl_util import split_trajs
from wheeledsim_rl.util.os_util import maybe_mkdir
from wheeledsim_rl.util.normalizer import ObservationNormalizer
from wheeledsim_rl.util.ouNoise import ouNoise
from wheeledsim_rl.policies.ou_policy import OUPolicy
from wheeledsim_rl.collectors.base_collector import Collector
from wheeledsim_rl.trainers.experiment_less_mem import Experiment

from wheeledsim_rl.networks.str_to_cls import str_to_cls as network_str_to_cls
from wheeledsim_rl.trainers.str_to_cls import str_to_cls as trainer_str_to_cls
from wheeledsim_rl.losses.str_to_cls import str_to_cls as loss_str_to_cls

from rosbag_to_dataset.util.dataloader_util import DataLoaderUtil

import argparse
from datetime import datetime
import os
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

	EnvDataLoaderObj = DataLoaderUtil(config["train_framelistfile"] , config["train_fp"], config, batch_size = 1, num_workers = 0, shuffle= False)
	print('\nloading environment')
	env_traj = EnvDataLoaderObj.convert_obs_to_traj(num_batches=1)[0]
	env = DummyEnv(env_traj)

	noise = ouNoise(lowerlimit=env.action_space.low, upperlimit=env.action_space.high, var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
	policy = OUPolicy(noise)
	
	TrainDataLoaderObj = DataLoaderUtil(config["train_framelistfile"] , config["train_fp"], config, batch_size = 64, num_workers = 4, persistent_workers=True,shuffle= False)

	print('\nloading train data...')
	traj_list_train = TrainDataLoaderObj.convert_obs_to_traj(num_batches=4)
	train_buf = NStepDictReplayBuffer(env, capacity=3000)

	for traj in traj_list_train:
		try:
			train_buf.insert(traj)
		except Exception as e:
			print(e)
			import pdb;pdb.set_trace()
	
	EvalDataLoaderObj = DataLoaderUtil(config["eval_framelistfile"] , config["eval_fp"], config, batch_size = 64, num_workers = 4, persistent_workers=True,shuffle= False)

	print('\nloading eval data...')
	traj_list_eval = EvalDataLoaderObj.convert_obs_to_traj(num_batches=4)

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

	latent_model = latent_model.to(config['model_device'])
	latent_model.module.input_normalizer = latent_model.module.input_normalizer.to("cpu")

	#change collecter and training buf to model so it can be converted

	trainer = trainer_cls(env, policy, latent_model, opt, train_buf,collector, losses=losses, steps_per_epoch=0, eval_dataset=trajs, **config['trainer']['params'], device="cpu")
	config['experiment_fp'] = config['experiment_fp']+"/epoch10"
	experiment = Experiment(trainer, config['name'], experiment_filepath=config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], train_data_loader=TrainDataLoaderObj, buffer_cycle=1)

	maybe_mkdir(os.path.join(config['experiment_fp'], config['name']), force=False)
	with open(os.path.join(config['experiment_fp'], config['name'], '_config.yaml'), 'w') as fp:
		yaml.dump(config, fp)

	if True:
		experiment.run()
		trainer.total_epochs = 5000
		config['experiment_fp'] = config['experiment_fp'][:-2]+"5000"
		experiment = Experiment(trainer, config['name'], experiment_filepath=config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], train_data_loader=TrainDataLoaderObj, buffer_cycle=1)

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
