from this import s
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
from wheeledsim_rl.trainers.experiment_caching import Experiment

from wheeledsim_rl.networks.str_to_cls import str_to_cls as network_str_to_cls
from wheeledsim_rl.trainers.str_to_cls import str_to_cls as trainer_str_to_cls
from wheeledsim_rl.losses.str_to_cls import str_to_cls as loss_str_to_cls

from rosbag_to_dataset.util.dataloader_util import DataLoaderUtil, BackgroundLoader

import argparse
from datetime import datetime
import os

if __name__ == '__main__':
	
	
	parser = argparse.ArgumentParser()
	parser.add_argument('--eval_config', type=str, required=True, help='Config for evaluation')
	args = parser.parse_args()
	config = yaml.load(open(args.eval_config, 'r'), Loader=yaml.FullLoader)
	now = datetime.now()
	config["experiment_fp"] = os.path.join(
            config["experiment_fp"], f'{now.strftime("%m-%d-%Y,%H-%M-%S")}')
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = config["output_device"]+","+config["device_ids"]
	print(config['model_fp'])
	latent_model = torch.load(config['model_fp'])
	print(latent_model.insize['state'])

	EnvDataLoaderObj = DataLoaderUtil(config["train_framelistfile"] , config["train_fp"], config, batch_size = 1, num_workers = 0, shuffle= False)
	print(f"\nloading environment")

	env_traj = BackgroundLoader(EnvDataLoaderObj,1)[0]
	env = DummyEnv(env_traj)
	noise = ouNoise(lowerlimit=env.action_space.low, upperlimit=env.action_space.high, var = np.array([0.01, 0.01]), offset=np.zeros([2, ]), damp=np.ones([2, ])*1e-4)
	policy = OUPolicy(noise)
	train_buf = None
	print('\nloading eval data...')
	EvalDataLoaderObj = DataLoaderUtil(config["eval_framelistfile"] , config["eval_fp"], config, batch_size = config["loader"]['eval']['batch_size'], shuffle = config["loader"]['eval']['shuffle'], num_workers = config["loader"]['eval']['num_workers'], persistent_workers = config["loader"]['eval']['persistent_workers'])

	eval_num_batches = EvalDataLoaderObj.calc_num_batches(config["loader"]['eval']['buffer_capacity'])
	if config['loader']['eval']['entire_data']:
		eval_num_batches = None
	traj_list_eval = BackgroundLoader(EvalDataLoaderObj,eval_num_batches)
	# print(traj_list_eval[0]['observation']['state'].shape)
	print(isinstance(traj_list_eval, list))

	if not isinstance(traj_list_eval, list):
		trajs = split_trajs(traj_list_eval.sample_idxs(torch.arange(len(traj_list_eval))))
	else:
		trajs = traj_list_eval

	
	print('\nloading Model...')
	losses = {}
	if config['modalities'] is not None:
		for name, info in config['modalities'].items():
			topic = info['topic']
			loss_params = info['loss']
			loss_cls = loss_str_to_cls[loss_params['type']]
			loss = loss_cls(**loss_params['params'])
			losses[topic] = loss

	latent_model = torch.load(config['model_fp'])
	latent_model = latent_model.to(config['model_device'])

	opt = optim.Adam(latent_model.parameters(), lr=config['lr'])
	
	collector = Collector(env, policy=policy)
	trainer_cls = trainer_str_to_cls[config['eval']['type']]
	latent_model = nn.DataParallel(latent_model,output_device = 0,device_ids=list(map(int,config["device_ids"].split(","))))

	latent_model = latent_model.to(config['model_device'])
	latent_model.module.input_normalizer = latent_model.module.input_normalizer.to("cpu")

	#change collecter and training buf to model so it can be converted

	trainer = trainer_cls(env, policy, latent_model, opt, train_buf,collector, losses=losses, steps_per_epoch=0, eval_dataset=trajs, **config['eval']['params'], device="cpu")
	print(trainer.n_eval_steps)
	print("helloo")
	experiment = Experiment(trainer, config['name'], experiment_filepath=config['experiment_fp'], save_logs_every=1, save_every=config['save_every'], buffer_cycle=1, config = config)

	maybe_mkdir(os.path.join(config['experiment_fp'], config['name']), force=False)
	with open(os.path.join(config['experiment_fp'], config['name'], '_config.yaml'), 'w') as fp:
		yaml.dump(config, fp)

	cur_epoch = 0
	for e in range(config['eval']['params']['epochs']):
		eval_feature_wise_rmse = trainer.evaluate(train_dataset=False)
		trainer.logger.record_item("Eval RMSE Features", np.around(eval_feature_wise_rmse.cpu().numpy(), 4), prefix="Performance")
		trainer.logger.record_item("Eval RMSE", eval_feature_wise_rmse.mean().item(), prefix="Performance")
		# self.logger.record_item("Train RMSE", train_feature_wise_rmse.mean().item(), prefix="Performance")

		trainer.logger.record_item("Return Mean", -eval_feature_wise_rmse.mean().detach().item(), prefix="Performance")

		trainer.avg_losses = {}

		trainer.log()
		traj_list_eval = BackgroundLoader(EvalDataLoaderObj,eval_num_batches)
