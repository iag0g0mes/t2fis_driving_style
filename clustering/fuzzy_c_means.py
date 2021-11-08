import numpy as np
#np.set_printoptions(precision=2)

import pandas as pd

from typing import Any, Dict, List, Tuple, NoReturn

import argparse
import os
import pickle
import json

from util.fuzzy_cmeans import FuzzyCMeans

def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--data_dir",
		required=True,
		type=str,
		help="Directory where the features (npy files) are saved",
	)

	parser.add_argument(
		"--model_dir",
		required=True,
		type=str,
		help="Directory where the model is saved",
	)

	parser.add_argument(
		"--result_dir",
		required=True,
		type=str,
		help="Directory where the model is saved",
	)

	parser.add_argument("--mode",
						required=True,
						type=str,
						help="train/val/test",
						choices=['train', 'test', 'val'])

	parser.add_argument("--obs_len",
						default=2,
						type=int,
						help="Observed length of the trajectory in seconds",
						choices=[1,2,3,4,5])

	parser.add_argument("--filter",
						default='ekf',
						type=str,
						help="Filter to process the data noise. (ekf/none/ekf-savgol/savgol",
						choices=['ekf', 'none', 'ekf-savgol', 'savgol'])

	return parser.parse_args()


def train(data:np.ndarray,
		  obs_len:int,
		  filter_name:str,
		  model_dir:str,
		  result_dir:str,
		  save_model:bool=True)->NoReturn:
	
	print('[Fuzzy C-Means][train] creating model...')

	fuzzy_cm = FuzzyCMeans(n_clusters=3,
						   m=3,
						   max_iter=1000,
						   tol=1e-5,
						   random_state=7)

	print('[Fuzzy C-Means][train] training...')

	_y = fuzzy_cm.fit_predict(X=np.transpose(data))
	
	print(f'[Fuzzy C-Means][train] fpc:{fuzzy_cm.fpc}')

	print('[Fuzzy C-Means][train] params (center and covariance):')
	for i, c in zip(range(1, 4), fuzzy_cm.centers):
		print(f'\tc_{i}-> center: {c}')

	print('[Fuzzy C-Means][train] results:')
	_c, _l = np.unique(_y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')

	if save_model:
		model_file=f'fuzzy_cm_{obs_len}s_{filter_name}.json'
		print (f'[Fuzzy C-Means][train] saving model ({model_file})...')
		fuzzy_cm.save(file_name=os.path.join(model_dir, model_file))


	result_file = f'results_fuzzy_cm_train_{obs_len}s_{filter_name}.csv'
	print (f'[Fuzzy C-Means][train] saving results ({result_file})...')
	labels = ['mean_velocity', 
			  'mean_acceleration', 
			  'mean_deceleration', 
			  'std_lateral_jerk', 
			  'driving_style']

	result = np.concatenate((data, _y), axis=1)
	df = pd.DataFrame(data=result, columns=labels)
	df.to_csv(os.path.join(result_dir,result_file))



def process(data:np.ndarray,
			obs_len:int,
			filter_name:str,
			model_dir:str,
			result_dir:str,
			mode:str)->NoReturn:

	model_file=f'fuzzy_cm_{obs_len}s_{filter_name}.json'
	assert os.path.exists(os.path.join(model_dir, model_file)),\
		f'[Fuzzy C-Means][{mode}][ERROR] model not found! ({model_file})'

	print(f'[Fuzzy C-Means][{mode}] loading the model...')

	fuzzy_cm = FuzzyCMeans.load(file_name=os.path.join(model_dir, model_file))
	
	assert fuzzy_cm is not None,\
		f'[Fuzzy C-Means][{mode}][ERROR] error while loading model! ({model_file})'

	_y = fuzzy_cm.predict(X=np.transpose(data))

	print(f'[Fuzzy C-Means][{mode}] fpc:{fuzzy_cm.fpc}')

	print(f'[Fuzzy C-Means][{mode}] params (center):')
	for i, c in zip(range(1, 4), fuzzy_cm.centers):
		print(f'\tc_{i}-> center: {c}')

	print(f'[Fuzzy C-Means][{mode}] results:')
	_c, _l = np.unique(_y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')


	result_file = f'results_fuzzy_cm_{mode}_{obs_len}s_{filter_name}.csv'
	print (f'[Fuzzy C-Means][{mode}] saving results ({result_file})...')
	labels = ['mean_velocity', 
			  'mean_acceleration', 
			  'mean_deceleration', 
			  'std_lateral_jerk', 
			  'driving_style']

	result = np.concatenate((data, _y), axis=1)
	df = pd.DataFrame(data=result, columns=labels)
	df.to_csv(os.path.join(result_dir,result_file))


if __name__ == '__main__':

	'''
		apply Fuzzy C-Mean clustering to classify the data into
		driving styles (calm, moderate, aggresive)
	'''

	print ('[Fuzzy C-Means] running....') 

	args = parse_arguments()


	if args.mode == 'test':
		args.obs_len = 2
		
	assert os.path.exists(args.data_dir),\
		f'[Fuzzy C-Means][main][ERROR] data_dir not found!({args.data_dir})'

	data_file = 'features_{}_{}s_{}.npy'.format(args.mode,
												args.obs_len,
												args.filter)
	data_file = os.path.join(args.data_dir, data_file)

	assert os.path.exists(data_file),\
		f'[Fuzzy C-Means][main][ERROR] data_file not found!({data_file})'

	print ('[Fuzzy C-Means][main] loading dataset....')
	# (m, 4)
	# [mean_v, mean_acc, mean_deac, std_jy]
	data = np.load(os.path.join(args.data_dir,data_file))

	if args.mode == 'train':
		train(data=data,
			  save_model=True,
			  obs_len=args.obs_len,
			  filter_name=args.filter,
			  model_dir=args.model_dir,
			  result_dir=args.result_dir)

	elif args.mode == 'test':
		process(data=data,
			 obs_len=args.obs_len,
			 filter_name=args.filter,
			 model_dir=args.model_dir,
			 result_dir=args.result_dir,
			 mode='test')

	else:#val
		process(data=data,
			 obs_len=args.obs_len,
			 filter_name=args.filter,
			 model_dir=args.model_dir,
			 result_dir=args.result_dir,
			 mode='val')
