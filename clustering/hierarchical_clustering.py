import numpy as np
#np.set_printoptions(precision=2)

import pandas as pd

from typing import Any, Dict, List, Tuple, NoReturn

import argparse
import os
import pickle

from sklearn.cluster import AgglomerativeClustering

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


def process(data:np.ndarray,
		    obs_len:int,
		    filter_name:str,
		    model_dir:str,
		    result_dir:str,
		    mode:str)->NoReturn:

	print(f'[Agglomerative Clustering][{mode}] creating model...')
	hierarchical = AgglomerativeClustering(n_clusters=3,
										   affinity='euclidean',
										   linkage='ward')

	num_data = min(60000, data.shape[0])
	indices = np.random.choice(np.arange(0, data.shape[0]), num_data, replace=False)
	data = data[indices, :]

	print(f'[Agglomerative Clustering][{mode}] learning...')
	_y = hierarchical.fit_predict(X=data)
	_y = np.expand_dims(_y, axis=1)

	print(f'[Agglomerative Clustering][{mode}] results:')
	_c, _l = np.unique(_y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')

	
	result_file = f'results_hierarchical_{mode}_{obs_len}s_{filter_name}.csv'
	print (f'[Agglomerative Clustering][{mode}] saving results ({result_file})...')
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
		apply K-means clustering to classify the data into
		driving styles (calm, moderate, aggresive)
	'''

	print ('[Agglomerative Clustering] running....') 

	args = parse_arguments()


	if args.mode == 'test':
		args.obs_len = 2
		
	assert os.path.exists(args.data_dir),\
		f'[Agglomerative Clustering][main][ERROR] data_dir not found!({args.data_dir})'

	data_file = 'features_{}_{}s_{}.npy'.format(args.mode,
				 								args.obs_len,
				 								args.filter)
	data_file = os.path.join(args.data_dir, data_file)

	assert os.path.exists(data_file),\
		f'[Agglomerative Clustering][main][ERROR] data_file not found!({data_file})'

	print ('[Agglomerative Clustering] loading dataset....')
	# (m, 4)
	# [mean_v, mean_acc, mean_deac, std_jy]
	data = np.load(os.path.join(args.data_dir,data_file))

	if args.mode == 'train':
		process(data=data,
			  obs_len=args.obs_len,
			  filter_name=args.filter,
			  model_dir=args.model_dir,
			  result_dir=args.result_dir,
			  mode='train')

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
