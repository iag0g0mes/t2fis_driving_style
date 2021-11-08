import numpy as np
#np.set_printoptions(precision=2)

import pandas as pd

from typing import Any, Dict, List, Tuple, NoReturn

import argparse
import os
import pickle

from sklearn.cluster import KMeans

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
	
	print('[k-means][train] creating kmeans model...')

	kmeans = KMeans(n_clusters=3,
					init='k-means++',
					max_iter=1000,
					tol=1e-5,
					random_state=7,
					algorithm="elkan")

	print('[k-means][train] training...')

	_y = kmeans.fit_predict(X=data)
	_y = np.expand_dims(_y, axis=1)

	print('[k-means][train] clusters:')
	for i, c in zip(range(1, 4), kmeans.cluster_centers_):
		print (f'\tc_{i}: {c}')

	print('[k-means][train] results:')
	_c, _l = np.unique(_y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')

	if save_model:
		model_file=f'kmeans_{obs_len}s_{filter_name}.pkl'
		print (f'[k-means][train] saving model ({model_file})...')
		with open(os.path.join(model_dir, model_file), 'wb') as f:
			pickle.dump(kmeans, f)


	result_file = f'results_kmeans_train_{obs_len}s_{filter_name}.csv'
	print (f'[k-means][train] saving results ({result_file})...')
	labels = ['mean_velocity', 
			  'mean_acceleration', 
			  'mean_deceleration', 
			  'std_lateral_jerk', 
			  'driving_style']

	result = np.concatenate((data, _y), axis=1)
	df = pd.DataFrame(data=result, columns=labels)
	df.to_csv(os.path.join(result_dir,result_file))

	result_file = result_file.replace('results', 'centers').replace('csv', 'txt')
	print (f'[k-means][train] saving results ({result_file})...')
	np.savetxt(os.path.join(result_dir, result_file), 
			   kmeans.cluster_centers_, 
			   fmt='%.8f',
			   delimiter=',')


def process(data:np.ndarray,
		    obs_len:int,
		    filter_name:str,
		    model_dir:str,
		    result_dir:str,
		    mode:str)->NoReturn:

	model_file=f'kmeans_{obs_len}s_{filter_name}.pkl'
	assert os.path.exists(os.path.join(model_dir, model_file)),\
		f'[K-Means][{mode}][ERROR] model not found! ({model_file})'

	print(f'[k-means][{mode}] loading the model...')
	kmeans = None
	with open(os.path.join(model_dir, model_file), 'rb') as f:
			kmeans = pickle.load(f)
	
	assert kmeans is not None,\
		f'[K-Means][{mode}][ERROR] error while loading model! ({model_file})'

	_y = kmeans.predict(X=data)
	_y = np.expand_dims(_y, axis=1)

	print(f'[k-means][{mode}] clusters:')
	for i, c in zip(range(1, 4), kmeans.cluster_centers_):
		print (f'\tc_{i}: {c}')

	print(f'[k-means][{mode}] results:')
	_c, _l = np.unique(_y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')

	
	result_file = f'results_kmeans_{mode}_{obs_len}s_{filter_name}.csv'
	print (f'[k-means][{mode}] saving results ({result_file})...')
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

	print ('[K-means] running....') 

	args = parse_arguments()


	if args.mode == 'test':
		args.obs_len = 2
		
	assert os.path.exists(args.data_dir),\
		f'[K-means][main][ERROR] data_dir not found!({args.data_dir})'

	data_file = 'features_{}_{}s_{}.npy'.format(args.mode,
				 								args.obs_len,
				 								args.filter)
	data_file = os.path.join(args.data_dir, data_file)

	assert os.path.exists(data_file),\
		f'[K-means][main][ERROR] data_file not found!({data_file})'

	print ('[K-means] loading dataset....')
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
