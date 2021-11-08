import numpy as np
from typing import Any, Dict, List, Tuple, NoReturn


import argparse
import os

import pandas as pd
import json

from sklearn.metrics import silhouette_score 
from sklearn.metrics import calinski_harabasz_score 
from sklearn.metrics import davies_bouldin_score 

def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--data_dir",
		default="",
		type=str,
		help="Directory where the results of each algorithm (csv files) are saved",
	)	

	parser.add_argument(
		"--result_dir",
		default="",
		type=str,
		help="Directory where the result of the analysis (csv files) should be saved",
	)

	return parser.parse_args()

def discriptive_analysis(X:np.ndarray, Y:np.ndarray) -> Dict:

	
	'''
		compute descriptive statistics metrics of each cluster

		- metrics:
			- central tendency : mean
			- dispersion       : std
			- bounds           : min, max
			- quantile         : 0.25, 0.5, 0.75
		- params:
			- X : np.ndarray : [mean_velocity, mean_acceleration,
								mean_deceleratuin, std_lateral_jerk] (m, 4)
			- Y : np.ndarray : [driving_style] (m)
		- return:
			 result : dict : metrics
	'''

	result = {}

	for c in range(0,3):
		indexes = np.where(Y==c)[0]

		_mean = 0.0 if len(indexes) == 0 else np.mean(X[indexes], axis=0).tolist()
		_std  = 0.0 if len(indexes) == 0 else np.std(X[indexes], axis=0).tolist()
		_min  = 0.0 if len(indexes) == 0 else np.min(X[indexes], axis=0).tolist()
		_max  = 0.0 if len(indexes) == 0 else np.max(X[indexes], axis=0).tolist()
		_q25  = 0.0 if len(indexes) == 0 else np.quantile(X[indexes], 0.25, axis=0).tolist()
		_q50  = 0.0 if len(indexes) == 0 else np.quantile(X[indexes], 0.50, axis=0).tolist()
		_q75  = 0.0 if len(indexes) == 0 else np.quantile(X[indexes], 0.75, axis=0).tolist()

		result[f'{c}'] = {}
		result[f'{c}']['mean'] = _mean 
		result[f'{c}']['std']  = _std 
		result[f'{c}']['min']  = _min 
		result[f'{c}']['max']  = _max 
		result[f'{c}']['q25']  = _q25 
		result[f'{c}']['q50']  = _q50 
		result[f'{c}']['q75']  = _q75 

		#print (result)

	return result

def clustering_analysis(X:np.ndarray, Y:np.ndarray) -> Dict:

	'''
		compute clustering analysis evaluation metrics

		- metrics:
			- Silhouette Coefficient
			- Calinski Harabasz Score
			- Davis-Bouldin Index
		- params:
			- X : np.ndarray : [mean_velocity, mean_acceleration,
								mean_deceleratuin, std_lateral_jerk] (m, 4)
			- Y : np.ndarray : [driving_style] (m)
		- return:
			 result : dict : metric
	'''

	result = {}
	result['silhouette'] = 0
	result['calinski']   = 0
	result['davis']      = 0
	
	if len(np.unique(Y)) > 1:
		result['silhouette'] = silhouette_score(X=X, labels=Y)
		result['calinski']   = calinski_harabasz_score(X=X, labels=Y)
		result['davis']      = davies_bouldin_score(X=X, labels=Y)

	return result

if __name__== '__main__':

	'''
		- compute evaluation metrics for each algorithm
		- based on clustering analysis and descriptive statistics

		- metrics:

			- Silhouette Coefficient
				+ https://en.wikipedia.org/wiki/Silhouette_(clustering)
				+ https://www.sciencedirect.com/science/article/pii/0377042787901257
				+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics
					.silhouette_score.html#sklearn-metrics-silhouette-score
			- Calinski Harabasz Score
				+ https://medium.com/@haataa/how-to-measure-clustering-performances
					-when-there-are-no-ground-truth-db027e9a871c
				+ https://www.tandfonline.com/doi/abs/10.1080/03610927408827101
				+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics
					.calinski_harabasz_score.html#sklearn.metrics.calinski_harabasz_score
			- Davis-Bouldin Index
				+ https://en.wikipedia.org/wiki/Davies%E2%80%93Bouldin_index
				+ https://ieeexplore.ieee.org/document/4766909
				+ https://scikit-learn.org/stable/modules/generated/sklearn.metrics
					.davies_bouldin_score.html#sklearn.metrics.davies_bouldin_score
			- Mean, STD, min, max, Q-25, Q-50, Q-75

	'''

	
	args = parse_arguments()

	assert os.path.exists(args.data_dir),\
		f'[Clustering Analysis][main][ERROR] data_dir not found!({args.data_dir})'


	result_files = os.listdir(args.data_dir)
	
	assert len(result_files) > 0,\
		f'[Clustering Analysis][main][ERROR] no file found in data_dir directory!({args.data_dir})'


	for result_file in result_files:
		#results_fuzzy_cm_train_2s_ekf
		if not result_file.endswith(f'csv'):
			continue

		print (f'[Clustering Analysis][main] \033[92mprocessing file\033[0m: {result_file}')

		_split = result_file.split('_') 
		mode = _split[-3]
		obs_len = int(_split[-2][0])
		filter_name = _split[-1].split('.')[0]

		print (f'[Clustering Analysis][main] loading data...')

		# [mean_v, mean_acc, mean_deac, std_jy, y]
		data = pd.read_csv(os.path.join(args.data_dir, result_file))

		X = data[['mean_velocity', 'mean_acceleration', 'mean_deceleration', 'std_lateral_jerk']].values
		Y = np.asarray(data[['driving_style']].values, dtype=int)
		Y = np.squeeze(Y)

	
		print (f'[Clustering Analysis][main] computing metrics...')
		c_a_result = clustering_analysis(X=X, Y=Y)
		stats_result = discriptive_analysis(X=X, Y=Y)
		

		
		result = dict(c_a_result)
		result.update(stats_result)

		result_file_name = f'analysis_{mode}_{obs_len}s_{filter_name}.json'
		print (f'[Clustering Analysis][main] saving results ({result_file_name})')
		with open(os.path.join(args.result_dir, result_file_name), 'w') as f:
			json.dump(result, f)
		
	