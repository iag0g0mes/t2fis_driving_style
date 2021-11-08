import numpy as np
from typing import Any, Dict, List, Tuple, NoReturn


import argparse
import os

import pandas as pd
import json

import matplotlib.pyplot as plt

def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--data_file",
		default="",
		type=str,
		help="Path where the results of the algorithm (csv files) is saved",
	)	

	parser.add_argument(
		"--result_dir",
		default="",
		type=str,
		help="Directory where the result of the analysis (csv files) should be saved",
	)	
	parser.add_argument(
		"--algorithm",
		default="",
		type=str,
		help="name of the algorithm",
		choices=['kmeans', 'gmc', 'fuzzy_cm', 'fls_t1', 'fls_t2']
	)

	return parser.parse_args()

def plot_clusters(root_dir:str, X:np.ndarray, Y:np.ndarray) -> NoReturn:

	'''
		plot the clusters and classification
	'''

	colors = {0: 'tab:blue', 1:'tab:orange', 2:'tab:green'}
	titles = ['velocity', 'acceleration', 'deceleration', 'lateral jerk']

	fig, axs = plt.subplots(4,4)

	for i in range(0, 4):
		for j in range(0, 4):
			for c in [2,1,0]:
				i_c = np.where(Y==c)[0]

				if len(i_c) == 0:
					continue

				axs[i, j].scatter(X[i_c, i], X[i_c, j], s=1., color=colors[c])

			axs[i, j].set(xlabel=titles[j], ylabel=titles[i])


	for ax in axs.flat:
		ax.label_outer()
	
	fig.legend(['c_0', 'c_1', 'c_2'])
	plt.savefig(root_dir)
	




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

	assert os.path.exists(args.data_file),\
		f'[Clustering Analysis][main][ERROR] data_file not found!({args.data_file})'


	print (f'[Clustering Analysis][main] \033[92mprocessing file\033[0m: {args.data_file}')

	_split = args.data_file.split('_') 
	mode = _split[-3]
	obs_len = int(_split[-2][0])
	filter_name = _split[-1].split('.')[0]

	print (f'[Clustering Analysis][main] loading data...')

	# [mean_v, mean_acc, mean_deac, std_jy, y]
	data = pd.read_csv(args.data_file)

	X = data[['mean_velocity', 'mean_acceleration', 'mean_deceleration', 'std_lateral_jerk']].values
	Y = np.asarray(data[['driving_style']].values, dtype=int)
	Y = np.squeeze(Y)

	result_file = f'{args.algorithm}_{mode}_{obs_len}s_{filter_name}.png'
	result_file = os.path.join(args.result_dir, result_file)
	plot_clusters(root_dir=result_file, X=X, Y=Y)