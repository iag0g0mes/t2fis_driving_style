import numpy as np
from typing import Any, Dict, List, Tuple, NoReturn


import argparse
import os

def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--data_dir",
		default="",
		type=str,
		help="Directory where the features (npy files) are saved",
	)

	parser.add_argument("--mode",
						required=True,
						type=str,
						help="train/val/test/sample",
						choices=['train', 'test', 'val','sample'])

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

def stats(traj:np.ndarray) -> NoReturn:

	#central tendency : mean
	#dispersion       : std
	#bounds           : min max
	#quantile         : 0.25, 0.5, 0.75

	labels = ['mean_v', 'mean_acc', 'mean_deac', 'std_jy']

	for i, l in zip(range(0, traj.shape[1]), labels):
		t = traj[:, i]
		_mean = round(np.mean(t),2)
		_std  = round(np.std(t),2)
		_min  = round(np.min(t),2)
		_max  = round(np.max(t),2)
		_q25  = round(np.quantile(t, 0.25),2)
		_q50  = round(np.quantile(t, 0.5),2)
		_q75  = round(np.quantile(t, 0.75),2)

		print (f'Feature: {l}')
		print ('\tmean:{} | std:{} | min:{} | max:{} | q25:{} | q50:{} | q75:{}'.format(_mean,
				_std, _min, _max, _q25, _q50, _q75))



if __name__== '__main__':

	#_filters = ['none', 'ekf', 'savgol', 'ekf-savgol']
	#_modes = ['train', 'val', 'test', 'sample']
	#_obs_len = [2,5]

	#seg = _obs_len[0]
	#mode = _modes[3]
	#filter_name = _filters[0]
	
	args = parse_arguments()

	if args.mode == 'test':
		args.obs_len = 2
		
	assert os.path.exists(args.data_dir),\
		f'[Analysis][main][ERROR] data_dir not found!({args.data_dir})'

	data_file = 'features_{}_{}s_{}.npy'.format(args.mode,
				 								args.obs_len,
				 								args.filter)

	assert os.path.exists(os.path.join(args.data_dir, data_file)),\
		f'[Analysis][main][ERROR] data_file not found!({data_file})'

	print ('[Analysis] loading dataset....')
	# (m, 4)
	# [mean_v, mean_acc, mean_deac, std_jy]
	data = np.load(os.path.join(args.data_dir,data_file))


	print ('[Analysis] mode:{} | filter:{} | obs_len:{}'.format(args.mode, 
															    args.filter, 
															    args.obs_len))
	print ('[Analysis] data shape:{}'.format(data.shape))

	print ('[Analysis] stats:')
	stats(data)





