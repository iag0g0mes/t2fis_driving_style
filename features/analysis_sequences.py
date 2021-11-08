import numpy as np
from typing import Any, Dict, List, Tuple, NoReturn

def stats(traj:np.ndarray) -> NoReturn:

	#central tendency : mean
	#dispersion       : std
	#bounds           : min max
	#quantile         : 0.25, 0.5, 0.75

	labels = ['vel', 'acc', 'jerk']

	for t, l in zip(traj, labels):
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

	_filters = ['none', 'ekf', 'savgol', 'ekf-savgol']
	_modes = ['train', 'val', 'test', 'sample']

	seg = 5 #2
	mode = _modes[3]
	filter_name = _filters[3]
	data_file = ('/home/iago/Documents/workspace/driving_style/'
				 'features/{}/sequences/sequences_{}_filtered_{}s_{}.npy').format(mode,
				 													   mode,
				 													   seg,
				 													   filter_name)

	print ('[Analysis] mode:{} | filter:{} | obs_len:{}'.format(mode, 
															    filter_name, 
															    seg))

	print ('[Analysis] loading dataset....')
	# (m, 8)
	# [x, y, vx, vy, ax, ay, jx, jy]
	data = np.load(data_file)
	data = np.squeeze(data)
	print ('[Analysis] loaded dataset shape:{}'.format(data.shape))

	all_data = np.concatenate(data)
	print('[Analysis] data shape:{}'.format(all_data.shape))
	
	del data

	vel = np.sqrt(np.power(all_data[:,2], 2) + np.power(all_data[:,3],2))
	acc = np.sqrt(np.power(all_data[:,4], 2) + np.power(all_data[:,5],2))
	jerk = np.sqrt(np.power(all_data[:,6], 2) + np.power(all_data[:,7],2))

	del all_data

	print ('[Analysis] stats (vel, acc, jerk):')
	stats([vel, acc, jerk])





