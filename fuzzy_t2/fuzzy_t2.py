import numpy as np 
import os
import pandas as pd
import argparse

from typing import Any, Dict, List, Optional, Tuple, NoReturn

from model import FLST2Model

from tqdm import tqdm


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
		"--rules_dir",
		required=True,
		type=str,
		help="Directory where the FLS' rules are saved",
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

	parser.add_argument("--expert_mode",
						required=True,
						type=str,
						help="single expert (rule_0) or multiple experts (rule_i, i=0,1,...,m)",
						choices=['single', 'multiple'])

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


def process(model:FLST2Model,
			data:np.ndarray,
		    obs_len:int,
		    filter_name:str,
		    result_dir:str,
		    mode:str)->NoReturn:

	y = []
	obs={}


	for d in tqdm(data):
		obs['velocity'] = d[0]
		obs['acceleration'] = d[1] 
		obs['deceleration'] = d[2]
		obs['lateral_jerk'] = d[3]

		r = model.inference(observation=obs)
		y.append(r['class'])

	y = np.asarray(y)

	print(f'[Fuzzy Logic System - T2][{mode}] results:')
	_c, _l = np.unique(y, return_counts=True)
	for i, c in zip(_c,_l):
		print (f'\tc_{i}: {c}')

	
	result_file = f'results_fuzzyT2_{mode}_{obs_len}s_{filter_name}.csv'
	print (f'[Fuzzy Logic System - T2][{mode}] saving results ({result_file})...')
	labels = ['mean_velocity', 
			  'mean_acceleration', 
			  'mean_deceleration', 
			  'std_lateral_jerk', 
			  'driving_style']

	y = np.expand_dims(y, axis=1)
	result = np.concatenate((data, y), axis=1)
	df = pd.DataFrame(data=result, columns=labels)
	df.to_csv(os.path.join(result_dir,result_file))




if __name__ == '__main__':


	print ('[Fuzzy Logic System - T2] running....') 

	args = parse_arguments()

	if args.mode == 'test':
		args.obs_len = 2
		
	assert os.path.exists(args.data_dir),\
		f'[Fuzzy Logic System - T2][main][ERROR] data_dir not found!({args.data_dir})'

	data_file = 'features_{}_{}s_{}.npy'.format(args.mode,
				 								args.obs_len,
				 								args.filter)
	data_file = os.path.join(args.data_dir, data_file)

	assert os.path.exists(data_file),\
		f'[Fuzzy Logic System - T2][main][ERROR] data_file not found!({data_file})'


	assert os.path.exists(args.rules_dir),\
		f'[Fuzzy Logic System - T2][main][ERROR] rules_dir not found!({args.rules_dir})'


	print ('[Fuzzy Logic System - T2] loading dataset....')
	# [mean_v, mean_acc, mean_deac, std_jy] (m, 4)
	data = np.load(os.path.join(args.data_dir,data_file))


	print('[Fuzzy Logic System - T2] bulding Fuzzy Logic System - T2 model...')
	fuzz = FLST2Model(rules_path=args.rules_dir, expert_mode=args.expert_mode)
	
	print('[Fuzzy Logic System - T2] processing data...')
	process(model=fuzz,
			data=data,
		    obs_len=args.obs_len,
		    filter_name=args.filter,
		    result_dir=args.result_dir,
		    mode=args.mode)
