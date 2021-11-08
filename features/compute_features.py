import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, NoReturn

import argparse
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pickle as pkl

from util.features import ComputeFeatures




def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_dir",
		default="",
		type=str,
		help="Directory where the sequences (npy files) are saved",
	)
	
	parser.add_argument(
		"--feature_dir",
		default="",
		type=str,
		help="Directory where the computed features are to be saved",
	)
	
	parser.add_argument("--mode",
						required=True,
						type=str,
						help="train/val/test/sample",
						choices=['train', 'test', 'val','sample'])

	parser.add_argument(
		"--batch_size",
		default=100,
		type=int,
		help="Batch size for parallel computation",
	)
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



def compute_and_save_features(sequences:np.ndarray,
							  save_dir:str,
							  obs_len:int,
							  batch_size:int,
							  mode:str,
							  filter_name:str) -> NoReturn:


	print (("[Compute Features][compute_and_save_features]"
		f" sequences size: {np.shape(sequences)}"))


	def _aux_compute_features(sequences:np.ndarray,
							  start_idx:int, 
							  batch_size:int,
							  manager:ComputeFeatures,
							  mode:str,
							  tmp_dir:str,
							  filter_name:str,
							  obs_len=int):

		_r = [manager.process(d) for d in\
				sequences[start_idx:start_idx+batch_size]]

		_r = np.asarray(_r)

		_path = (f'features_{mode}_{obs_len}s_{start_idx}'
		 		 f'_{start_idx+batch_size}_{filter_name}.npy')

		print ((f"[Compute Features][compute_and_save_features][{mode}][{filter_name}]"
				f" saving features [{start_idx}/{start_idx+batch_size}]"
				f"[{np.shape(sequences)[0]}]"))
		np.save(os.path.join(tmp_dir, _path), _r)




	batch_size = max(1, min(batch_size, int(np.shape(sequences)[0]*0.2)))
	tmp_dir = tempfile.mkdtemp()
	compute_features = ComputeFeatures()


	Parallel(n_jobs=-2)(delayed(_aux_compute_features)(
					sequences=sequences,
					start_idx=i,
					batch_size=batch_size,
					manager=compute_features,
					mode=mode,
					tmp_dir=tmp_dir,
					filter_name=filter_name,
					obs_len=obs_len
	) for i in range(0, np.shape(sequences)[0], batch_size))

	merge_saved_features(temp_dir = tmp_dir,
		  				 save_dir = save_dir,
		  				 mode = mode,
		  				 prefix = f'{obs_len}s_{filter_name}',
		  				 delete_after_process=False)


def merge_saved_features(temp_dir:str, 
						 save_dir:str, 
						 mode:str,
						 prefix:str,
						 delete_after_process:bool=True)->NoReturn:

	"""
		merge files (.npy) into a single file (.npy)

		- params:
			- temp_dir : str : path where the files were stored
			- save_dir : str : path where the result shoudl be saved
			- mode     : str : train/test
			- prefix   : str : name for the result file
			- delete_after_process : bool : true -> delete unit file after process
											false -> otherwise
	"""
	all_features = []

	for feature_file in os.listdir(temp_dir):
		if (not feature_file.endswith(".npy")) or\
			(mode not in feature_file):
			continue

		_p_file = os.path.join(temp_dir,feature_file)
		all_features.append(np.load(_p_file))
		if delete_after_process:
			os.remove(_p_file)	

	all_features = np.squeeze(np.concatenate(all_features))

	file_name = 'features_{}_{}.npy'.format(mode, prefix)

	print (('[Compute Features][merge_saved_features]'
		   ' file: {} | shape {}').format(file_name,
		   								  np.shape(all_features)))
	
	_path = os.path.join(save_dir,file_name)
	np.save(_path, all_features)





if __name__ == '__main__':
	"""Load sequences and save the computed features."""

	print("[Compute Features] running...")

	start = time.time()

	args = parse_arguments()

	assert os.path.exists(args.data_dir),\
		f'[Compute Features][main][ERROR] data_dir not founded! ({data_dir})'
	save_dir = args.feature_dir if os.path.exists(args.feature_dir) else os.getcwd()

	if args.mode == "test":
		args.obs_len = 2
	
	print("[Compute Features] loading sequences...")
	sequence_path = os.path.join(args.data_dir, 
		f'sequences_{args.mode}_filtered_{args.obs_len}s_{args.filter}.npy')

	assert os.path.exists(sequence_path),\
		f'[Compute Features][main][ERROR] data file not founded! ({sequence_path})'
	
	sequences = np.load(sequence_path)
	sequences = np.squeeze(sequences) 


	print("[Compute Features] computing features...")
	compute_and_save_features(sequences=sequences,
							  save_dir = save_dir,
							  obs_len  = args.obs_len,
							  mode     = args.mode,
							  batch_size  = args.batch_size,
							  filter_name = args.filter)

	print('[Compute Features] time: {} minutes'.format((time.time() - start)/60.))

