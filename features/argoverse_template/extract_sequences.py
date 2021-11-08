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



from util.argoverse_manager import Argoverse



def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--data_dir",
		default="",
		type=str,
		help="Directory where the sequences (csv files) are saved",
	)
	
	parser.add_argument(
		"--feature_dir",
		default="",
		type=str,
		help="Directory where the extracted sequences are to be saved",
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
						help="Observed length of the trajectory in seconds")
	parser.add_argument("--filter",
						default='ekf',
						type=str,
						help="Filter to process the data noise. (ekf/none/ekf-savgol/savgol)",
						choices=['ekf', 'none', 'ekf-savgol', 'savgol'])

	return parser.parse_args()





if __name__ == '__main__':
	"""Load sequences and save the computed features."""

	print("[Extract Sequences] running...")

	start = time.time()

	args = parse_arguments()

	assert os.path.exists(args.data_dir), f'[Extract Sequences][main][ERROR] data_dir not founded! ({data_dir})'
	
	argoverse = Argoverse(root=args.data_dir)

	save_dir = args.feature_dir if os.path.exists(args.feature_dir) else os.getcwd()

	print ("[Extract Sequences] getting filtered trajectories")
	argoverse.get_filtered_trajectories(obs_len=args.obs_len,
										mode=args.mode,
										batch_size=args.batch_size,
										filter_name=args.filter,
										save_dir=save_dir)

	
	print('[Extract Sequences] time: {} minutes'.format((time.time() - start)/60.))

