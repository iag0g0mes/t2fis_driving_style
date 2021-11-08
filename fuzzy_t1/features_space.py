import numpy as np
import pandas as pd
from typing import Any, Dict, List, Tuple, NoReturn
import argparse
import os

def parse_arguments() -> Any:
	"""Parse command line arguments."""

	parser = argparse.ArgumentParser()
	
	parser.add_argument(
		"--data_file",
		required=True,
		type=str,
		help="Path for the feature file",
	)

	return parser.parse_args()



def discretize(X:np.ndarray, size:int)->Tuple:
	
	return pd.cut(X, size, retbins=True)[1]



if __name__ == '__main__':

	print('[Features Space] running...')

	args = parse_arguments()

	assert os.path.exists(args.data_file),\
		f'[Features Space][main][ERROR] data_file not found!({args.datafile}'

	print('[Features Space] loading dataset...')

	data = np.load(args.data_file)

	print(f'[Features Space] data shape: {data.shape}')

	features = [('Mean Velocity', 5),
				('Mean Acceleration', 3),
				('Mean Deceleration', 3),
				('STD Lateral Jerk', 3)]

	for i, f in zip(range(0, 4), features):
		d = discretize(X=data[:, i], size=f[1])
		print (f'{f[0]}:')
		print(f'\t bins:{d}')
		q = np.quantile(data[:, i], [0.25, 0.4, 0.6, 0.75])
		print(f'\t quantiles [0.25, 0.4, 0.6, 0.75]:{q}')
	

