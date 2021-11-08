import os

from typing import Any, Dict, List, Optional, Tuple, NoReturn

from joblib import Parallel, delayed
import shutil
import tempfile
import time
import pickle

import numpy as np 
import pandas as pd


from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


import matplotlib.pyplot as plt

from util.ekf_filter import EKF
from util.savgol_filter import SavitzkyGolov
from util.plain_features import PlainFeatures

class Argoverse(object):
	def __init__(self, root : str):


		"""	
			create the dataset manager object

			- params:
				- root : str : root path for the data
		"""

		self.root_path = root

		# - seq_list -> list with paths for each sequence file
		self.manager = ArgoverseForecastingLoader(self.root_path)

		self.ekf = EKF()
		self.savgol = SavitzkyGolov(window_length=5,
									poly=3)
		self.compute_features = PlainFeatures()

		

	def _get_trajectories_by_track_id(self, 
									  sequence:str,
									  track_id:str,
									  obs_len: int=2)->List[pd.DataFrame]:
		"""
			return a segment of trajectory for each vehicle 
				defined by its track_id and sequence file

			- params:
				- sequence : str : sequence file
				- track_id : str : id of the agent
				- obs_len  : int : length of the trajectory segment in seconds
			- return:
				- segments : List[pd.DataFrame] : list of trajectories with 
										length 'obs_len' in seconds 
		"""

		obs_len = obs_len*10 #10Hz

		df = self.manager.get(sequence).seq_df

		if track_id == 'AGENT':
			traj = df[df['OBJECT_TYPE']==track_id]
		else:
			traj = df[df['TRACK_ID']==track_id]

		traj = traj[['TIMESTAMP', 'X', 'Y']]
		#print(np.shape(traj))
		
		segments = []
		if traj.shape[0] >= obs_len:
			segments = [traj.iloc[i*obs_len:(i+1)*obs_len].values \
					for i in range(0, traj.shape[0]//obs_len)]

		return np.asarray(segments)

	def _get_trajectories_by_sequence(self, 
									  sequence:str,
									  target_name:str='AGENT',
									  obs_len:int=2)->List[pd.DataFrame]:

		"""	
			return all segments of trajectory with length
			obs_len (in seconds) from all traffic agents
			of a sequence

			- params:
				- sequence: str : name of the sequence (file)
				- obs_len : int : length of the segments (in seconds)
				- target_name  : str : 'AGENT' whether to return the path of the 
											agent of each sequence only
								  'ALL'   whether to return the path of all 
											vehicles of the sequence
			- return:
				- segments: np.ndarray : List with all segments 
							of trajectory with length 'obs_len'
		""" 

		assert target_name in ['AGENT', 'ALL'],\
			('[Argoverse Manager][_get_trajectories_by_sequence]'
			 ' unkown target_name ({})').format(target_name)

		def _get_segments(track_ids: List,
						  start_idx:int,
						  sequence:str, 
						  batch_size:int=10,
						  obs_len:int=2)->List:

			result = []
			for _v_id in track_ids[start_idx:start_idx+batch_size]:
				t = self._get_trajectories_by_track_id(sequence=sequence, 
													   obs_len=obs_len,
													   track_id=_v_id)

				if np.shape(t)[0]>0:
					result.append(t)
			return result
	

		# main function 

		_r = []
		if target_name=='ALL':
			track_ids = self.manager.get(sequence).track_id_list

			batch_size = max(1,round(len(track_ids)*0.5))
	 
			_r = Parallel(n_jobs=-2)(delayed(_get_segments)(
					track_ids=track_ids,
					start_idx=i,
					sequence=sequence,
					batch_size=batch_size,
					obs_len=obs_len
			) for i in range(0, len(track_ids), batch_size))

		else: #target_name == AGENT	
			_r = [[self._get_trajectories_by_track_id(sequence=sequence, 
												   obs_len=obs_len,
												   track_id=target_name)]]
		segments = []
		if np.shape(_r)[0] > 0:
			segments = [np.concatenate(p)\
						for p in _r \
						if np.shape(p)[0] > 0]

		#print(np.concatenate(segments).shape, np.shape(_r))
		return np.concatenate(segments) if np.shape(segments)[0] > 0 else []

	def _aux_get_traj_segments(self,
							   start_idx:int,
							   sequences:List,
							   temp_dir:str,
							   mode:str, 
							   target_name:str,
							   batch_size:int=10,
							   obs_len:int=2)->NoReturn:

		"""
			auxiliary function to get trajectories segments
			and save them into a temp folder
		"""
		result = []
		for seq in sequences[start_idx:start_idx+batch_size]:
			_seq = self._get_trajectories_by_sequence(sequence = seq,
													  obs_len = obs_len,
													  target_name = target_name) 

			if np.shape(_seq)[0]>0:
				result.append(_seq)

		if np.shape(result)[0]>0:		
			result = np.concatenate(result)
			print(('[Argoverse][get_trajectories][{}]'
				   ' saving sequence [{}/{}][{}]').format(target_name,
														  start_idx,
														  start_idx+batch_size, 
														  len(sequences)))

			_path = os.path.join(temp_dir,'trajectory_{}_{}_{}.npy'.format(mode, 
						start_idx, start_idx+batch_size))
			np.save(_path, result)
		else:
			print(('[Argoverse][get_trajectories][{}]'
				   ' empty sequence [{}/{}][{}]').format(target_name,
														 start_idx,
														 start_idx+batch_size,
														 len(sequences)))

	def get_trajectories(self, 
						 save_dir:str, 
						 mode:str,
						 batch_size:int,
						 merge_files:bool=True,
						 target_name:str='AGENT',
						 obs_len:int=2)->str:

		"""
			return all segments of trajectories with length 
				'obs_len' in seconds of the whole dataset

			- params:
				- obs_len : int : length in seconds
				- save_dir: str : path to save the result
				- mode    : str : train/test
				- merge_files: bool : True, if all files should be merger
						into a single file
									  False, otherwise
				- target_name  : str : 'AGENT' whether to return the path of the 
											agent of each sequence only
								  'ALL'   whether to return the path of all 
											vehicles of the sequence
			- return:
				- traj_path: path in which the trajectories
					files were saved
		"""

		assert target_name in ['AGENT', 'ALL'],\
			('[Argoverse Manager][get_trajectories]'
			 ' unkown target_name ({})').format(target_name)


		sequences = self.manager.seq_list#[:500]
		batch_size = min(batch_size, int(len(sequences)*0.2))

		temp_save_dir = tempfile.mkdtemp()

		# get trajectories and save them into a temp dir
		Parallel(n_jobs=-2)(delayed(self._aux_get_traj_segments)(
				start_idx=i,
				sequences=sequences,
				batch_size=batch_size,
				obs_len=obs_len,
				temp_dir=temp_save_dir,
				mode=mode,
				target_name=target_name
		) for i in range(0, len(sequences), batch_size))


		if merge_files:
			# merge all trajectory files saved into the temp dir
			# remove all temp files after the merge
			print('[Argoverse][get_trajectories] merging files...')
			self.merge_saved_features(temp_dir=temp_save_dir,
									  save_dir=save_dir,
									  mode=mode,
									  prefix=f'trajectory_{obs_len}s_{target_name}')
		else:
			# return the path in which trajectory segments files
			#  were stored
			return temp_save_dir


	def _savgol_traj(self, traj:np.ndarray)->np.ndarray:
		'''
			applied a Savitzky-Golov filter to the trajectory
			 - path, velocity, acceleration and jerk

			- params:
				traj : nd.array : path [[x,y]] (m,2)
			- return:
				filtered : nd.array : [[x,y,v_x, v_y, a_x, a_y, j_x, j_y]]
		'''
		self.savgol.set_window_size(traj.shape[0]//2)

		return self.savgol.process(traj)

	def _ekf_savgol_traj(self, traj:np.ndarray)->np.ndarray:
		'''
			applied a Savitzky-Golov filter to the trajectory after 
			compute features using ekf
			- path, velocity, acceleration and jerk

			- params:
				traj : nd.array : path [[x,y]] (m,2)
			- return:
				filtered : nd.array : [[x,y,v_x, v_y, a_x, a_y, j_x, j_y]]
		'''
		ekf_traj = self.ekf.process(traj)
		
		self.savgol.set_window_size(traj.shape[0]//2)
		
		return self.savgol.filter(ekf_traj)

	def _ekf_traj(self, traj:np.ndarray)->np.ndarray:
		'''
			compute features using ekf 
			 - path, velocity, acceleration and jerk

			- params:
				traj : nd.array : path [[x,y]] (m,2)
			- return:
				filtered : nd.array : [[x,y,v_x, v_y, a_x, a_y, j_x, j_y]]
		'''

		return self.ekf.process(traj)

	def _none_traj(self, traj:np.ndarray)->np.ndarray:
		'''
			compute features using the raw data
			 - path, velocity, acceleration and jerk

			- params:
				traj : nd.array : path [[x,y]] (m,2)
			- return:
				filtered : nd.array : [[x,y,v_x, v_y, a_x, a_y, j_x, j_y]]
		'''
		return self.compute_features.process(traj)

	def _aux_get_filtered_traj(self,
							   start_idx:int, 
							   trajectories:np.ndarray,
							   batch_size:int,
							   mode:str,
							   temp_dir:str,
							   index_file:int,
							   filter_name:str)->NoReturn:

		if filter_name == 'ekf':
			seq_traj = np.concatenate([self._ekf_traj(traj=p)\
					for p in trajectories[start_idx:start_idx+batch_size]])

		elif filter_name == 'savgol':
			seq_traj = np.concatenate([self._savgol_traj(traj=p)\
					for p in trajectories[start_idx:start_idx+batch_size]])

		elif filter_name == 'ekf-savgol':
			seq_traj = np.concatenate([self._ekf_savgol_traj(traj=p)\
					for p in trajectories[start_idx:start_idx+batch_size]])

		elif filter_name == 'none':
			seq_traj = np.concatenate([self._none_traj(traj=p)\
					for p in trajectories[start_idx:start_idx+batch_size]])

		else:
			assert False, ('[Argoverse Manager][get_filtered_trajectories]',
							' unknown filter_name ({})').format(filter_name)				

		print (("[Argoverse][get_filtered_trajectories][{}][{}]"
				" saving sequence [{}/{}][{}]").format(filter_name, 
													   index_file,
													   start_idx,
													   start_idx+batch_size, 
													   len(trajectories)))

		_path = os.path.join(temp_dir,\
				'{}_features_{}_{}_{}.npy'.format(index_file, mode, start_idx,\
				start_idx+batch_size))
		
		np.save(_path, seq_traj)	

	def get_filtered_trajectories(self,
								  mode:str, 
								  save_dir:str,
								  filter_name:str,
								  batch_size:int=100,
								  obs_len:int=2)->NoReturn:
		"""
			get all trajectories filtered by Extended Kalman Filter

			- params:
				mode : str : train/test
				save_dir : str : folder in which the result 
									should be stored
				filter_name: str : name of the filter to handle the data noise
				batch_size : int : size of the batch
				obs_len : int : length of each trajectory in seconds
		"""	

		if mode == 'test':
			obs_len = 2

		print ("[Argoverse][get_filtered_trajectories] getting trajectories...")
		traj_dir = self.get_trajectories(obs_len=obs_len,
										 save_dir=save_dir,
										 mode=mode,
										 merge_files=False,
										 batch_size=batch_size,
										 target_name='AGENT')
		
		
		temp_save_dir = tempfile.mkdtemp()
		index = 1
		num_files = len(os.listdir(traj_dir))

		print (("[Argoverse][get_filtered_trajectories] getting filtered"
				" trajectories [{}]...").format(num_files))

		for traj_file in os.listdir(traj_dir):
			if (not traj_file.endswith(".npy")) or\
				(mode not in traj_file) or\
				('trajectory' not in traj_file):
				continue

			trajectories = np.load(os.path.join(traj_dir,traj_file))

			batch_size = max(1, min(batch_size, int(trajectories.shape[0]*0.2)))

			print (("[Argoverse][get_filtered_trajectories][{}/{}] filtering"
					" trajectories...").format(index, num_files))
			
			Parallel(n_jobs=-2)(delayed(self._aux_get_filtered_traj)(
					start_idx=i,
					trajectories=trajectories,
					batch_size=batch_size,
					mode=mode,
					temp_dir=temp_save_dir,
					index_file=index,
					filter_name=filter_name
			) for i in range(0, trajectories.shape[0], batch_size))
			
			index = index + 1 


		print("[Argoverse][get_filtered_trajectories] merging files...")
		self.merge_saved_features(temp_dir=temp_save_dir,
								  save_dir=save_dir,
								  mode=mode,
								  prefix=f'filtered_{obs_len}s_{filter_name}',
								  delete_after_process=False)
		self.merge_saved_features(temp_dir=traj_dir,
								  save_dir=save_dir,
								  mode=mode,
								  prefix=f'trajectory_{obs_len}s',
								  delete_after_process=False)
		print("[Argoverse][get_filtered_trajectories] done!")




	def merge_saved_features(self, 
							 temp_dir:str, 
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

		file_name = 'sequences_{}_{}.npy'.format(mode, prefix)

		print (('[Argoverse][merge_saved_features]'
			   ' file: {} | shape {}').format(file_name,
			   								  np.shape(all_features)))
		
		_path = os.path.join(save_dir,file_name)
		np.save(_path, all_features)


# if __name__ == '__main__':
# 	start = time.time()
# 	print ("[Argoverse Manager] running...")
# 	argoverse = Argoverse('/home/iago/Documents/argoverse/data/prediction/train/data')
# 	save_dir = '/home/iago/Documents/workspace/driving_style/features'


# 	print ("[Argoverse Manager] getting filtered trajectories")
# 	argoverse.get_filtered_trajectories(obs_len=2,
# 										mode='train',
# 										save_dir=save_dir)

	
# 	print('[Argoverse Manager] time: {} minutes'.format((time.time() - start)/60.))
