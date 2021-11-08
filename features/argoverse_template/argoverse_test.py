import os

from typing import Any, Dict, List, Optional, Tuple, NoReturn

from joblib import Parallel, delayed
import shutil
import tempfile
import time

import numpy as np 
import pandas as pd


from argoverse.data_loading.argoverse_forecasting_loader import ArgoverseForecastingLoader


import matplotlib.pyplot as plt

from util.ekf_filter import EKF

from util.plain_features import PlainFeatures

from util.savgol_filter import SavitzkyGolov

np.set_printoptions(precision=2)

class Argoverse(object):


	def __init__(self, root : str):


		"""	
			create the dataset manager object

			- params:
				- root : str : root path for the data
		"""

		self.root_path = root


		self.manager = ArgoverseForecastingLoader(self.root_path)


		self.ekf = EKF()
		self.feature = PlainFeatures()
		self.savgol = SavitzkyGolov(window_length=11,
									poly=3)
		
		all_vel_raw = []
		all_vel_ekf = []
		all_vel_savgol = []
		all_vel_ekf_savgol = []
		
		all_acc_raw = []
		all_acc_ekf = []
		all_acc_savgol = []
		all_acc_ekf_savgol = []
		
		all_j_raw = []
		all_j_ekf = []
		all_j_savgol = []
		all_j_ekf_savgol = []

		for index in range(0, len(self.manager.seq_list)):
			print(index)

			seq_path = self.manager.seq_list[index]
			df = self.manager.get(seq_path).seq_df

			traj = df[df['OBJECT_TYPE']=='AGENT']
			traj = traj[['TIMESTAMP', 'X', 'Y']].values
			self.savgol.set_window_size(traj.shape[0]//2)
			
			#features
			plain_features = self.feature.process(traj=traj)
			ekf_features = self.ekf.process(traj=traj)
			savgol_features = self.savgol.process(traj=traj)
			ekf_savgol_features = self.savgol.filter(vector=ekf_features)

			plain_features = np.squeeze(plain_features)
			ekf_features = np.squeeze(ekf_features)
			savgol_features = np.squeeze(savgol_features)
			ekf_savgol_features = np.squeeze(ekf_savgol_features)
		

			#path
			plt.figure(1)
			plt.plot(plain_features[:,0],plain_features[:,1], color='blue', label='raw', marker='.')
			plt.plot(ekf_features[:,0],ekf_features[:,1], color='green', label='ekf', marker='.')
			plt.plot(savgol_features[:,0],savgol_features[:,1], color='orange', label='savgol', marker='.')
			plt.plot(ekf_savgol_features[:,0],ekf_savgol_features[:,1], color='red', label='ekf-savgol', marker='.')
			plt.legend()
			plt.savefig(f'path_{index}.png')
			plt.clf()
			
			#velocity
			vel_plain = np.sqrt(np.power(plain_features[:,2],2) + np.power(plain_features[:,3],2))
			vel_ekf = np.sqrt(np.power(ekf_features[:,2],2) + np.power(ekf_features[:,3],2))
			vel_savgol = np.sqrt(np.power(savgol_features[:,2],2) + np.power(savgol_features[:,3],2))
			vel_ekf_savgol = np.sqrt(np.power(ekf_savgol_features[:,2],2) + np.power(ekf_savgol_features[:,3],2))

			plt.figure(2)
			plt.plot(vel_plain, color='blue', label='raw', marker='.')
			plt.plot(vel_ekf, color='green', label='ekf', marker='.')
			plt.plot(vel_savgol, color='orange', label='savgol', marker='.')
			plt.plot(vel_ekf_savgol, color='red', label='ekf-savgol', marker='.')
			plt.legend()
			plt.savefig(f'vel_{index}.png')
			plt.clf()

			acc_plain = np.sqrt(np.power(plain_features[:,4],2) + np.power(plain_features[:,5],2))
			acc_ekf = np.sqrt(np.power(ekf_features[:,4],2) + np.power(ekf_features[:,5],2))
			acc_savgol = np.sqrt(np.power(savgol_features[:,4],2) + np.power(savgol_features[:,5],2))
			acc_ekf_savgol = np.sqrt(np.power(ekf_savgol_features[:,4],2) + np.power(ekf_savgol_features[:,5],2))

			plt.figure(3)
			plt.plot(acc_plain, color='blue', label='raw', marker='.')
			plt.plot(acc_ekf, color='green', label='ekf', marker='.')
			plt.plot(acc_savgol, color='orange', label='savgol', marker='.')
			plt.plot(acc_ekf_savgol, color='red', label='ekf-savgol', marker='.')
			plt.legend()
			plt.savefig(f'acc_{index}.png')
			plt.clf()


			j_plain = np.sqrt(np.power(plain_features[:,6],2) + np.power(plain_features[:,7],2))
			j_ekf = np.sqrt(np.power(ekf_features[:,6],2) + np.power(ekf_features[:,7],2))
			j_savgol = np.sqrt(np.power(savgol_features[:,6],2) + np.power(savgol_features[:,7],2))
			j_ekf_savgol = np.sqrt(np.power(ekf_savgol_features[:,6],2) + np.power(ekf_savgol_features[:,7],2))

			plt.figure(3)
			plt.plot(j_plain, color='blue', label='raw', marker='.')
			plt.plot(j_ekf, color='green', label='ekf', marker='.')
			plt.plot(j_savgol, color='orange', label='savgol', marker='.')
			plt.plot(j_ekf_savgol, color='red', label='ekf-savgol', marker='.')
			plt.legend()
			plt.savefig(f'j_{index}.png')
			plt.clf()

			all_vel_raw.append(vel_plain)
			all_vel_ekf.append(vel_ekf)
			all_vel_savgol.append(vel_savgol)
			all_vel_ekf_savgol.append(vel_ekf_savgol)
			
			all_acc_raw.append(acc_plain)
			all_acc_ekf.append(acc_ekf)
			all_acc_savgol.append(acc_savgol)
			all_acc_ekf_savgol.append(acc_ekf_savgol)
			
			all_j_raw.append(j_plain)
			all_j_ekf.append(j_ekf)
			all_j_savgol.append(j_savgol)
			all_j_ekf_savgol.append(j_ekf_savgol)

		all_vel_raw = np.concatenate(all_vel_raw)
		all_vel_ekf = np.concatenate(all_vel_ekf)
		all_vel_savgol = np.concatenate(all_vel_savgol)
		all_vel_ekf_savgol = np.concatenate(all_vel_ekf_savgol)
		all_acc_raw = np.concatenate(all_acc_raw)
		all_acc_ekf = np.concatenate(all_acc_ekf)
		all_acc_savgol = np.concatenate(all_acc_savgol)
		all_acc_ekf_savgol = np.concatenate(all_acc_ekf_savgol)
		all_j_raw = np.concatenate(all_j_raw)
		all_j_ekf = np.concatenate(all_j_ekf)
		all_j_savgol = np.concatenate(all_j_savgol)
		all_j_ekf_savgol = np.concatenate(all_j_ekf_savgol)

		print('\033[92m PLAIN \033[0m')
		self.stats(traj=[all_vel_raw, all_acc_raw, all_j_raw])
		print('\033[92m EKF \033[0m')
		self.stats(traj=[all_vel_ekf, all_acc_ekf, all_j_ekf])
		print('\033[92m SAVGOL \033[0m')
		self.stats(traj=[all_vel_savgol, all_acc_savgol, all_j_savgol])
		print('\033[92m EKF-SAVGOL \033[0m')
		self.stats(traj=[all_vel_ekf_savgol, all_acc_ekf_savgol, all_j_ekf_savgol])

		plt.figure(1)
		plt.boxplot([all_vel_raw,all_vel_ekf,all_vel_savgol,all_vel_ekf_savgol], labels=['raw','ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('box_plot_vel.png')
		plt.clf()

		plt.figure(2)
		plt.boxplot([all_acc_raw,all_acc_ekf,all_acc_savgol,all_acc_ekf_savgol], labels=['raw','ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('box_plot_acc.png')
		plt.clf()

		plt.figure(3)
		plt.boxplot([all_j_raw,all_j_ekf,all_j_savgol,all_j_ekf_savgol], labels=['raw','ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('box_plot_j.png')
		plt.clf()

		plt.figure(1)
		plt.boxplot([all_vel_ekf,all_vel_savgol,all_vel_ekf_savgol], labels=['ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('2box_plot_vel.png')
		plt.clf()

		plt.figure(2)
		plt.boxplot([all_acc_ekf,all_acc_savgol,all_acc_ekf_savgol], labels=['ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('2box_plot_acc.png')
		plt.clf()

		plt.figure(3)
		plt.boxplot([all_j_ekf,all_j_savgol,all_j_ekf_savgol], labels=['ekf', 'savgol', 'ekf-savgol'])
		plt.savefig('2box_plot_j.png')
		plt.clf()
		
	def stats(self, traj:np.ndarray) -> NoReturn:

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


			#plt.show()

if __name__ == '__main__':
	argoverse = Argoverse('/home/iago/Documents/argoverse/data/prediction/sample/data')
