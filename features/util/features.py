import numpy as np


from typing import Any, Dict, List, Tuple, NoReturn


class ComputeFeatures(object):

	def __init__(self):

		pass

	def get_mean_velocity(self, traj:np.ndarray) -> float:
		'''
			compute velocity

			- params: 
				- traj : np.ndarray : [x,y, vx, vy, ax, ay, jx, jy]
		'''

		vel = np.sqrt(np.power(traj[:,2], 2) + np.power(traj[:,3],2))

		return vel.mean()

	def get_mean_acc_deac(self, traj:np.ndarray) -> Tuple:
		
		'''
			compute mean acceleration (+) and deaceleration (-)

			- params: 
				- traj : np.ndarray : [x,y, vx, vy, ax, ay, jx, jy]
		'''
		vel = np.sqrt(np.power(traj[:,2], 2) + np.power(traj[:,3],2))
		acc = np.sqrt(np.power(traj[:,4], 2) + np.power(traj[:,5],2))

		diff_vel = np.ediff1d(vel)

		acc_pos = acc[np.where(diff_vel >= 0)[0] + 1]
		acc_neg = acc[np.where(diff_vel < 0)[0] + 1]

		acc_pos_mean = acc_pos.mean() if acc_pos.shape[0]>0 else 0.0
		acc_neg_mean = acc_neg.mean() if acc_neg.shape[0]>0 else 0.0
		
		return acc_pos_mean, acc_neg_mean


	def get_std_lat_jerk(self, traj:np.ndarray) -> float:
		'''
			compute standard deviation of the lateral jerk

			- params:
				- traj : np.ndarray : [x, y, vx, vy, ax, ay, jx, jy]
			- return
				- std_jy : float
		'''

		return traj[:, 7].std()

	def process(self, sequence: np.ndarray) -> np.ndarray:

		'''
			return the computed features

			- params:
				sequence: np.ndarray : [x,y,vx, vy, ax, ay, jx, jy] (m, 8)
			- return:
				features: np.ndarray : [mean_v, mean_acc, mean_deac, std_jy]
		'''
		
		mean_v = self.get_mean_velocity(traj=sequence)
		mean_acc, mean_deac = self.get_mean_acc_deac(traj=sequence)
		std_jy = self.get_std_lat_jerk(traj=sequence)

		return np.array([mean_v, mean_acc, mean_deac, std_jy])