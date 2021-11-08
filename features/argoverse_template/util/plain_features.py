import numpy as np


class PlainFeatures(object):

	def __init__(self):
		pass

	def get_velocity(self, traj:np.ndarray)->np.ndarray:
		
		x = traj[:, 1]
		y = traj[:, 2]
		t = traj[:, 0]

		vel_x = np.ediff1d(x)/np.ediff1d(t)
		vel_y = np.ediff1d(y)/np.ediff1d(t)
		

		vel = np.sqrt(np.power(vel_x,2) + np.power(vel_y,2))

		return vel_x, vel_y, vel

	def get_acceleration(self, traj:np.ndarray)->np.ndarray:
		
		vel_x, vel_y, _ = self.get_velocity(traj)

		#0,1 -> v0
		#1,2 -> v1, a0
		t = traj[1:,0]

		acc_x = np.ediff1d(vel_x)/np.ediff1d(t)
		acc_y = np.ediff1d(vel_y)/np.ediff1d(t)
		

		acc = np.sqrt(np.power(acc_x,2) + np.power(acc_y,2))

		return acc_x, acc_y, acc


	def get_jerk(self, traj:np.ndarray)->np.ndarray:
		
		acc_x, acc_y, _ = self.get_acceleration(traj)

		#0,1 -> v0
		#1,2 -> v1, a0
		#2,3 -> v2, a1, j0
		t = traj[2:,0]

		j_x= np.ediff1d(acc_x)/np.ediff1d(t)
		j_y= np.ediff1d(acc_y)/np.ediff1d(t)

		j = np.sqrt(np.power(j_x,2) + np.power(j_y,2))

		return j_x, j_y, j


	def process(self, traj:np.ndarray) -> np.ndarray:
		
		v_x, v_y, v = self.get_velocity(traj=traj)
		a_x, a_y, a = self.get_acceleration(traj=traj)
		j_x, j_y, j = self.get_jerk(traj=traj)
		x = traj[:,1]
		y = traj[:,2]
		
		result = np.dstack((x[3:],
							y[3:],
							v_x[2:],
							v_y[2:],
							a_x[1:],
							a_y[1:],
							j_x,
							j_y))

		return np.asarray([result])