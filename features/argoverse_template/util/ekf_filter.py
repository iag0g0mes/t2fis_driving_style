import numpy as np 
import pandas as pd

import time
from typing import Any, Dict, List, Tuple, NoReturn


class EKF(object):


	def __init__(self):

		self.init()

	def init(self):
		#state vector
		self.X = np.array([[0],    # x
						   [0],    # y
						   [0],    # v_x
						   [0],    # v_y
						   [0],    # a_x
						   [0],    # a_y
						   [0],    # j_x
						   [0]])   # j_y
		#identity matrix
		self.I = np.eye(8)
		#process covariance  
		self.P = 1000*self.I
		#jacobian h(x) 
		self.H = np.array([[1., 0., 0., 0., 0., 0., 0., 0.],
						   [0., 1., 0., 0., 0., 0., 0., 0.]])
		#covariance noise process
		coeficient_Q = np.array([0.1, # x
								 0.1, # y
								 0.5, # v_x
								 0.5, # v_y
								 0.1, # a_x
								 0.1, # a_y
								 0.5, # j_x
								 0.5])# j_y
		self.Q = np.eye(8)*coeficient_Q
		#covariance noise observation
		self.R = np.array([[1., 0.],  # x_obs
						   [0., 1.]]) # y_obs

	def _update(self, z:np.array) -> np.ndarray:

		z = z.reshape(2,1) # x_obs, y_obs

		# estimate innovation
		y = z - np.dot(self.H, self.X)
		# estimate innovation covariance 
		S = np.dot(np.dot(self.H, self.P), np.transpose(self.H)) + self.R
		# estimate the near-optiman kalman gain
		K = np.dot(np.dot(self.P, np.transpose(self.H)), np.linalg.inv(S))
		# update state
		self.X = self.X + np.dot(K, y)
		# update process covariance
		self.P = np.dot((self.I - np.dot(K, self.H)), self.P)


		return self.X.reshape(8)

	def _predict(self, dt:float):

		"""
			A) prediction step
			  - Equations:

			x = xo + vx*dt
			y = yo + vy*dt
			v_x = vo_x + ax*t
			v_y = vo_y + ay*t
			a_x = (v_x - vo_x)/dt
			a_y = (v_y - vo_y)/dt
			j_x = (a_x - ao_x)/dt
			j_y = (a_y - ao_y)/dt
		"""
		# # 		# x   y  v_x v_y a_x  a_y j_x j_y
		# jac_X = [[1., 0., dt, 0., 0., 0., 0., 0.], # x
		# 		 [0., 1., 0., dt, 0., 0., 0., 0.], # y
		# 		 [0., 0., 1., 0., dt, 0., 0., 0.], # v_x
		# 		 [0., 0., 0., 1., 0., dt, 0., 0.], # v_y
		# 		 [0., 0., 1./dt, 0., 0., 0., 0., 0.], # a_x
		# 		 [0., 0., 0., 1./dt, 0., 0., 0., 0.], # a_y
		# 		 [0., 0., 0., 0., 1./dt, 0., 0., 0.], # j_x
		# 		 [0., 0., 0., 0., 0., 1./dt, 0., 0.]] # j_y

				# x   y  v_x v_y a_x  a_y j_x j_y
		jac_X = [[1., 0., dt, 0., 0., 0., 0., 0.], # x
				 [0., 1., 0., dt, 0., 0., 0., 0.], # y
				 [1./dt, 0., 0., 0., 0., 0., 0., 0.], # v_x
				 [0., 1./dt, 0., 0., 0., 0., 0., 0.], # v_y
				 [0., 0., 1./dt, 0., 0., 0., 0., 0.], # a_x
				 [0., 0., 0., 1./dt, 0., 0., 0., 0.], # a_y
				 [0., 0., 0., 0., 1./dt, 0., 0., 0.], # j_x
				 [0., 0., 0., 0., 0., 1./dt, 0., 0.]] # j_y

		# estimate P (process covariance) (without control input U)
		self.P = np.dot(np.dot(jac_X, self.P), np.transpose(jac_X)) 

		# estimate new state vector X (prediction)
		_x = self.X[0] + self.X[2]*dt
		_y = self.X[1] + self.X[3]*dt
		_v_x = (_x - self.X[0])/dt#self.X[2] + self.X[4]*dt
		_v_y = (_y - self.X[1])/dt#self.X[3] + self.X[5]*dt
		_a_x = (_v_x - self.X[2])/dt
		_a_y = (_v_y - self.X[3])/dt
		_j_x = (_a_x - self.X[4])/dt
		_j_y = (_a_y - self.X[5])/dt

		self.X = np.array([_x, 
						   _y, 
						   _v_x,
						   _v_y,
						   _a_x,
						   _a_y,
						   _j_x,
						   _j_y]).reshape((8,1))
		
	# def _predict2(self, dt:float):

	# 	"""
	# 		A) prediction step
	# 		  - Equations:

	# 		x = xo + vx*dt + (axt^2)/2
	# 		y = yo + vy*dt + (ayt^2)/2
	# 		v_x = vo_x + ax*t
	# 		v_y = vo_y + ay*t
	# 		a_x = (v_x - vo_x)/dt
	# 		a_y = (v_y - vo_y)/dt
	# 		j_x = (a_x - ao_x)/dt
	# 		j_y = (a_y - ao_y)/dt
	# 	"""
	# 			# x   y  v_x    v_y      a_x        a_y     j_x  j_y
	# 	jac_X = [[1., 0., dt,    0., (dt*dt)/2.,     0.    , 0., 0.], # x
	# 			 [0., 1., 0.,    dt,     0.,     (dt*dt)/2., 0., 0.], # y
	# 			 [0., 0., 1.,    0.,     dt,         0.    , 0., 0.], # v_x
	# 			 [0., 0., 0.,    1.,     0.,         dt,     0., 0.], # v_y
	# 			 [0., 0., 1./dt, 0.,     0.,         0.,     0., 0.], # a_x
	# 			 [0., 0., 0.,  1./dt,    0.,         0.,     0., 0.], # a_y
	# 			 [0., 0., 0.,    0.,   1./dt,        0.,     0., 0.], # j_x
	# 			 [0., 0., 0.,    0.,     0.,       1./dt,    0., 0.]] # j_y

	# 	# estimate P (process covariance) (without control input U)
	# 	self.P = np.dot(np.dot(jac_X, self.P), np.transpose(jac_X)) 

	# 	# estimate new state vector X (prediction)
	# 	_x = self.X[0,0] + self.X[2,0]*dt + (self.X[4,0]*dt*dt)/2.
	# 	_y = self.X[1,0] + self.X[3,0]*dt + (self.X[5,0]*dt*dt)/2.
	# 	_v_x = self.X[2,0] + self.X[4,0]*dt
	# 	_v_y = self.X[3,0] + self.X[5,0]*dt
	# 	_a_x = (_v_x - self.X[2,0])/dt
	# 	_a_y = (_v_y - self.X[3,0])/dt
	# 	_j_x = (_a_x - self.X[4,0])/dt
	# 	_j_y = (_a_y - self.X[5,0])/dt

	# 	self.X = np.array([_x, 
	# 					   _y, 
	# 					   _v_x,
	# 					   _v_y,
	# 					   _a_x,
	# 					   _a_y,
	# 					   _j_x,
	# 					   _j_y]).reshape((8,1))

	def clean(self)->NoReturn:
		self.init()

	def process(self, traj:np.ndarray) -> np.ndarray:
		
		self.X[0] = traj[0, 1] #x
		self.X[1] = traj[0, 2] #y
		self.X[2] = (traj[1, 1] - traj[0,1])/(traj[1, 0] - traj[0,0]) #vx
		self.X[3] = (traj[1, 2] - traj[0,2])/(traj[1, 0] - traj[0,0]) #vy
		self.X[4] = np.random.randint(-2,2,1)
		self.X[5] = np.random.randint(-2,2,1)

		last_t = traj[0,0]

		result = []

		for tj in traj[1:,:]:
			dt = tj[0]  - last_t
			self._predict(dt = dt)
			x = self._update(z=tj[1:])
			result.append(x)
			last_t = tj[0]

		return np.asarray([result])


