import numpy as np

from scipy.signal import savgol_filter

class SavitzkyGolov(object):

	def __init__(self, window_length:int, poly:int):

		self.size=window_length
		self.poly=poly

	def set_window_size(self, size):
		self.size=size

		if self.size%2==0:
			self.size = self.size+1

	def process(self, traj:np.ndarray)->np.ndarray:
		
		x = traj[:,1]
		y = traj[:,2]

		result = []

		for i in range(0, 4):
			_x = savgol_filter(x=x, 
							   window_length=self.size, 
							   polyorder=self.poly, 
							   deriv=i,
							   delta=0.1)
			_y = savgol_filter(x=y, 
							   window_length=self.size, 
							   polyorder=self.poly, 
							   deriv=i,
							   delta=0.1)
			result.append(_x)
			result.append(_y)

		return np.dstack(result)

	def filter(self, vector:np.ndarray)->np.ndarray:
		vector = np.squeeze(vector)

		x = vector[:,0]
		y = vector[:,1]

		p = np.dstack((np.zeros(len(x)), x, y))
		p = np.squeeze(p)

		return self.process(p)

	def filter2(self, vector:np.ndarray)->np.ndarray:
		vector = np.squeeze(vector)

		result = [savgol_filter(x=vector[:, i], 
						  window_length=self.size, 
						  polyorder=self.poly)\
				 for i in range(0, vector.shape[1])]

		return np.dstack(result)