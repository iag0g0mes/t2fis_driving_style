import numpy as np
import pickle
import json
import os

from typing import Any, Dict, List, Tuple, NoReturn, NewType

from skfuzzy.cluster import cmeans 
from skfuzzy.cluster import cmeans_predict



class FuzzyCMeans(object):
	"""
		object for fuzzy c-means clustering

		-> based on skfuzzy.cluster 
		-> functions: cmeans and cmeans_predict
		-> see: 
		https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html#cmeans
		https://pythonhosted.org/scikit-fuzzy/api/skfuzzy.cluster.html#cmeans-predict
	"""
	def __init__(self, 
				 n_clusters:int, 
				 m:float, 
				 max_iter:float,
				 random_state:int,
				 tol:float):
		
		self.params = {}
		#desired number of clusters
		self.params['n_clusters'] = n_clusters
		#array exponentiation applied to the membership function u_old 
		self.params['m'] = m
		#maximum number of iterations
		self.params['max_iter'] = max_iter
		#random seed
		self.params['random_state'] = random_state
		#stopping criterion
		self.params['tol'] = tol
		#cluster centers
		self.params['centers'] = None

		#results
		self.results = {}
		#final fuzzy c-partioned matrix
		self.results['u'] = None
		#initial guess at fuzzy c-partioned matrix
		self.results['u0'] = None
		#final euclidian distance matrix
		self.results['d']  = None
		#objective function history
		self.results['jm'] = None
		#number of iterations run
		self.results['p']  = None
		#final fuzzy partition coefficient
		self.results['fpc'] = None
 
	def fit_predict(self, X:np.ndarray)->np.ndarray:

		'''
			train and predict

			- params:
				- X : np.ndarray (S, N) : where S is the number of features,
											and N is the number of samples
			- return:
				- Y : np.ndarray (1, N) : output

		'''

		cntr, u, u0, d, jm, p, fpc = cmeans(data=X,
											c=self.params['n_clusters'],
											m=self.params['m'],
											error=self.params['tol'],
											maxiter=self.params['max_iter'],
											init=None,
											seed=self.params['random_state'])

		self.params['centers'] = cntr
		

		self.results['u'] = u
		self.results['u0'] = u0
		self.results['d'] = d 
		self.results['jm'] = jm 
		self.results['p'] = p 
		self.results['fpc'] = fpc

		y = np.transpose(np.argmax(u, axis=0))

		return np.expand_dims(y, axis=1)


	def predict(self, X:np.ndarray)->np.ndarray:
		'''
			predict

			- params:
				- X : np.ndarray (S, N) : where S is the number of features,
											and N is the number of samples
			- return:
				- Y : np.ndarray (1, N) : output

		'''

		assert not np.any(self.params['centers'] == None),\
			"[Fuzzy C-Means][predict][ERROR] the model has not yet been trained!"

		u, u0, d, jm, p, fpc = cmeans_predict(test_data=X,
											  cntr_trained=self.params['centers'],
											  m=self.params['m'],
											  error=self.params['tol'],
											  maxiter=self.params['max_iter'],
											  init=None)


		y = np.transpose(np.argmax(u, axis=0))

		self.results['u'] = u
		self.results['u0'] = u0
		self.results['d'] = d
		self.results['jm'] = jm
		self.results['p'] = p
		self.results['fpc'] = fpc

		return np.expand_dims(y, axis=1)

	@property
	def y(self) -> np.ndarray:
		if np.any(self.results['u'] == None):
			return None
		else:
			y = np.transpose(np.argmax(u, axis=0))
			return np.expand_dims(y, axis=1)

	@property
	def centers(self)->np.ndarray:
		return None if np.any(self.params['centers'] == None) else\
			self.params['centers']
	
	@property
	def fpc(self)->float:
		return 0 if self.results['fpc'] is None else\
			self.results['fpc']
	

	def save(self, file_name:str)->NoReturn:
		'''
			save the model into a json file

			-params:
				- file_name: str:


		'''
		_dict = self.params.copy()

		if _dict['centers'] is not None:
			_dict['centers'] = _dict['centers'].tolist()


		with open(file_name, 'w') as f:
			json.dump(_dict, f)

	@staticmethod
	def load(file_name:str)->object:
		'''
			load the model from a json file

			-params:
				-file_name : str : path where the json
					file is stored
		'''

		with open(file_name, 'r') as f:
			_dict = json.load(f)

		obj = FuzzyCMeans(n_clusters=_dict['n_clusters'],
						  m=_dict['m'],
						  max_iter=_dict['max_iter'],
						  random_state=_dict['random_state'],
						  tol=_dict['tol'],)

		obj.params = _dict.copy()
		obj.params['centers'] = np.array(obj.params['centers'])

		return obj

