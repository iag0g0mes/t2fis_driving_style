import numpy as np

def OWA(X:np.ndarray, weights:np.ndarray) -> float:

	'''
		Yager's OWA
		Ordered Weighted Averaging
		- params
			- X : np.ndarray : set of values [n]
			- weights : np.ndarray : expert weights [n]
		- return:
			- value : float : result of the aggregation
	'''

	assert np.shape(X)[0] == np.shape(weights)[0],\
		('[Aggregation][OWA][ERROR]'
		 ' mismatch shape X:{} | weights:{}').format(np.shape(X),
		 											 np.shape(weights))

	assert np.sum(weights)== 1,\
		('[Aggregation][OWA][ERROR]'
		 ' sum of the weights vector must be 1. ({})'.format(np.sum(weights)))

	Xs = np.sort(X)[::-1]#descending order

	return np.sum(np.multiply(Xs,weights))

def OWA_T1(X:np.ndarray, kind:int=2) -> float:

	'''
		Ordegered Weight Averagins

		- Font:
		
		ZHOU, Shang-Ming et al. Type-1 OWA operators for 
			aggregating uncertain information with uncertain 
			weights induced by type-2 linguistic quantifiers. 
			Fuzzy Sets and Systems, v. 159, n. 24, p. 3281-3296, 
			2008.

	
		- params:
			- X : np.ndarray : input
			- kind : float : weights tendency
							 1 -> center values
							 2 -> higher values
							 3 -> lower values  
		- return:
			- value : float : result of the aggregation
	'''

	assert kind in [1,2,3],\
		('[Aggregation][OWA_T1][ERROR] kind must be either'
		 ' 1, or 2, or 3. (given:{})'.format(kind))

	#Equation 4
	def Q(r:float, a:float, b:float)->float:
		if r<a:
			return 0
		elif r>=a and r<=b:
			return (r-a)/(b-a)
		else:#r>b
			return 1


	if kind == 1:
		a,b = 0.3, 0.8
	elif kind == 2:
		a,b = 0., 0.5
	else: #kind 2
		a,b = 0.5, 1.

	n = np.shape(X)[0]

	w = np.array([Q(r=i/n, a=a, b=b) - Q(r=(i-1)/n, a=a, b=b)\
					for i in range(1, n+1)])

	Xs = np.sort(X)[::-1]#descending order

	return np.sum(np.multiply(Xs, w))

