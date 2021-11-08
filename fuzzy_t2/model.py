import os

import numpy as np 
import pandas as pd


from typing import Any, Dict, List, Optional, Tuple, NoReturn

import skfuzzy as fuzz
import skfuzzy.control as ctrl


from pyit2fls import IT2FS_Gaussian_UncertMean, IT2FS_Gaussian_UncertStd,\
					 IT2FLS, IT2FS_plot, IT2FS, gaussian_mf, trapezoid_mf,\
					 min_t_norm, max_s_norm,crisp



from aggregation import OWA_T1

import matplotlib.pyplot as plt



class FLST2Model(object):


	def __init__(self, rules_path:str, expert_mode:str):

		self.antecedent = {}
		self.consequent = {}

		self.expert_mode = expert_mode

		self.build_model()
		self.fuzz_inf = self.build_rules(rules_dir=rules_path)



	def build_model(self)->NoReturn:
		# ANTECENDENT
		self.antecedent = {}

		### Acceleration
		acc_trapmf = [[[0.1, 0. , 2. , 3.5 , 1.],[0.1, 0. , 1.5, 2.7, 0.8]],#sup, inf -> small
		              [[2. , 3.5, 5.5, 7. , 1.],[2.7, 4. , 5. , 6.2, 0.8]],#sup, inf -> medium
		              [[5.5, 7. , 10., 9.9, 1.],[6.2, 7.5, 10., 9.9, 0.8]]]#sup, inf -> large
		self.antecedent['Acceleration'] = self._create_trapmf_set(
							labels = ['small', 'medium', 'large'],
							universe = np.linspace(0, 10, 1001),
							params=np.asarray(acc_trapmf)
						)

		### Deceleration
		dec_trapmf = [[[0.1, 0. , 2. , 3.5 , 1.],[0.1, 0. , 1.5, 2.7, 0.8]],#sup, inf -> small
		              [[2. , 3.5, 5.5, 7. , 1.],[2.7, 4. , 5. , 6.2, 0.8]],#sup, inf -> medium
		              [[5.5, 7. , 10., 9.9, 1.],[6.2, 7.5, 10., 9.9, 0.8]]]#sup, inf -> large
		self.antecedent['Deceleration'] = self._create_trapmf_set(
							labels = ['small', 'medium', 'large'],
							universe = np.linspace(0, 10, 1001),
							params=np.asarray(dec_trapmf)
						)

		### Lateral Jerk
		lat_jk_trapmf = [[[0.1, 0. , 4. , 6.  , 1.],[0.1 , 0.  , 3., 4.5 , 0.8]],#sup, inf -> small
		                 [[3. , 6. , 10., 12.9 , 1.],[4.5 , 6.9 , 9.1, 11.4, 0.8]],#sup, inf -> medium
		                 [[10., 12., 16., 15.9, 1.],[11.5, 13., 16., 15.9, 0.8]]]#sup, inf -> large
		self.antecedent['LateralJerk'] = self._create_trapmf_set(
							labels = ['small', 'medium', 'large'],
							universe = np.linspace(0, 16, 1001),
							params= np.asarray(lat_jk_trapmf)
						)

		### Velocity
		vel_trapmf = [[[0.1, 0. , 15., 25. , 1.],[0.1 , 0. , 12., 20. , 0.8]],#sup, inf -> very_slow
		              [[15., 25., 35., 45. , 1.],[20. , 28., 32., 40. , 0.8]],#sup, inf -> slow
		              [[35., 45., 55., 65. , 1.],[40. , 48., 52., 60. , 0.8]],#sup, inf -> normal
		              [[55., 65 , 75., 85. , 1.],[60. , 68., 72., 80. , 0.8]],#sup, inf -> fast
		              [[75., 85., 100., 99.9, 1.],[80., 88., 100., 99.9, 0.8]]]#sup, inf -> very_fast
		self.antecedent['Velocity'] = self._create_trapmf_set(
							labels = ['very_slow', 'slow', 'normal', 'fast', 'very_fast'],
							universe = np.linspace(0, 100, 1001),
							params=np.asarray(vel_trapmf)
						)
		
		# CONSEQUENT
		### Behavior (Driving Style)
		behav_trapmf = [[[0.1, 0.0, 0.2,  0.4,  1.],[0.1 , 0.0,  0.15,  0.3, 0.8]],#sup, inf -> small
		                [[0.2, 0.4, 0.6,  0.8,  1.],[0.3, 0.45, 0.55,  0.7, 0.8]],#sup, inf -> medium
		                [[0.6, 0.8, 1.0,  0.99, 1.],[0.7, 0.85, 1.0,   0.99, 0.8]]]#sup, inf -> large
		self.consequent['DrivingStyle'] = self._create_trapmf_set(
							labels = ['calm', 'moderate', 'aggressive'],
							universe = np.linspace(0, 1, 1001),
							params= np.asarray(behav_trapmf)
						)

	def _create_trapmf_set(self, 
						   labels:List[str], 
						   universe:np.ndarray, 
						   params:np.ndarray) -> Dict:

		assert np.shape(labels)[0] == np.shape(params)[0],\
			('[Fuzzy Logic System T2 model][_create_trapmf_set][ERROR]'
			  ' size of labels must be equal to the size of params!'
			  ' ({}!={})').format(np.shape(labels)[0], np.shape(params)[0])

		fuzz_set = {}

		for l, p in zip(labels, params):
			fuzz_set[l] = \
				IT2FS(universe,
					  trapezoid_mf, p[0], #sup mf
					  trapezoid_mf, p[1]) #inf mf

		return fuzz_set

	def build_rules(self, rules_dir:str)->IT2FLS:

		assert os.path.exists(rules_dir),\
			('[Fuzzy Logic System T2 model][build_rules][ERROR]'
			 ' rules_dir not found!{}').format(rules_dir)

		rules_files = os.listdir(rules_dir)
		

		#build fuzz logic system (type 2)
		fuzz_inf = IT2FLS()
		fuzz_inf.add_input_variable('Velocity')
		fuzz_inf.add_input_variable('Acceleration')
		fuzz_inf.add_input_variable('Deceleration')
		fuzz_inf.add_input_variable('LateralJerk')
		fuzz_inf.add_output_variable('DrivingStyle')

		#get rules
		rules = None
		if self.expert_mode=='single':
			rules_files[0] = 'rules_0.csv'
			print('[Fuzzy Logic System T2 mode][build rules]', end='')
			print(f' single expert system! (rule:{rules_files[0]})')

			rules = self._single_expert_rules(os.path.join(rules_dir, rules_files[0]))
		elif self.expert_mode=='multiple':
			print('[Fuzzy Logic System - T2][build_rules]', end='')
			print(f' multiple expert system: (n_e: {len(rules_files)})')
			rules = self._multiple_expert_rules(rules_files, root_dir=rules_dir)
		else:
			assert False,\
				('[Fuzzy Logic System T2 model][build_rules][ERROR]'
				 ' expert_mode invalid! {}').format(self.expert_mode)

		assert rules is not None,\
			('[Fuzzy Logic System T2 model][build_rules][ERROR]'
			 ' error while building rules..')

		for rule in rules:
			fuzz_inf.add_rule(rule[0], rule[1])

		return fuzz_inf

	
	def _single_expert_rules(self, rule_file:str)->List:
		
		rules = pd.read_csv(rule_file)

		assert rules.shape[1] == 5,\
			('[Fuzzy Logic System T2 model][build_rules] wrong rule_file shape'
			 '{} != (m, 5)'.format(rules.shape))

		domain = {'calm':'calm',
				  'more_calm_than_moderate':'calm',
				  'between_calm_and_moderate':'moderate',
				  'more_moderate_than_calm':'moderate',
				  'moderate':'moderate',
				  'more_moderate_than_aggressive':'moderate',
				  'between_moderate_and_aggressive':'aggressive',
				  'more_aggressive_than_moderate':'aggressive',
				  'aggressive':'aggressive'}


		#self._check_rules(rules=rules)
		fuzz_rules = []
		for line in rules.iterrows():
			index, r = line[0], line[1]

			xs = domain[r['driving_style']]

			_in = [('Velocity',self.antecedent['Velocity'][r['velocity']]),
				   ('Acceleration',self.antecedent['Acceleration'][r['acceleration']]),
				   ('Deceleration',self.antecedent['Deceleration'][r['deceleration']]),
				   ('LateralJerk',self.antecedent['LateralJerk'][r['lateral_jerk']])]
			_out = [('DrivingStyle',self.consequent['DrivingStyle'][xs])]

			fuzz_rules.append([_in, _out])

		return fuzz_rules

	def _multiple_expert_function(self, label:str)->float:

		domain = {'calm':1,
				  'more_calm_than_moderate':2,
				  'between_calm_and_moderate':3,
				  'more_moderate_than_calm':4,
				  'moderate':5,
				  'more_moderate_than_aggressive':6,
				  'between_moderate_and_aggressive':7,
				  'more_aggressive_than_moderate':8,
				  'aggressive':9}

		return (1./9.)*domain[label]

	def _fuzz_driving_style(self, value=float)->Tuple:

		memb_value = []
		set_labels = []

		for label in self.consequent['DrivingStyle']:
			umf = trapezoid_mf(x=value, 
				params=self.consequent['DrivingStyle'][label].umf_params) 
			lmf = trapezoid_mf(x=value, 
				params=self.consequent['DrivingStyle'][label].lmf_params) 

			memb_value.append([umf, lmf])
			set_labels.append(label)

		return np.asarray(memb_value), np.asarray(set_labels)



	def _multiple_expert_rules(self, rules_files:List[str], root_dir:str)->NoReturn:
		
		rules = None

		#get rules
		decisions = []
		for rule_file in rules_files:
			_file = pd.read_csv(os.path.join(root_dir,rule_file))

			decisions.append(_file['driving_style'].values)

			rules = _file[['velocity', 'acceleration', 'deceleration', 'lateral_jerk']]

		decisions = np.asarray(decisions).T
		
		#aggregate decisions
		y = []
		for d in decisions:
			#print(d, end="")
			xs = np.array([self._multiple_expert_function(label=l) for l in d])
			value = OWA_T1(X=xs,kind=2)

			memb_value, set_labels = self._fuzz_driving_style(value=value)

			center_memb = np.array([(m[0]+m[1])/2. for m in memb_value])
			y.append(set_labels[np.argmax(center_memb)])


		#create rules
		fuzz_rules = []

		for line, _y in zip(rules.iterrows(), y):
			
			index, r = line[0], line[1]

			_in = [('Velocity',self.antecedent['Velocity'][r['velocity']]),
				   ('Acceleration',self.antecedent['Acceleration'][r['acceleration']]),
				   ('Deceleration',self.antecedent['Deceleration'][r['deceleration']]),
				   ('LateralJerk',self.antecedent['LateralJerk'][r['lateral_jerk']])]
			_out = [('DrivingStyle',self.consequent['DrivingStyle'][_y])]

			fuzz_rules.append([_in, _out])

		return fuzz_rules



	def inference(self, observation:Dict) -> Dict:

		"""
			perform inference at the fuzzy system
		"""

		vel = observation['velocity']*3.6 #m/s -> Km/h
		acc = observation['acceleration'] #m/s^2
		dec = observation['deceleration'] #m/s^2
		ljk = observation['lateral_jerk'] #std (m/s^3)


		_in = {}
		_in['Velocity'] = vel
		_in['Acceleration'] = acc
		_in['Deceleration'] = dec
		_in['LateralJerk'] = ljk

		it2out, tr = self.fuzz_inf.evaluate(_in, 
											min_t_norm,
											max_s_norm,
											np.linspace(0,1,1001))


		_c = crisp(tr['DrivingStyle'])		

		y = []
		l = []

		y, l = self._fuzz_driving_style(value=_c)

		result = {}
		result['class'] = np.argmax([(m[0]+m[1])/2. for m in y])
		result['crisp'] = _c
		result['membership_values'] = np.asarray(y)
		result['set_labels']=np.asarray(l)
		return result


	def plot(self, root_path:str)-> NoReturn:

		IT2FS_plot(self.antecedent['Velocity']['very_slow'],
				   self.antecedent['Velocity']['slow'],
				   self.antecedent['Velocity']['normal'],
				   self.antecedent['Velocity']['fast'],
				   self.antecedent['Velocity']['very_fast'],
				   title='Velocity',
				   legends=['Very Slow' , 'Slow', 'Normal', 'Fast', 'Very Fast'],
				   filename=os.path.join(root_path,'antecedent_velocity.png'))

		IT2FS_plot(self.antecedent['Acceleration']['small'],
				   self.antecedent['Acceleration']['medium'],
				   self.antecedent['Acceleration']['large'],
				   title='Acceleration',
				   legends=['Small', 'Medium', 'Large'],
				   filename=os.path.join(root_path, 'antecedent_acceleration.png'))
		
		IT2FS_plot(self.antecedent['Deceleration']['small'],
				   self.antecedent['Deceleration']['medium'],
				   self.antecedent['Deceleration']['large'],
				   title='Deceleration',
				   legends=['Small', 'Medium', 'Large'],
				   filename=os.path.join(root_path, 'antecedent_deceleration.png'))


		IT2FS_plot(self.antecedent['LateralJerk']['small'],
				   self.antecedent['LateralJerk']['medium'],
				   self.antecedent['LateralJerk']['large'],
				   title='LateralJerk',
				   legends=['Small', 'Medium', 'Large'],
				   filename=os.path.join(root_path, 'antecedent_lateral_std_jerk.png'))


		IT2FS_plot(self.consequent['DrivingStyle']['calm'],
				   self.consequent['DrivingStyle']['moderate'],
				   self.consequent['DrivingStyle']['aggressive'],
				   title='Driving Style',
				   legends=['Calm', 'Moderate', 'Aggressive'],
				   filename=os.path.join(root_path,'consequent_driving_stye.png'))


