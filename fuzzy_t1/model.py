import os

import numpy as np 
import pandas as pd


from typing import Any, Dict, List, Optional, Tuple, NoReturn

import skfuzzy as fuzz
import skfuzzy.control as ctrl

from aggregation import OWA_T1

import matplotlib.pyplot as plt



class FLST1Model(object):


	def __init__(self, rules_path:str, expert_mode:str):

		self.antecedent = {}
		self.consequent = {}
		self.expert_mode = expert_mode
		self.build_model()
		self.system = self.build_rules(rules_dir=rules_path)
		self.fuzz_inf = ctrl.ControlSystemSimulation(self.system, 
													 flush_after_run=10)


	def build_model(self)->NoReturn:
		# ANTECENDENT
		self.antecedent = {}

		### Acceleration
		self.antecedent['Acceleration'] = ctrl.Antecedent(universe=np.linspace(0,10, 11), 
			                                                   label='Acceleration')
		self.antecedent['Acceleration']['small'] = \
				fuzz.trapmf(self.antecedent['Acceleration'].universe, [0., 0., 3., 4.])
		self.antecedent['Acceleration']['medium'] = \
				fuzz.trapmf(self.antecedent['Acceleration'].universe, [3., 4., 6., 7.])
		self.antecedent['Acceleration']['large'] = \
				fuzz.trapmf(self.antecedent['Acceleration'].universe, [6., 7., 10., 10.])

		### Deceleration
		self.antecedent['Deceleration'] = ctrl.Antecedent(universe=np.linspace(0,10, 11), 
			                                                   label='Deceleration')
		self.antecedent['Deceleration']['small'] = \
				fuzz.trapmf(self.antecedent['Deceleration'].universe, [0., 0., 3., 4.])
		self.antecedent['Deceleration']['medium'] = \
				fuzz.trapmf(self.antecedent['Deceleration'].universe, [3., 4., 6., 7.])
		self.antecedent['Deceleration']['large'] = \
				fuzz.trapmf(self.antecedent['Deceleration'].universe, [6., 7., 10., 10.])


		### Lateral Jerk
		self.antecedent['LateralJerk'] = ctrl.Antecedent(universe=np.linspace(0,16, 17), 
			                                                   label='LateralJerk')
		self.antecedent['LateralJerk']['small'] = \
				fuzz.trapmf(self.antecedent['LateralJerk'].universe, [0., 0., 4., 6.])
		self.antecedent['LateralJerk']['medium'] = \
				fuzz.trapmf(self.antecedent['LateralJerk'].universe, [4., 6., 10.,  12.])
		self.antecedent['LateralJerk']['large'] = \
				fuzz.trapmf(self.antecedent['LateralJerk'].universe, [10., 12., 16., 16.])

		### Velocity
		self.antecedent['Velocity'] = ctrl.Antecedent(universe=np.linspace(0,100, 101), 
			                                               label='Velocity')
		self.antecedent['Velocity']['very_slow'] = fuzz.trapmf(
								self.antecedent['Velocity'].universe, [0., 0., 15., 20.])
		self.antecedent['Velocity']['slow'] = fuzz.trapmf(
								self.antecedent['Velocity'].universe, [15., 20., 30., 35.])
		self.antecedent['Velocity']['normal'] = fuzz.trapmf(
								self.antecedent['Velocity'].universe, [30., 35., 50., 55.])
		self.antecedent['Velocity']['fast'] = fuzz.trapmf(
								self.antecedent['Velocity'].universe, [50., 55., 70., 75.])
		self.antecedent['Velocity']['very_fast'] = fuzz.trapmf(
								self.antecedent['Velocity'].universe, [70., 75., 100., 100.])
		
		# CONSEQUENT
		### Behavior (Driving Style)
		self.consequent['Behavior'] =  ctrl.Consequent(universe=np.linspace(0,1., 11), 
			                                               label='Behavior')
		self.consequent['Behavior']['calm'] = fuzz.trapmf(self.consequent['Behavior'].universe, 
															[0., 0., 0.2,  0.4])
		self.consequent['Behavior']['moderate'] = fuzz.trapmf(self.consequent['Behavior'].universe, 
															[0.2, 0.4, 0.6, 0.8])
		self.consequent['Behavior']['aggressive'] = fuzz.trapmf(self.consequent['Behavior'].universe, 
															[0.6, 0.8, 1., 1.])


	def build_rules(self, rules_dir:str)->ctrl.ControlSystem:

		assert os.path.exists(rules_dir),\
			('[Fuzzy Logic System T1 model][build_rules][ERROR]'
			 ' rules_dir not found!{}').format(rules_dir)

		rules_files = os.listdir(rules_dir)
		
		rules = None
		if self.expert_mode=='single':
			rules_files[0] = 'rules_0.csv'
			print('[Fuzzy Logic System T1 mode][build rules]', end='')
			print(f' single expert system! (rule:{rules_files[0]})')

			rules = self._single_expert_rules(os.path.join(rules_dir, rules_files[0]))
		elif self.expert_mode=='multiple':
			print('[Fuzzy Logic System - T1][build_rules]', end='')
			print(f' multiple expert system: (n_e: {len(rules_files)})')
			rules = self._multiple_expert_rules(rules_files, root_dir=rules_dir)
		else:
			assert False,\
				('[Fuzzy Logic System T1 model][build_rules][ERROR]'
				 ' expert_mode invalid! {}').format(self.expert_mode)

		assert rules is not None,\
			('[Fuzzy Logic System T1 model][build_rules][ERROR]'
			 ' error while building rules..')

		system  = ctrl.ControlSystem(rules=rules)
		return system

	
	def _single_expert_rules(self, rule_file:str)->List:
		
		rules = pd.read_csv(rule_file)

		assert rules.shape[1] == 5,\
			('[Fuzzy Logic System T1 model][build_rules] wrong rule_file shape'
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

			fr = ctrl.Rule(antecedent=(self.antecedent['Velocity'][r['velocity']] &\
							   self.antecedent['Acceleration'][r['acceleration']] &\
							   self.antecedent['Deceleration'][r['deceleration']] &\
							   self.antecedent['LateralJerk'][r['lateral_jerk']]),
						  consequent=self.consequent['Behavior'][xs],
						  label=f'rule - {index}')


			fuzz_rules.append(fr)

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

			y.append(set_labels[np.argmax(memb_value)])
			#print(y[-1])


		#create rules
		fuzz_rules = []

		for line, _y in zip(rules.iterrows(), y):
			index, r = line[0], line[1]

			fr = ctrl.Rule(antecedent=(self.antecedent['Velocity'][r['velocity']] &\
							   self.antecedent['Acceleration'][r['acceleration']] &\
							   self.antecedent['Deceleration'][r['deceleration']] &\
							   self.antecedent['LateralJerk'][r['lateral_jerk']]),
						  consequent=self.consequent['Behavior'][_y],
						  label=f'rule - {index}')


			fuzz_rules.append(fr)

		return fuzz_rules

	def _fuzz_driving_style(self,value:float)->Tuple:
		memb_value = []
		set_labels = []
		for label, term in self.consequent['Behavior'].terms.items():
			mi =fuzz.interp_membership(self.consequent['Behavior'].universe,
									   term.mf,
									   value)
			memb_value.append(mi)
			set_labels.append(label)

		return memb_value, set_labels

	def inference(self, observation:Dict) -> Dict:

		"""
			perform inference at the fuzzy system
		"""

		vel = observation['velocity']*3.6 #m/s -> Km/h
		acc = observation['acceleration'] #m/s^2
		dec = observation['deceleration'] #m/s^2
		ljk = observation['lateral_jerk'] #std (m/s^3)


		self.fuzz_inf.input['Acceleration'] = acc
		self.fuzz_inf.input['Deceleration'] = dec
		self.fuzz_inf.input['LateralJerk'] = ljk
		self.fuzz_inf.input['Velocity'] = vel

		self.fuzz_inf.compute()

		y = self.fuzz_inf.output['Behavior']
		
		memb_value, set_labels = self._fuzz_driving_style(value=y)



		result = {}
		result['value'] = y
		result['membership_values'] = np.asarray(memb_value)
		result['set_labels']=set_labels
		return result


	def plot(self)-> NoReturn:

		self.antecedent['Acceleration'].view()
		self.antecedent['Deceleration'].view()
		self.antecedent['Velocity'].view()
		self.consequent['Behavior'].view()
		plt.show()


