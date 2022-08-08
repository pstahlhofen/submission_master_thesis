"""
This example builds a python representation of the Net1 water distribution
network using WNTR, implements a simple leakage detection algorithm using
linear regression and calculates the least sensitive point in the network
with respect to the detection algorithm for fixed leak duration.
This is accomplished by iteratively approximating the leakage area
such that a leak of that area is not detected in *one* node, while it is
detected in all other nodes of the network. The node in which the leak
could not be detected is the least sensitive point of the network for fixed
leak starting time and duration. The leak area is approximated using binary
search.
To generalize the least sensitive point over a range of starting times,
an initial guess for the maximal leak area is obtained by running
the algorithm described above for different randomly selected starting times.
After determining the highest undetected leak area for each of the timesteps,
the maximum of these samples is used for a global search: For a range of
starting times, each junction is tested again with the derived maximum area.
Again, if the leak only stays unnoticed for one of the junctions across all
timesteps, the global least sensitive point is found. If this was not the 
case, the global search is repeated with a higher area value and continued
in a binary scheme. Junctions and starting times out of question for the
least sensitive point are pruned in order to reduce computational complexity.
"""
import wntr
from sklearn.metrics import f1_score
import pygad
import numpy as np
import pandas as pd

from bisection import bisection_search
from time_constants import SECONDS_PER_HOUR, HOURS_PER_DAY, SECONDS_PER_DAY
import wn_util
from LeakageDetectors import SingleSensorForecaster, BetweenSensorInterpolator
from LeakProperties import LeakProperties

# I know, this is a bit ugly, but the UserWarnings caused by the WNTRSimulator
# are not analyzed by this script and I could not turn them off locally.
import warnings
warnings.filterwarnings('ignore')

class LspFinder():
	'''
	Tool to find the least sensitve point in a water network.

	The least sensitive point is the junction at which one could place the
	largest possible leak that still remains undetected. For a formal
	definition, see ../Formalizations/gradient_lsp.pdf

	Parameters
	-----------

	network_file: str, name of an EPANET inp-file
	file containing the network to be analyzed

	leak_duration: int
	duration of each tested leak in seconds

	leakage_detector: an instance of a subclass of AbstractLeakageDetector
	the leakage detector must have already been trained (see documentation of
	leakage_detactor.train) with suitable pressure values. These should be
	produced by a leak-free simulation of the same network.

	sim_start: pd.Timestamp, optional, default: pd.Timestamp(0)
	timestamp for the beginning of the simulation (e.g. 1997-04-06 19:04:00)
	This is useful to convert all time offsets in seconds given by wntr into
	timestamps which reflect useful information like the hour of a day or
	summer/winter periods.
	Note: In the pandas implementation, Timestamp(0) evaluates to
	'1970-01-01 00:00'

	ignore_nodes: list of str, optional, default: empty list
	list of junction names that should be ignored in the search for the least
	sensitive point. This can be useful to exclude "boring" junctions.
	'''

	def __init__(self, network_file, leak_duration, leakage_detector, sim_start=pd.Timestamp(0), ignore_nodes=[]):
		self.network_file = network_file
		self.leak_duration = leak_duration
		self.leakage_detector = leakage_detector
		self.sim_start = sim_start
		
		wn = wntr.network.WaterNetworkModel(network_file)
		self.junction_name_list = wn.junction_name_list
		for node_name in ignore_nodes:
			self.junction_name_list.remove(node_name)

		# Constructing a range of potential start times of leaks
		# such that the whole leak fits inside the simulation time.
		# See wn_util.start_time_range for the construction.
		sim_time = wn.options.time.duration
		timestep = wn.options.time.hydraulic_timestep
		self.start_time_range = wn_util.start_time_range(
			start=0,
			end=sim_time,
			leak_duration=leak_duration,
			timestep=timestep
		)

		self.bisection_search_cache = dict(
			start_times=self.start_time_range,
			junctions=self.junction_name_list
		)
		self.combined_search_cache = dict()

	def absolute_time(self, offset):
		'''Convert an offset in seconds to a pd.Timestamp.'''
		return self.sim_start + pd.Timedelta(seconds=offset)

	def place_leak_and_detect(self, leak_properties):
		'''
		Detect a leak with given properties.

		self.network_file is used to load the network in which the leak is
		placed.

		Parameters
		-----------

		leak_properties: LeakProperties.LeakProperties object
		this is passed to compute_pressures. See documentation of
		the LeakProperties class for details

		Returns
		--------
		an alarms.Alarms object as produced by
		self.leakage_detector.detect
		'''
		wn = wntr.network.WaterNetworkModel(self.network_file)
		pressures = wn_util.compute_pressures(
			wn, leak_properties=leak_properties
		)
		alarms = self.leakage_detector.detect(pressures)
		return alarms

	def alarms_for_area(self, leak_area, *, start_time):
		'''
		Place a leak of equal area at each junction in the network.

		This re-constructs the network and runs a seperate simulation
		for each junction.

		Parameters
		-----------

		leak_area: float
		area in m^2, passed to self.place_leak_and_detect

		start_time: int, required key-word argument
		start time of the leak in seconds

		Returns
		--------
		num_alarms: int
		The numer of junctions for which any alarm was triggered

		lsp_candidates: list of str
		The names of all the junctions for which no alarm was triggered. These
		remain candidates for the least sensitive point.
		'''
		print(f'Leak area: {leak_area * 10000:.2f} cm^2')
		num_alarms = 0
		lsp_candidates = []
		for junction_name in self.junction_name_list:
			print(f'Placing leak at node {junction_name}')
			leak_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			alarms = self.place_leak_and_detect(leak_properties)
			end_time = start_time + leak_properties.duration
			if alarms.any_during(start_time, end_time):
				num_alarms += 1
			else:
				print(
					f'Leak with size {leak_area * 10000:.2f} cm^2'
					f' at node {junction_name}'
					f' triggered no alarm.'
				)
				lsp_candidates.append(junction_name)
		print('*'*30 + '\n')
		return num_alarms, lsp_candidates

	def maximize_leak_area(self, initial_area, junction_name, start_time, maximization_trials, lower_bound=None, upper_bound=None, verbose=False):
		'''
		Maximize the leak area for a junction, s.t. no alarm is triggered.

		Parameters
		-----------

		initial_area: float
		an initial guess for the leakage area.
		Important: Must be chosen s.t. it does not create an alarm

		junction_name: str
		name of a network junction where the leak should be placed

		start_time: int, start time of the leak in seconds

		maximization_trials: int, must be positive
		number of trials in the leak maximization process

		upper_bound: float, default=None
		a leak-area value in m^2 which is known to produce an alarm

		Returns
		--------
		max_area: float
		the maximal leak area which does not produce an alarm
		'''
		if verbose:
			print(f'Maximize the leak area for junction {junction_name}')
		# Take some area and check if it produces a leak, for a fixed junction
		def check_leak_at_junction(area):
			leak_properties = LeakProperties(
				area=area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			alarms = self.place_leak_and_detect(leak_properties)
			end_time = start_time + leak_properties.duration
			alarms_triggered = alarms.any_during(start_time, end_time)
			alarm_string = 'alarm' if alarms_triggered else 'no alarm'
			if verbose:
				print(f'Leak area: {area * 10000:.2f} cm^2 -> {alarm_string}')
			return int(alarms_triggered)
		# Running bisection_search with a sought_result of 0.5 will never
		# finish before the end of all trials, but get closer to the point
		# where the area starts to trigger the alarm. The highest area below
		# this threshold is the maximal area we are looking for.
		_, bounds = bisection_search(
			check_leak_at_junction, 0.5,
			initial_area, maximization_trials,
			lower_bound=lower_bound, upper_bound=upper_bound
		)
		max_area = bounds[0]
		return max_area

	def test_leak_over_time(self, area):
		'''
		Test a leak with fixed area across all nodes and timesteps.

		For each timestep, this method will record the junctions where a leak
		with size `area` did NOT trigger an alarm. It will use
		self.bisection_search_cache both to read relevant junctions and timesteps in the
		beginning of the search and to store junctions and timesteps that
		remain relevant for further search after one iteration. Hence, if you
		call this method again with a different area, it will presumably
		perform much faster.

		Parameters
		-----------

		area: float
		leak area in m^2

		Returns
		--------
		n_excluded: int
		number of junctions, which are definitely NOT the
		least sensitive point. These are junctions, at which a leak
		of the given area always triggered an alarm.

		unnoticed: dictionary
		the keys are start times of the leaks in seconds, the values (str) are
		names of the network junctions.  For each time, the names of the
		junctions for which NO alarm was triggered are given.
		'''
		# Remember start times which remain candidates
		print(f'Trying area: {area * 10000:.2f} cm^2')
		start_time_candidates = []
		junction_candidates = set()
		unnoticed = dict()
		for start_time in self.bisection_search_cache['start_times']:
			unnoticed[start_time] = []
			print(
				f'Placing leak starting at {self.absolute_time(start_time)}'
			)
			for junction_name in self.bisection_search_cache['junctions']:
				leak_properties = LeakProperties(
					area=area,
					junction_name=junction_name,
					start_time=start_time,
					duration=self.leak_duration
				)
				alarms = self.place_leak_and_detect(leak_properties)
				end_time = start_time + leak_properties.duration
				if not alarms.any_during(start_time, end_time):
					junction_candidates.add(junction_name)
					unnoticed[start_time].append(junction_name)
					print(
						f'-- Leak at node {junction_name}'
						f' triggered no alarm!'
					)
			if unnoticed[start_time]: # list is not empty
				start_time_candidates.append(start_time)
		print('*'*30 + '\n')
		if junction_candidates:
			self.bisection_search_cache['start_times'] = start_time_candidates
			self.bisection_search_cache['junctions'] = junction_candidates
		# This is imported to make the function work with bisection_search
		n_excluded = len(self.junction_name_list) - len(junction_candidates)
		return n_excluded, unnoticed

	def find_lsp_bisection(self, *, start_time_trials, maximization_trials, trials_per_timestep, global_trials, initial_area=0.01):
		'''
		Use bisection_search to determine the least sensitive point.

		The procedure of this function consists of two main steps:
		1. for a fixed number of time trials...
			1.1 find the least sensitive point at that time
			1.2 maximize the leak area for that time at the least sensitive
			point such that the leakage detection algorithm causes no alarm.
		Subsequently, use the highest of the maxima determined above
		as a starting point for step 2
		2. find the least sensitive point globally (over all starting times)

		In each of the substeps, binary search is applied to find a leakage
		area fulfilling the condition. In case of 1.1 and 2, the area is
		picked such that it does NOT trigger an alarm in one node, while it
		does trigger an alarm in all other nodes. See bisection_search for the
		search procedure.  For the global lest-sensitive-point search (Step
		2), see self.test_leak_over_time.

		This method has no return value. Instead, it will print its results.

		Parameters
		-----------
	
		trial_params: dict, numbers of trials for different search-phases
		The dictionary must contain the following entries:
			- start_time_trials: int, number of different leak starting times
			  in phase 1
			- trials_per_timestep: int, number of trials per starting time in
			  phase 1
			- maximization_trials: int, number of trials to maximize the leak
			  area in phase 1.2 (passed to maximize_leak_area)
			- global_trials: int, maximum amount of search_leak_over_time
			  calls in phase 2

		initial_area: float, area in m^2, default=0.01
		the area that is used at the beginning of the search in each time
		trial in step 1.1

		This method has no return value. Instead, it will print information
		about the least sensitive point.
		'''
		start_times = np.random.choice(
			self.start_time_range,
			size=start_time_trials,
			replace=False
		)
		max_area = 0
		for index, start_time in enumerate(start_times):
			print('='*15 + f'TIME TRIAL {index + 1}' + '='*15)
			end_time = start_time + self.leak_duration
			print(
				f'Time parameters of the leak\n'
				f'Start: {self.absolute_time(start_time)}\n'
				f'End: {self.absolute_time(end_time)}\n'
			)
			leak_area, bounds, lsp_candidates = bisection_search(
				self.alarms_for_area,
				len(self.junction_name_list) - 1,
				initial_area,
				trials_per_timestep,
				meta_info=True,
				start_time=start_time
			)
			lsp = lsp_candidates[0]
			print(f'The least sensitive point is node {lsp}\n')
			new_max_area = self.maximize_leak_area(
				leak_area, lsp, start_time,
				maximization_trials,
				lower_bound=leak_area,
				upper_bound=bounds[1],
				verbose=True
			)
			print(
				f'The maximum leak area at the least sensitive point'
				f' (node {lsp}) and the given time parameters (see above)'
				f' is {new_max_area * 10000:.2f} cm^2.\n'
			)
			max_area = max([new_max_area, max_area])
		print()
		print(
			f'Maximum leak area across the time trials:'
			f' {max_area * 10000:.2f} cm^2'
		)
		print('#'*30 + '\n')
		leak_area, bounds, unnoticed = bisection_search(
			self.test_leak_over_time, 
			len(self.junction_name_list) - 1,
			max_area,
			global_trials,
			lower_bound=max_area,
			meta_info=True
		)
		total_lsp = self.bisection_search_cache['junctions'].pop()
		print(
			f'The total least sensitive point across all starting times'
			f' is junction {total_lsp}.'
		)

	def find_lsp_genetic(self, max_tested_area=0.02, max_initial_area=0.01,
			save_last_generation=True, load_last_generation=False,
			performance_path=None, verbose=False):
		'''
		Find the least sensitive point using the genetic algorithm from pygad.

		This will start a search with the three different genes 'start_time',
		'network_junction' and 'leak_area'. The fitness function is equal to
		the leak area, if the leak was not detected and equal to 0 if it was
		detected. One may use save_last_generation=True or
		load_last_generation=True to save the last generation from one run of
		the genetic algorithm or to load a saved last generation from a
		previous run and use it as the initial population, respectively.

		Parameters
		-----------

		max_tested_area: float, upper limit of the gene space for area.
		This should be chosen with care: A too small max_tested_area will
		cause large leaks not to be explored while a too large max_tested_area
		might lead to many mutations with large areas, which are quickly
		discarded because they are detected and receive a fitness score of 0.
		One might use increasing values for successive runs with saved and
		re-loaded last generations.

		max_init_area: The maximal area of chromosomes in the initial
		population. If load_last_generation=True, this is ignored.

		save_last_generation: bool, optional, default=True
		If True, save the last generation created by the genetic algorithm to
		a file called 'last_generation.npy'

		load_last_generation: bool, optional, default=False
		If True, load a population from the file 'last_generation.npy' which
		must have been created by previous runs of the algorihtm and use it as
		initial population.

		verbose: bool, optional, default=False
		In addition to the usual output, print the ordinal number of the
		genertion in which the least sensitive point was found. This can be
		used for fault diagnosis.

		performance_path: str, optional, default=None
		If the name of a directory is given here, information about the
		performance of the algorithm will be written to that directory. In
		case the directory does not exist yet, a new one will be created at
		the given path. The performance information include leak properties of
		the best attack found, parameter settings of the genetic algorithm and
		the evolution of the best achieved fitness value over the generations.
		See wn_util.describe_performance for details.

		This method has no return value. Instaed, it will print information
		about the least sensitive point.
		'''
		def fitness_function(solution, solution_idx):
			start_time_idx, junction_name_idx, leak_area = solution
			start_time = self.start_time_range[start_time_idx]
			junction_name = self.junction_name_list[junction_name_idx]
			leak_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			alarms = self.place_leak_and_detect(leak_properties)
			end_time = start_time + leak_properties.duration
			if alarms.any_during(start_time, end_time):
				return 0
			else:
				return leak_area
		start_time_idxs = np.arange(
			len(self.start_time_range), dtype=np.int64
		)
		junction_name_idxs = np.arange(
			len(self.junction_name_list), dtype=np.int64
		)
		gene_space = [\
			start_time_idxs,
			junction_name_idxs,
			{'low': 0, 'high': max_initial_area}
		]
		gene_type = [np.int64, np.int64, np.float64]
		num_genes = 3

		num_generations = 50
		sol_per_pop = 20
		if load_last_generation:
			initial_population = np.load(
				'last_generation.npy', allow_pickle=True
			)
		else:
			initial_start_time_idxs = np.random.choice(
				start_time_idxs, size=sol_per_pop, replace=False
			)
			initial_junction_name_idxs = np.random.choice(
				junction_name_idxs, size=sol_per_pop, replace=False
			)
			initial_areas = np.random.uniform(
				low=0, high=max_initial_area, size=sol_per_pop
			)
			initial_population = np.column_stack(
				(
					initial_start_time_idxs,
					initial_junction_name_idxs,
					initial_areas
				)
			)
		reproduction_rate = 0.25
		num_parents_mating = int(sol_per_pop * reproduction_rate)
		parent_selection_type = 'sss'
		keep_parents = 1
		crossover_type = 'uniform'
		mutation_type = 'random'
		mutation_probability = 0.1
		save_solutions = False

		ga_instance = pygad.GA(
			fitness_func=fitness_function,
			gene_space=gene_space,
			gene_type=gene_type,
			num_genes=num_genes,
			num_generations=num_generations,
			sol_per_pop=sol_per_pop,
			initial_population=initial_population,
			num_parents_mating=num_parents_mating,
			parent_selection_type=parent_selection_type,
			keep_parents=keep_parents,
			crossover_type=crossover_type,
			mutation_type=mutation_type,
			mutation_probability=mutation_probability,
			save_solutions=save_solutions
		)
		ga_instance.run()
		ga_solution = ga_instance.best_solution()[0]
		start_time_idx, junction_name_idx, leak_area = ga_solution
		start_time = self.start_time_range[start_time_idx]
		junction_name = self.junction_name_list[junction_name_idx]
		if performance_path is not None:
			lsp_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			wn_util.describe_performance(
				ga_instance, lsp_properties, performance_path
			)
		print(
			f'The least sensitive point is junction {junction_name}'
			f' at {self.absolute_time(start_time)}'
			f' with a leak area of {leak_area*10000:.2f} cm^2'
		)
		if verbose:
			bsg = ga_instance.best_solution_generation
			print(f'The best solution was found in generation {bsg}.')
		if save_last_generation:
			np.save('last_generation.npy', ga_instance.population)
		return

	def find_lsp_combined(self, maximization_trials=10, initial_area=0.01,
			performance_path=None, reset_search_cache=True, verbose=False):
		'''
		Find the least sensitive point with a genetic algorithm and bisection.

		This will start a genetic algorithm with the two different genes
		'start_time' and 'network_junction'. To evaluate the fitness function,
		bisection search is used to find the maximal leak area at the given
		junction and point in time, for which no alarm is triggered (see
		self.maximize_leak_area). To avoid running the maximization
		simulations for each candidate, the greatest leak area found so far is
		stored in self.combined_search_cache['area']. Before the maximization,
		a leak of that area is placed at the given junction-time combination:
		If it triggeres an alarm, the candidate does not yield an improvement
		to the highest fitness value found so far. Hence, the fitness function
		will return 0 in these cases.

		Parameters
		-----------

		maximization_trials: int, optional, default=10
		number of bisection search trials that are used to maximize the leak
		area. A higher number of trials will produce more accurate results at
		the cost of computation time.
		
		initial_area: float, optional, default=0.01
		The leak area that is used as a starting point for the maximization of
		the first solution candidate. For all following evaluations of the
		fitness function, the greatest leak area found so far is used as the
		starting point.

		verbose: bool, optional, default=False
		In addition to the usual output, print the ordinal number of the
		genertion in which the least sensitive point was found. This can be
		used for fault diagnosis.

		performance_path: str, optional, default=None
		If the name of a directory is given here, information about the
		performance of the algorithm will be written to that directory. In
		case the directory does not exist yet, a new one will be created at
		the given path. The performance information include leak properties of
		the best attack found, parameter settings of the genetic algorithm and
		the evolution of the best achieved fitness value over the generations.
		See wn_util.describe_performance for details.

		reset_search_cache: bool, default=True
		If True, self.combined_search_cache is reset before the next call of
		this function. This is useful if one wants to run multiple trials
		starting from zero. If this is set to false, a maximum leak area of 0
		might be returned. In the case, the least sensitive point from one of
		the previous runs could not be improved.

		This method has no return value. Instaed, it will print information
		about the least sensitive point.
		'''
		def fitness_function(solution, solution_idx):
			start_time_idx, junction_name_idx = solution
			start_time = self.start_time_range[start_time_idx]
			junction_name = self.junction_name_list[junction_name_idx]
			# self.combined_search_cache will be set
			# after the first evaluation of this fitness function
			if self.combined_search_cache:
				# Check if the current combination can tolerate
				# a greater leak
				greatest_leak_area = self.combined_search_cache['area']
				leak_properties = LeakProperties(
					area=greatest_leak_area,
					junction_name=junction_name,
					start_time=start_time,
					duration=self.leak_duration
				)
				alarms = self.place_leak_and_detect(leak_properties)
				end_time = start_time + leak_properties.duration
				if alarms.any_during(start_time, end_time):
					return 0
				else:
					# if the greatest leak area so far did not create an alarm
					# try to increase the area
					leak_area = self.maximize_leak_area(
						greatest_leak_area, junction_name, start_time,
						maximization_trials
					)
			else: # in the very first function evaluation
				leak_area = self.maximize_leak_area(
					initial_area, junction_name, start_time,
					maximization_trials,
				)
			self.combined_search_cache['start_time'] = start_time
			self.combined_search_cache['junction_name'] = junction_name
			self.combined_search_cache['area'] = leak_area
			return leak_area
		start_time_idxs = np.arange(
			len(self.start_time_range), dtype=np.int64
		)
		junction_name_idxs = np.arange(
			len(self.junction_name_list), dtype=np.int64
		)
		gene_space = [start_time_idxs, junction_name_idxs]
		gene_type = [np.int64, np.int64]
		num_genes = 2

		num_generations = 50
		sol_per_pop = 20
		initial_start_time_idxs = np.random.choice(
			start_time_idxs, size=sol_per_pop, replace=False
		)
		initial_junction_name_idxs = np.random.choice(
			junction_name_idxs, size=sol_per_pop, replace=False
		)
		initial_population = np.column_stack(
			(initial_start_time_idxs, initial_junction_name_idxs)
		)
		reproduction_rate = 0.25
		num_parents_mating = int(sol_per_pop * reproduction_rate)
		parent_selection_type = 'sss'
		keep_parents = 1
		crossover_type = 'uniform'
		mutation_type = 'random'
		mutation_probability = 0.1
		save_solutions = False

		ga_instance = pygad.GA(
			fitness_func=fitness_function,
			gene_space=gene_space,
			gene_type=gene_type,
			num_genes=num_genes,
			num_generations=num_generations,
			sol_per_pop=sol_per_pop,
			initial_population=initial_population,
			num_parents_mating=num_parents_mating,
			parent_selection_type=parent_selection_type,
			keep_parents=keep_parents,
			crossover_type=crossover_type,
			mutation_type=mutation_type,
			mutation_probability=mutation_probability,
			save_solutions=save_solutions
		)
		ga_instance.run()
		ga_solution, leak_area, _ = ga_instance.best_solution()
		start_time_idx, junction_name_idx = ga_solution
		start_time = self.start_time_range[start_time_idx]
		junction_name = self.junction_name_list[junction_name_idx]
		if performance_path is not None:
			lsp_properties = LeakProperties(
				area=leak_area,
				junction_name=junction_name,
				start_time=start_time,
				duration=self.leak_duration
			)
			wn_util.describe_performance(
				ga_instance, lsp_properties, performance_path
			)
		print(
			f'The least sensitive point is junction {junction_name}'
			f' at {self.absolute_time(start_time)}'
			f' with a leak area of {leak_area*10000:.2f} cm^2'
		)
		if verbose:
			bsg = ga_instance.best_solution_generation
			print(f'The best solution was found in generation {bsg}.')
		if reset_search_cache:
			self.combined_search_cache = dict()
		return

	def find_lsp(self, algorithm, **params):
		'''
		Find the least sensitive point using the given algorithm.

		Currently, the 'genetic' and 'bisection' algorithms are two different
		implemented approaches to find the least sensitive point in a water
		network. Please note that the parameters (**params) are different for
		the two algorithms

		Parameters
		-----------

		algorithm: str, one of 'bisection' or 'genetic'
		search algorithm to use (for detailed documentation see
		find_lsp_bisection and find_lsp_genetic)

		**params:
		parameters passed to the algorithms as key-word arguments. These
		differ between algorithms (again, see documentation of
		find_lsp_genetic and find_lsp_bisection for allowed key-word
		arguments)

		This method has no return value. Instead, it will print informaion
		about the least sensitive point.
		'''
		allowed_algorithms = ['bisection', 'genetic', 'combined']
		if algorithm not in allowed_algorithms:
			raise ValueError(f'algorithm must be one of {allowed_algorithms}')
		if algorithm=='bisection':
			self.find_lsp_bisection(**params)
		elif algorithm=='genetic':
			self.find_lsp_genetic(**params)
		elif algorithm=='combined':
			self.find_lsp_combined(**params)

if __name__=='__main__':
	nodes_with_sensors = ['4', '13', '16', '22', '31']
	train_days = 5
	k = 1
	threshold = 0.562
	leakage_detector = BetweenSensorInterpolator(
		nodes_with_sensors, train_days, k, threshold
	)
	network_path = '../Data/Hanoi_1week/'
	train_wn = wntr.network.WaterNetworkModel(network_path + 'train.inp')
	train_pressures = wn_util.compute_pressures(train_wn)
	leakage_detector.train(train_pressures)

	test_network_file = network_path + 'test.inp'
	leak_duration = 3 * SECONDS_PER_HOUR
	sim_start = pd.Timestamp(year=2017, month=7, day=6)
	ignore_nodes = ['2', '3']
	lspFinder = LspFinder(
		test_network_file, leak_duration, leakage_detector,
		sim_start, ignore_nodes
	)

	trial_params = dict(
		start_time_trials = 3,
		trials_per_timestep = 10,
		maximization_trials = 6,
		global_trials = 6
	)
	lspFinder.find_lsp(algorithm='combined')

