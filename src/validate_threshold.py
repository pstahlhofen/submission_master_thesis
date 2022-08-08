import pandas as pd
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.optimize import minimize_scalar

from time_constants import SECONDS_PER_DAY
from LeakProperties import LeakProperties
from LeakageDetectors import SingleSensorForecaster, BetweenSensorInterpolator

def time_to_detection_score(alarm_times, leak_properties):
	'''
	Compute a score for the time it takes to detect a leak.

	The score is returned on a 0-1 scale, where 1 means that the leak was
	detected immediately and 0 means that it was not detected, which could
	also mean that an alarm was raised only after the leak had already been
	fixed. All intermediate values are computed as 1 - relative_ttd, where
	relative_ttd is a quotient of the actual time before the leak was detected
	and the leak's duration.

	Parameters
	-----------
	alarm_times: array-like
	times of alarms in seconds

	leak_properties: LeakProperties.LeakProperties object
	this is used to retrieve the leak start and duration

	Returns
	--------
	score between 0 (bad) and 1 (perfect)
	'''
	start = leak_properties.start_time
	alarm_delays = [alarm - start for alarm in alarm_times if alarm >= start]
	if not alarm_delays:
		return 0
	total_ttd = min(alarm_delays)
	relative_ttd = total_ttd / leak_properties.duration
	return 1 - relative_ttd if relative_ttd < 1 else 0

class ThresholdValidator():
	'''
	Tool to test alarm thresholds of a given LeakageDetector

	Parameters
	-----------

	leakage_detector: an instance of a subclass of AbstractLeakageDetector
	A threshold can be tested for this leakage_detector. For this purpose, the
	threshold of the given instance will be overwritten.

	scenario_paths: list of str
	paths to folders each of which must contain the following files:
	leak_info.json: information about a leak that was placed in a
	WaterNetworkModel
	pressures.csv: pressure values from a simulation of the same model
	These files can be constructed using create_validation_set.py.
	'''

	def __init__(self, leakage_detector, scenario_paths):
		self.leakage_detector = leakage_detector
		self.scenario_paths = scenario_paths

	def validate(self, threshold, verbose=False):
		'''
		Evaluates the quality of a threshold.

		As a basis, this method uses the scenarios in self.scenario_paths. The
		score for each threshold is determined by the harmonic mean of the
		time to detection score and the true negative rate:
		2 * ttd_score * tnr / (ttd_score + tnr)

		Parameters
		-----------

		threshold: float
		the threshold to be validated

		verbose: int, default=False
		If True, the different scores (TTD and TNR) are printed, as well as
		the resulting score

		Returns
		---------
		scores: dict, keys=sceanrio paths, values=scores
		The resulting scores for each scenario
		'''
		self.leakage_detector.threshold = threshold
		scores = dict()
		for scenario_path in self.scenario_paths:
			if scenario_path[-1] != '/':
				scenario_path += '/'
			if verbose:
				print(f'Path: {scenario_path}')
			pressure_file = scenario_path + 'pressures.csv'
			pressures = pd.read_csv(pressure_file, index_col='time')
			alarms = self.leakage_detector.train_and_detect(pressures)
			# Getting times at which alarms might be generated
			test_start = self.leakage_detector.train_days * SECONDS_PER_DAY
			test_times = pressures.loc[test_start:].index

			alarm_times = alarms.times()
			has_alarm = [time in alarm_times for time in test_times]

			leak_file = scenario_path + 'leak_info.json'
			leak_properties = LeakProperties.from_json(leak_file)
			leak_start = leak_properties.start_time
			leak_end = leak_start + leak_properties.duration
			leak_check = lambda time: time >= leak_start and time < leak_end
			has_leak = [leak_check(time) for time in test_times]

			ttd_score = time_to_detection_score(alarm_times, leak_properties)
			if verbose:
				print(f'Time To Detection Score: {ttd_score:.3f}')

			conf_mat = confusion_matrix(has_leak, has_alarm)
			true_negatives, false_positives = conf_mat[0]
			tnr = true_negatives / (true_negatives + false_positives)
			if verbose:
				print(f'True Negative Rate: {tnr:.3f}')

			harmonic_mean = 2 * tnr * ttd_score / (tnr + ttd_score)
			if verbose:
				print(f'Score: {harmonic_mean:.3f}')
			scores[scenario_path] = harmonic_mean
		return scores

	def negative_average_score(self, threshold):
		'''
		Computes the negative average of all scores of a threshold (see
		self.validate). This is useful as input to optimizers.
		'''
		scores = self.validate(threshold)
		average_score = np.array(list(scores.values())).mean()
		return average_score * (-1)

if __name__=='__main__':
	# The zero-threshold is just a placeholder which gets changed
	# during the validation process.
	leakage_detector = BetweenSensorInterpolator(
		nodes_with_sensors = ['4', '13', '16', '22', '31'],
		train_days = 7,
		k = 1,
		threshold = 0
	)
	scenario_paths = []
	n_scenarios = 3
	first_scenario = 7
	for i in range(first_scenario, n_scenarios + first_scenario):
		scenario_paths.append(f'../Data/Threshold_Validation/Scenario-{i}/')
	tv = ThresholdValidator(leakage_detector, scenario_paths)
	optimization_result = minimize_scalar(
		tv.negative_average_score,
		bounds=(0, 20)
	)
	best_threshold = optimization_result.x
	best_average_score = optimization_result.fun * (-1)
	print(
		f'The optimal threshold {best_threshold:.3f}'
		f' achieved an average score of {best_average_score:.3f}!'
	)
	scores = tv.validate(best_threshold, verbose=True)
	mean = np.array(list(scores.values())).mean()
	print(f'Average Score: {mean:.3f}')
