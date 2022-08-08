'''
Create plots documenting the change of prediction residuals due to leak area.


'''
import wntr
import numpy as np
import matplotlib.pyplot as plt

from LeakageDetectors import BetweenSensorInterpolator
import wn_util
from LeakProperties import LeakProperties
from time_constants import SECONDS_PER_HOUR, SECONDS_PER_DAY

def plot_node_property(areas, y, node_name, property_name, ax=None):
	'''
	Plot the property of a node which changes due to leak area.

	Parameters
	-----------
	areas: list-like object containing numbers
	different leak area values

	y: list-like object containing numbers
	values of the analyzed property (e.g. residual mean)

	node_name, property_name: str, used for figure labelling

	ax: optional, matplotlib Axes object
	if given, the plot is drawn on ax. Otherwise matplotlib uses the current
	Axes.
	'''
	if ax is None:
		ax = plt.gca() # get current axes
	ax.plot(areas, y, label=node_name)
	ax.set_xlabel('area')
	ax.set_ylabel(property_name)
	return ax

def plot_maxima(residuals_for_area, node_name, ax=None):
	'''
	Plot the maximum residual against leak area for the given node.

	Parameters
	-----------
	residuals_for_area: dict
	- Keys: leak area (float)
	- Values: pd.DataFrame containing residuals for each time in seconds (row
	  index) and node (column_index) as returned by Alarms.residual_matrix

	node_name: str, name of node to be analyzed

	ax: optional, matplotlib Axes object
	if given, the plot is drawn on ax. Otherwise matplotlib uses the current
	Axes.
	'''
	areas, maxima = zip(
		*[
			(area, residuals[node_name].max())\
			for area, residuals in residuals_for_area.items()
		]
	)
	plot_node_property(areas, maxima, node_name, 'residual max', ax=ax)
	return

def plot_means(residuals_for_area, node_name, ax=None):
	'''
	Plot the mean residual against leak area for the given node.

	Parameters
	-----------
	residuals_for_area: dict
	- Keys: leak area (float)
	- Values: pd.DataFrame containing residuals for each time in seconds (row
	  index) and node (column_index) as returned by Alarms.residual_matrix

	node_name: str, name of node to be analyzed

	ax: optional, matplotlib Axes object
	if given, the plot is drawn on ax. Otherwise matplotlib uses the current
	Axes.
	'''
	areas, means = zip(
		*[
			(area, residuals[node_name].mean())\
			for area, residuals in residuals_for_area.items()
		]
	)
	plot_node_property(areas, means, node_name, 'residual mean', ax=ax)
	return

if __name__=='__main__':
	network_path = '../Data/Hanoi_1week/'
	train_wn = wntr.network.WaterNetworkModel(network_path + 'train.inp')
	train_pressures = wn_util.compute_pressures(train_wn)

	nodes_with_sensors = ['4', '13', '16', '22', '31']
	train_days = 5
	k = 1
	threshold = 0
	leakage_detector = BetweenSensorInterpolator(
		nodes_with_sensors, train_days, k, threshold
	)
	leakage_detector.train(train_pressures)

	leak_areas = np.arange(0, 0.011, 0.001)
	residuals_for_area = dict()
	network_file = network_path + 'test.inp'
	for leak_area in leak_areas:
		wn = wntr.network.WaterNetworkModel(network_file)
		if leak_area:
			leak_properties = LeakProperties(
				junction_name = '4',
				start_time = 1 * SECONDS_PER_DAY + 12 * SECONDS_PER_HOUR,
				duration = 3 * SECONDS_PER_HOUR,
				area = leak_area
			)
			pressures = wn_util.compute_pressures(wn, leak_properties)
		else:
			pressures = wn_util.compute_pressures(wn) # without leak
		alarms = leakage_detector.detect(pressures)
		residuals_for_area[leak_area] = alarms.residual_matrix()
	output_path = '../Max_Residual_Change/Scenario-3/'
	for node_name in nodes_with_sensors:
		plt.figure()
		plot_maxima(residuals_for_area, node_name)
		plt.savefig(output_path + 'node_' + node_name + '.png')
	fig, ax = plt.subplots()
	for node_name in nodes_with_sensors:
		plot_maxima(residuals_for_area, node_name, ax)
	plt.legend()
	plt.savefig(output_path + 'compared_residuals.png')
	leak_properties.to_json(output_path + 'leak_info.json')
