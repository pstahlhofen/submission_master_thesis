import wntr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import wn_util
from time_constants import SECONDS_PER_DAY, SECONDS_PER_HOUR
from LeakageDetectors import BetweenSensorInterpolator, SingleSensorForecaster
from LeakProperties import LeakProperties

def residual_plot(alarms, leak_properties, ax=None, title=None):
	'''
	Plot the residuals of an Alarms object and mark the actual leak time.
	
	This method can be used to diagnose the performance of a leakage detector.

	Note: This method is only usable if all alarm DataFrames have the same
	time index, i.e. if the threshold of the leakage detector was 0.

	Parameters
	-----------

	alarms: alarms.Alarms object
	for each DataFrame in alarms.values() the content of the 'time' column is
	assumed to be the same.

	leak_properties: LeakProperties.LeakProperties
	these are used to mark the leak time in the plot by vertical lines

	ax: matplotlib.axes._subplots.Axes, optional, default=None
	an Axes object to plot to. If none is given, a new Axes object is created.

	title: str, optional
	a title to add to the Axes object (not to the whole figure)

	Returns
	-------
	an Axes object with the plot
	'''
	all_residuals = alarms.residual_matrix()

	ax = all_residuals.plot(ax=ax)
	leak_start = leak_properties.start_time
	leak_end = leak_start + leak_properties.duration
	kwargs = dict(alpha=.1, color='k')

	ax.axvline(leak_start, **kwargs)
	ax.axvline(leak_end, **kwargs)
	if title is not None:
		ax.set_title(title)
	return ax

def plot_ssf(k):
	'''
	Create a residual plot for a SingleSensorForecaster with given k value.

	Note: This was used experimentally to create plots and requires the
	existance of global variables.
	'''
	fig, ax = plt.subplots()
	ssf = SingleSensorForecaster(
		nodes_with_sensors = nodes_with_sensors,
		train_days=train_days,
		k=k,
		threshold=0
	)
	alarms_ssf = ssf.train_and_detect(pressures)
	residual_plot(alarms_ssf, leak_properties, ax, title=f'$k={k}$')
	fig.suptitle('Single Sensor Forecaster')
	plt.savefig(f'../Leakage_Detector_Plots/ssf_{k}.png')

def plot_bsi(k):
	'''
	Create a residual plot for a BetweenSensorInterpolator with given k value.

	Note: This was used experimentally to create plots and requires the
	existance of global variables.
	'''
	fig, ax = plt.subplots()
	bsi = BetweenSensorInterpolator(
		nodes_with_sensors = nodes_with_sensors,
		train_days=train_days,
		k=k,
		threshold=0
	)
	alarms_bsi = bsi.train_and_detect(pressures)
	residual_plot(alarms_bsi, leak_properties, ax, title=f'$k={k}$')
	fig.suptitle('Between Sensor Interpolator')
	plt.savefig(f'../Leakage_Detector_Plots/bsi_{k}.png')

if __name__=='__main__':
	wn = wntr.network.WaterNetworkModel(
		'../Data/Hanoi_Leakage_Detector_Comparison.inp'
	)
	nodes_with_sensors = ['4', '13', '16', '31', 'Flow_1', 'Flow_2']
	train_days = 5
	leak_properties = LeakProperties(
		junction_name='23',
		area=0.005,
		start_time=6*SECONDS_PER_DAY+12*SECONDS_PER_HOUR,
		duration=3*SECONDS_PER_HOUR
	)
	pressures = wn_util.compute_pressures(
		wn, leak_properties, flow_links=['1', '2']
	)
	for k in [1,2,4,6,8]:
		plot_ssf(k)

