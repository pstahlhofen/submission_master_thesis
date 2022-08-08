import pandas as pd
import sys
import matplotlib.pyplot as plt

from time_constants import SECONDS_PER_HOUR
from LeakageDetectors import BetweenSensorInterpolator
from LeakProperties import LeakProperties
from test_detectors import residual_plot

def plot_pressure_change(pressures, leak_properties):
	'''Plot the pressure change caused by a leak at the leaky junction.'''
	leak_junction_name = leak_properties.junction_name
	leak_start = leak_properties.start_time
	leak_end = leak_start + leak_properties.duration
	offset = 3 * SECONDS_PER_HOUR
	plot_start = leak_start - offset
	plot_end = leak_end + offset
	ax = pressures.loc[plot_start:plot_end, leak_junction_name].plot()
	kwargs = dict(alpha=0.5, color='r')
	ax.axvline(leak_start, **kwargs)
	ax.axvline(leak_end, **kwargs)
	plt.show()

if len(sys.argv) < 2:
	print('Please specify a scenario path')
scenario_path = sys.argv[1]
if scenario_path[-1] != '/':
	scenario_path += '/'
pressures = pd.read_csv(scenario_path + 'pressures.csv', index_col='time')
leak_properties = LeakProperties.from_json(scenario_path + 'leak_info.json')
bsi = BetweenSensorInterpolator(
	nodes_with_sensors=['4', '13', '16', '22', '31'],
	train_days=5,
	k=1,
	threshold=0
)
alarms = bsi.train_and_detect(pressures)
residual_plot(alarms, leak_properties, title=scenario_path.split('/')[-2])
plt.show()
