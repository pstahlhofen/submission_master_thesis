import pandas
import functools
import numpy as np
import pandas as pd

class Alarms(dict):
	'''
	Container for alarm-events found by a leakage detector.

	This class has everything in common with its parent 'dict', instead of the
	following exceptions:
	1. The object can only be initialized as an empty container.
	2. The values can only be of type pandas.core.frame.DataFrame and must
	contain a column named "times".
	3. The string representation of the object was changed to account for its
	meaning as an alarm-container.
	4. A few methods were added for convenience. See the documentation below.
	'''

	def __init__(self):
		super().__init__()
		self.error_message = (
			f'Alarms object can only have DataFrames as values'
			f' and each of the DataFrames must have a "time" column!'
		)

	def _is_ok(self, item):
		isDataFrame = isinstance(item, pandas.core.frame.DataFrame)
		return isDataFrame and 'time' in item.columns

	def __setitem__(self, key, value):
		if not self._is_ok(value):
			raise ValueError(self.error_message)
		super().__setitem__(key, value)

	def update(self, new_dict):
		''' Make sure that only DataFrames are added.'''
		if not all([self._is_ok(val) for val in new_dict.values()]):
			raise ValueError(self.error_message)
		super().update(new_dict)

	def __repr__(self):
		if not self.any():
			return 'No alarm was triggered at any node'
		res = ''
		for node_name, alarm in self.items():
			if alarm.size == 0:
				res += f'No alarm at node {node_name}\n'
			else:
				res += f'Alarms at node {node_name}\n'
				res += alarm.to_string(index=False) + '\n'
			res += '='*30 + '\n'
		return res

	def any(self):
		''' Return 'True' if any value is non-empty.'''
		return any([alarm.size > 0 for alarm in self.values()])

	def times(self):
		'''Compute the unoin of the alarm timestamps of all sensors.'''
		sensor_alarm_times = [alarm['time'] for alarm in self.values()]
		return functools.reduce(np.union1d, sensor_alarm_times)

	def any_during(self, start_time, end_time):
		'''
		Any alarm in the half-open interval [start_time, end_time)?
		'''
		for time in self.times():
			if time >= start_time and time < end_time:
				return True
		# if all alarms were outside the given time window:
		return False

	def residual_matrix(self):
		'''
		Return a pd.DataFrame of all sensor-nodes' residuals, indexed by time.

		Note: This method is only usable if all alarm DataFrames have the same
		time column, i.e. if the threshold of the leakage detector was 0.

		Returns
		--------
		a pd.DataFrame containing residuals stored in self. The columns will
		be named like self.keys().
		'''
		if not self.any():
			raise ValueError(
					'Cannot create residual matrix: No alarm was triggered.'
			)
		equal_time = lambda f1, f2:\
			f1.shape==f2.shape and (f1['time']==f2['time']).all()
		alarm_frames = list(self.values())
		for i in range(len(alarm_frames) - 1):
			if not equal_time(alarm_frames[i], alarm_frames[i+1]):
				raise ValueError(
					f'Can only create residual matrix if all alarm frames'
					f' have the same time index. This is achieved by setting'
					f' the threshold of the creating leakage detector to 0.'
				)
		residual_columns = []
		for node_name, alarm_frame in self.items():
			residual_column = alarm_frame.rename(
				columns={'residual': node_name}
			)
			residual_column.index = residual_column['time']
			residual_column.drop(columns=['time'], inplace=True)
			residual_columns.append(residual_column)
		return pd.concat(residual_columns, axis=1)
