from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

from time_constants import SECONDS_PER_DAY, HOURS_PER_DAY
import wn_util
from alarms import Alarms

def train_test_split(X, y, split_index):
	'''
	Split predictors and targets at some index.

	Parameters
	-----------

	X, y: np.ndarray
	classification predictor variables and target variables, respectively

	split_index: The index at which X and y should be split

	Returns
	--------
	X_train, X_test, y_train, y_test: np.ndarray
	the training predictors, test predictors, training targets and
	test targets, respectively
	'''
	X_train, X_test, y_train, y_test = (
		X[:split_index], X[split_index:],
		y[:split_index], y[split_index:]
	)
	return X_train, X_test, y_train, y_test

class AbstractLeakageDetector():
	'''
	This class is not meant to be instantiated directly.

	Please see the documentation of its childrens constructors for help.
	'''

	def __init__(self, nodes_with_sensors, train_days, k, threshold):
		self.nodes_with_sensors = nodes_with_sensors
		self.train_days = train_days
		self.k = k
		self.threshold = threshold

	def compute_alarm(self, residuals, timestamps, time_offset):
		'''
		Helper method to generate the alarm DataFrame

		Parameters
		-----------

		residuals: np.ndarray
		differences between predicted and actual pressure values

		timestamps: array-like, time values in seconds
		note that timestamps must span the whole simulation time, hence it may
		be longer than residuals, which only spans the test time

		time_offset: int, time in seconds
		how many seconds have already been used for training?

		Returns
		--------
		alarm: pd.DataFrame with columns 'time' (in seconds) and 'residual'
		this will contain only those instances where the residuals exceeded
		self.threshold.
		'''
		alarm_idxs = np.flatnonzero(residuals > self.threshold)
		residuals = residuals[alarm_idxs]
		alarm_idxs += time_offset
		alarm_times = timestamps[alarm_idxs]
		alarm = pd.DataFrame(
			dict(
				time=alarm_times,
				residual=residuals
			)
		)
		return alarm

	def set_overlap_pressures(self, pressures, overlap_steps):
		'''
		Store the last pressure values used for training.

		This is useful if one uses previous timesteps and wants to predict
		some values immediately after the training period. The last few
		pressure values from the training set can then later be re-used for
		the prediction of the first few test values.

		Parameters
		----------

		pressures: pd.DataFrame
		training pressure values

		overlap_steps: int
		how many of the last pressure values should be safed

		This sets self.overlap_pressures
		'''
		timestep = pressures.index[1] - pressures.index[0]
		last_measuring_time = pressures.index[-1]
		overlap_pressures = pressures.tail(overlap_steps)
		overlap_pressures.index -= (last_measuring_time + timestep)
		self.overlap_pressures = overlap_pressures

class SingleSensorForecaster(AbstractLeakageDetector):
	'''
	Linear Forecaster for pressure values based on a fixed number of timesteps

	For M different nodes and a fixed number of timesteps k, this detecter
	uses M linear models which each use the last k timesteps for their
	corresponding node in order to predict the next timestep.

	Parameters
	-----------

	nodes_with_sensors: list of str
	the names of junctions in a wntr.network.WaterNetworkModel that should be
	equipped with virtual pressure sensors.

	train_days: int
	this is used in self.train_and_detect to determine how many of the rows of
	a given pressure matrix should be used for training the models. The actual
	leakage detection will be performed on the remaining part.
	For example, for a water network with pressure measurements
	every 30 minutes, train_days=2 would mean that the first 48*2=96 rows of
	the pressure matrix are used for training.

	k: int
	number of timesteps to use for pressure detecting

	threshold: int or float, must be positive
	the threshold for the generation of alarms
	It's unit is meter. (pressure = meter * gravity * waterdensity)
	'''

	def __init__(self, nodes_with_sensors, train_days, k, threshold):
		super().__init__(nodes_with_sensors, train_days, k, threshold)
		self.scalers = dict()
		self.models = dict()

	def k_step_predictors(self, X):
		'''
		Split columns of X into k-shingles.

		This method is used to construct predictors
		for self.detect.

		Parameters
		-----------

		X: np.ndarray
		matrix of shape N X M

		Returns
		--------
		A List with M elements. Each element is an np.ndarray
		with shape N-self.k x self.k

		Note:
		The last elements of each column are not returned, as they are
		only targets and no predictors.
		'''
		N, M = X.shape
		res = []
		for m in range(M):
			k_steps = np.row_stack(
				[X[n:n+self.k, m] for n in range(N-self.k)]
			)
			res.append(k_steps)
		return res

	def k_step_targets(self, X):
		'''
		Give shortened columns of X as a list, dropping the k first elements.
		'''
		N, M = X.shape
		return [X[self.k:N,m] for m in range(M)]

	def train(self, pressures, set_overlap=False):
		'''
		Learn linear predictions for self.nodes_with_sensors.

		For M different nodes, this method trains M linear models which use
		the last self.k timesteps for their corresponding node in order to
		predict the next timestep. The predictors are normalized with a
		separate StandardScaler for each model. The scalers are stored in
		self.scalers and the models are stored in self.models.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		set_overlap: bool, optional, default=False
		if True, this will cause the last self.k rows of pressures to be
		stored in self.overlap_pressures. These can later be utilized by
		setting use_overlap=True in self.detect. This has the advantage that a
		leak starting at the very first timestep after training can already be
		found.
		
		This method sets self.models and self.scalers, but it has no return
		value.
		'''
		if set_overlap:
			self.set_overlap_pressures(pressures, self.k)
		pressures = pressures[self.nodes_with_sensors].to_numpy()
		Xs = self.k_step_predictors(pressures)
		ys = self.k_step_targets(pressures)
		for X, y, node_name in zip(Xs, ys, self.nodes_with_sensors):
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[node_name] = scaler
			X = scaler.transform(X)
			model = LinearRegression()
			model.fit(X, y)
			self.models[node_name] = model

	def detect(self, pressures, use_overlap=False):
		'''
		Use the learned forecasting models to detect leakages.

		Note: self.scalers and self.models must be filled by self.train before
		this method can be used.

		For each node name in self.nodes_with_sensors self.models[node_name]
		is used to predict its pressure values after the predictors (pressure
		values from previous timesteps) have been scaled with
		self.scalers[node_name]. If the difference between prediction and
		actual pressure value exceeds self.threshold, an alarm is generated.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		use_overlap: bool, optional, default=False
		if True and if set_overlap has been set to true in self.train,
		self.overlap pressures are prepended to pressures before starting the
		leakage detection. The index values of pressures remain in tact.
		self.overlap pressures will get matching negative timesteps as
		indices. use_overlap=True has the advantage that the last elements of
		the training set can already be used for forecasting during the first
		couple of timesteps.

		Returns
		--------
		A dictionary-like alarms.Alarms object.
		Keys: junction names
		Values: pd.DataFrames containing timestamp and residual for each alarm
		'''
		if use_overlap:
			try:
				pressures = pd.concat((self.overlap_pressures, pressures))
			except AttributeError:
				print(
					f'You cannot use overlap pressures '
					f'if none were set during training!'
				)
				exit()
		timesteps = pressures.index
		pressures = pressures[self.nodes_with_sensors].to_numpy()
		Xs = self.k_step_predictors(pressures)
		ys = self.k_step_targets(pressures)
		alarms_at_nodes = Alarms()
		for X, y, node_name in zip(Xs, ys, self.nodes_with_sensors):
			scaler = self.scalers[node_name]
			X = scaler.transform(X)
			model = self.models[node_name]
			residuals = np.abs(model.predict(X) - y)
			alarm_idxs = np.flatnonzero(residuals > self.threshold)
			residuals = residuals[alarm_idxs]
			# The first possible alarm can be rasied at the k+1th timestep
			alarm_times = timesteps[alarm_idxs + self.k]
			alarm = pd.DataFrame(dict(time=alarm_times, residual=residuals))
			alarms_at_nodes[node_name] = alarm
		return alarms_at_nodes

	def train_and_detect(self, pressures):
		'''
		Train pressure forecasting models and use them for leakage detection.

		self.train_days determines how many rows of the pressure
		matrix are used to train the models. Leakage detection is performed on
		the remaining part.

		Note: There is an overlap between training and test set because the
		models always require the k previous timesteps for the prediction.
		Hence, the last k rows of the training set will also be included in
		the test set to detect leaks starting at the very beginning of the
		test set.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		Returns
		--------
		A dictionary-like alarms.Alarms object.
		Keys: junction names
		Values: pd.DataFrames containing timestamp and residual for each alarm
		'''
		# I need to subtract 1 because pandas slices include the upper end
		train_seconds = self.train_days * SECONDS_PER_DAY - 1
		self.train(pressures.loc[:train_seconds])
		offset = self.k * (pressures.index[1] - pressures.index[0])
		alarms = self.detect(pressures.loc[train_seconds-offset:])
		return alarms

class BetweenSensorInterpolator(AbstractLeakageDetector):
	'''
	Linear interpolator of pressure values between observed nodes

	For each of the given nodes, this will build a model to predict the
	pressure values at that node based on the pressure values of all the other
	nodes. Hence, for M nodes, M different models will be trained. Each of the
	models receives an M-1 dimensional input vector and outputs a single
	scalar value at each timestep.

	Important: As opposed to the SingleSensorForecaster, this detector does
	not predict values that may occur in the future. Rather, it tries to
	interpolate values that occured at other nodes, but within the same
	timestep.

	One may specify a window size k to average the input pressure values to
	the model with the pressure values of the k-1 previous timesteps. See
	paper "One explanation to rule them all" for a formal description.

	Parameters
	-----------

	nodes_with_sensors: list of str, junction names
	observed nodes used to construct the linear models

	train_days: int
	this is used in self.train_and_detect to determine how many of the rows of
	a given pressure matrix should be used for training the models. The actual
	leakage detection will be performed on the remaining part.
	For example, for a water network with pressure measurements
	every 30 minutes, train_days=2 would mean that the first 48*2=96 rows of
	the pressure matrix are used for training.

	k: int, number of timesteps to average for the classificaiton input
	use 1 to take only the current timestep into account

	threshold: int or float, must be positive
	if the difference between a model's prediction and the actual pressure
	value exceeds the threshold, an alarm is generated. The unit of the
	threshold is meter. (pressure = meter * gravity * waterdensity)
	'''

	def __init__(self, nodes_with_sensors, train_days, k, threshold):
		super().__init__(nodes_with_sensors, train_days, k, threshold)
		self.scalers = dict()
		self.models = dict()

	def k_step_averages(self, X):
		'''Average self.k successive rows of a matrix together.'''
		N = X.shape[0]
		res = np.row_stack(
			[X[n:n+self.k, :].mean(axis=0) for n in range(N-self.k+1)]
		)
		return res

	def k_step_concats(self, X):
		'''Concatenate self.k successive rows of a matrix together.'''
		N =  X.shape[0]
		res = np.row_stack(
			[X[n:n+self.k, :].flatten() for n in range(N-self.k+1)]
		)
		return res

	def train(self, pressures, set_overlap=False):
		'''
		Learn a linear interpolation between self.nodes_with_sensors.

		For M different nodes, this will create M linear models, each of which
		tries to predict the pressure value of one node based on the pressure
		values of the other nodes. Predictor and prediction generally belong
		to the same timestep, unless the predictors were averaged with
		previous timesteps (see self.k). After the averaging (if any) the
		predictors are normalized with a separate StandardScaler for each
		model. The scalers are stored in self.scalers and the models are
		stored in self.models. Note that self.scaler[my_node_name] yields a
		scaler that was used to scale pressure values of all nodes except
		my_node. 

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		set_overlap: bool, optional, default=False
		if True and if self.k > 1, this will cause the last self.k-1 rows of
		pressures to be stored in self.overlap_pressures. These can later be
		utilized by setting use_overlap=True in self.detect. This has the
		advantage that a leak starting at the very first timestep after
		training can already be found, even when avering over previous
		timesteps.
		
		This method sets self.models and self.scalers, but it has no return
		value.
		'''
		if set_overlap:
			assert self.k > 1, (
				f'You cannot set overlap pressures because no overlap is used'
				f' in this detector!'
			)
			self.set_overlap_pressures(pressures, self.k-1)
		pressures = pressures[self.nodes_with_sensors]
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = StandardScaler()
			scaler.fit(X)
			self.scalers[predicted_node_name] = scaler
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k-1:]
			model = LinearRegression()
			model.fit(X, y)
			self.models[predicted_node_name] = model

	def detect(self, pressures, use_overlap=False):
		'''
		Use the learned interpolation to detect leakages.

		Note: self.scalers and self.models must be filled by self.train before
		this method can be used.

		For each node name in self.nodes_with_sensors self.models[node_name]
		is used to predict its pressure values after the predictors (pressure
		values from other nodes) have been scaled with
		self.scalers[node_name]. If the difference between prediction and
		actual pressure value exceeds self.threshold, an alarm is generated.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		use_overlap: bool, optional, default=False
		if True and if set_overlap has been set to True in self.train,
		self.overlap pressures are prepended to pressures before starting the
		leakage detection. The index values of pressures remain in tact.
		self.overlap pressures will get matching negative timesteps as
		indices. use_overlap=True has the advantage that the last elements of
		the training set can already be used for averaging during the first
		couple of timesteps.

		Returns
		--------
		A dictionary-like alarms.Alarms object.
		Keys: junction names
		Values: pd.DataFrames containing timestamp and residual for each alarm
		'''
		if not (self.models and self.scalers):
			raise RuntimeError('You cannot detect leakages before training!')
		if use_overlap:
			try:
				pressures = pd.concat((self.overlap_pressures, pressures))
			except AttributeError:
				print(
					f'You cannot use overlap pressures '
					f'if none were set during training!'
				)
				exit()
		timesteps = pressures.index
		pressures = pressures[self.nodes_with_sensors]
		alarms_at_nodes = Alarms()
		for predicted_node_name in self.nodes_with_sensors:
			X = pressures.drop(columns=[predicted_node_name]).to_numpy()
			if self.k > 1:
				X = self.k_step_averages(X)
			scaler = self.scalers[predicted_node_name]
			X = scaler.transform(X)
			y = pressures[predicted_node_name].to_numpy()
			y = y[self.k-1:]
			model = self.models[predicted_node_name]
			residuals = np.abs(model.predict(X) - y)
			alarm_idxs = np.flatnonzero(residuals > self.threshold)
			residuals = residuals[alarm_idxs]
			# The first possible alarm can be rasied at the k-th timestep
			alarm_times = timesteps[alarm_idxs + self.k-1]
			alarm = pd.DataFrame(dict(time=alarm_times, residual=residuals))
			alarms_at_nodes[predicted_node_name] = alarm
		return alarms_at_nodes

	def train_and_detect(self, pressures):
		'''
		Train linear interpolation models and use them for leakage detection.

		self.train_days determines how many rows of the pressure
		matrix are used to train the models. Leakage detection is performed on
		the remaining part.

		Note: If self.k > 1, there is an overlap between training and test set
		because the models always require the k-1 previous timesteps for
		averaging. Hence, the last k-1 rows of the training set will also be
		included in the test set to detect leaks starting at the very
		beginning of the test set.

		Parameters
		-----------

		pressures: pd.DataFrame
		network pressure values

		Returns
		--------
		A dictionary-like alarms.Alarms object.
		Keys: junction names
		Values: pd.DataFrames containing timestamp and residual for each alarm
		'''
		# I need to subtract 1 because pandas slices include the upper end
		train_seconds = self.train_days * SECONDS_PER_DAY - 1
		self.train(pressures.loc[:train_seconds])
		overlap = (self.k-1) * (pressures.index[1] - pressures.index[0])
		alarms = self.detect(pressures.loc[train_seconds-overlap:])
		return alarms

