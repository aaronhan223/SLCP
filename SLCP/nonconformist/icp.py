#!/usr/bin/env python

"""
Inductive conformal predictors.
"""

from __future__ import division
from collections import defaultdict
from functools import partial
import config
import numpy as np
from sklearn.base import BaseEstimator
import pdb

from nonconformist.base import RegressorMixin, ClassifierMixin
from nonconformist.util import calc_p


# -----------------------------------------------------------------------------
# Base inductive conformal predictor
# -----------------------------------------------------------------------------
class BaseIcp(BaseEstimator):
	"""Base class for inductive conformal predictors.
	"""
	def __init__(self, nc_function, local, k, significance=0.1, condition=None):
		self.cal_x, self.cal_y = None, None
		self.nc_function = nc_function
		self.alpha = significance
		self.local = local
		self.k = k

		# Check if condition-parameter is the default function (i.e.,
		# lambda x: 0). This is so we can safely clone the object without
		# the clone accidentally having self.conditional = True.
		default_condition = lambda x: 0
		is_default = (callable(condition) and
		              (condition.__code__.co_code ==
		               default_condition.__code__.co_code))

		if is_default:
			self.condition = condition
			self.conditional = False
		elif callable(condition):
			self.condition = condition
			self.conditional = True
		else:
			self.condition = lambda x: 0
			self.conditional = False

	def fit(self, x, y):
		"""Fit underlying nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the nonconformity scorer.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the nonconformity scorer.

		Returns
		-------
		None
		"""
		# TODO: incremental?
		if self.local:
			self.error_ref = self.nc_function.fit(x, y)
			self.x_ref = x
		else:
			self.nc_function.fit(x, y)

	def knn(self, x):
		'''
		Compute k-th nearest neighbours of given sample in the reference dataset.
		'''
		diff = x[:, None, :] - self.x_ref[None, :, :]
		dist = np.sum(diff ** 2, axis=-1)
		idx = np.argsort(dist, axis=-1)[:, :self.k]
		return idx

	def kernel_smoothing(self, x):
		diff = x[:, None, :] - self.x_ref[None, :, :]
		dist = np.sum(diff ** 2, axis=-1)
		idx = np.argsort(dist, axis=-1)[:, :self.k]
		h = np.quantile(dist, 0.5) / np.log(diff.shape[1])
		# h = np.quantile(np.sort(dist, axis=-1)[:, :self.k], 0.5) / np.log(self.k)
		weights = np.exp(-dist / h)
		weights[:, ::-1].sort(axis=1)
		final_weights = weights[:, :self.k]
		final_weights = final_weights / np.expand_dims(np.sum(final_weights, axis=1), axis=1)
		return idx, final_weights
	
	def slcp_equal_weights(self, x):
		alpha_hi = 1 - config.ConformalParams.alpha / 2
		alpha_lo = 1 - config.ConformalParams.alpha / 2
		idx = self.knn(x)
		err_ref = np.sort(self.error_ref[idx], 1)
		err_ref_q = np.zeros((err_ref.shape[0], err_ref.shape[2]))
		q_hi = int(np.ceil(alpha_hi * (err_ref.shape[1] + 1))) - 1
		q_lo = int(np.ceil(alpha_lo * (err_ref.shape[1] + 1))) - 1
		q_hi = min(max(q_hi, 0), err_ref.shape[1] - 1)
		q_lo = min(max(q_lo, 0), err_ref.shape[1] - 1)
		err_ref_q[:, 0] = err_ref[:, q_lo, 0]
		err_ref_q[:, 1] = err_ref[:, q_hi, 1]
		return err_ref_q

	def compute_quantile(self, alpha_hi, alpha_lo, weights, err):
		nc_sorted = np.argsort(err, axis=1)
		nc_sorted_lo, nc_sorted_hi = nc_sorted[:,:,0], nc_sorted[:,:,1]
		weights_sorted_lo = np.array(list(map(lambda x, y: y[x], nc_sorted_lo, weights)))
		weights_sorted_hi = np.array(list(map(lambda x, y: y[x], nc_sorted_hi, weights)))
		threshold_lo = np.sum(np.cumsum(weights_sorted_lo, axis=1) <= alpha_lo, axis=1)
		threshold_hi = np.sum(np.cumsum(weights_sorted_hi, axis=1) <= alpha_hi, axis=1)
		err = np.sort(err, 1)
		err_lo, err_hi = err[:, :, 0], err[:, :, 1]
		err_ref_q = np.zeros((err.shape[0], err.shape[2]))
		err_ref_q[:, 0] = err_lo[np.arange(len(err_lo)), threshold_lo]
		err_ref_q[:, 1] = err_hi[np.arange(len(err_hi)), threshold_hi]
		return err_ref_q

	def compute_mean(self, weights, err):
		err_lo, err_hi = err[:, :, 0], err[:, :, 1]
		err_ref_q = np.zeros((err.shape[0], err.shape[2]))
		err_ref_q[:, 0] = np.sum(weights * err_lo, axis=1)
		err_ref_q[:, 1] = np.sum(weights * err_hi, axis=1)
		return err_ref_q

	def slcp_rbf_weights(self, x, mean):
		alpha_hi = 1 - config.ConformalParams.alpha / 2
		alpha_lo = 1 - config.ConformalParams.alpha / 2
		idx, weights = self.kernel_smoothing(x)
		if mean:
			err_ref_q = self.compute_mean(weights, self.error_ref[idx])
		else:
			err_ref_q = self.compute_quantile(alpha_hi, alpha_lo, weights, self.error_ref[idx])
		return err_ref_q

	def calibrate(self, x, y, increment=False):
		"""Calibrate conformal predictor based on underlying nonconformity
		scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for calibrating the conformal predictor.

		y : numpy array of shape [n_samples, n_features]
			Outputs of examples for calibrating the conformal predictor.

		increment : boolean
			If ``True``, performs an incremental recalibration of the conformal
			predictor. The supplied ``x`` and ``y`` are added to the set of
			previously existing calibration examples, and the conformal
			predictor is then calibrated on both the old and new calibration
			examples.

		Returns
		-------
		None
		"""
		self._calibrate_hook(x, y, increment)
		self._update_calibration_set(x, y, increment)
		if self.conditional:
			category_map = np.array([self.condition((x[i, :], y[i])) for i in range(y.size)])
			self.categories = np.unique(category_map)
			self.cal_scores = defaultdict(partial(np.ndarray, 0))

			for cond in self.categories:
				idx = category_map == cond
				cal_scores = self.nc_function.score(self.cal_x[idx, :], self.cal_y[idx])
				self.cal_scores[cond] = np.sort(cal_scores,0)[::-1]
		else:
			self.categories = np.array([0])
			cal_scores = self.nc_function.score(self.cal_x, self.cal_y)
			if self.local:
				if self.nc_function.get_kernel():
					err_ref_q = self.slcp_rbf_weights(x, self.nc_function.get_mean())
				else:
					err_ref_q = self.slcp_equal_weights(x)
				cal_scores = np.maximum(cal_scores[:, 0] - err_ref_q[:, 0], cal_scores[:, 1] - err_ref_q[:, 1])
			self.cal_scores = {0: np.sort(cal_scores, 0)[::-1]}

	def _calibrate_hook(self, x, y, increment):
		pass

	def _update_calibration_set(self, x, y, increment):
		if increment and self.cal_x is not None and self.cal_y is not None:
			self.cal_x = np.vstack([self.cal_x, x])
			self.cal_y = np.hstack([self.cal_y, y])
		else:
			self.cal_x, self.cal_y = x, y


# -----------------------------------------------------------------------------
# Inductive conformal classifier
# -----------------------------------------------------------------------------
class IcpClassifier(BaseIcp, ClassifierMixin):
	"""Inductive conformal classifier.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``
		and ``calc_nc(x, y)``.

	smoothing : boolean
		Decides whether to use stochastic smoothing of p-values.

	Attributes
	----------
	cal_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of calibration set.

	cal_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	classes : numpy array of shape [n_classes]
		List of class labels, with indices corresponding to output columns
		 of IcpClassifier.predict()

	See also
	--------
	IcpRegressor

	References
	----------
	.. [1] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
		intervals with regression neural networks. Neural Networks, 24(8),
		842-851.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_iris
	>>> from sklearn.tree import DecisionTreeClassifier
	>>> from nonconformist.base import ClassifierAdapter
	>>> from nonconformist.icp import IcpClassifier
	>>> from nonconformist.nc import ClassifierNc, MarginErrFunc
	>>> iris = load_iris()
	>>> idx = np.random.permutation(iris.target.size)
	>>> train = idx[:int(idx.size / 3)]
	>>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
	>>> test = idx[int(2 * idx.size / 3):]
	>>> model = ClassifierAdapter(DecisionTreeClassifier())
	>>> nc = ClassifierNc(model, MarginErrFunc())
	>>> icp = IcpClassifier(nc)
	>>> icp.fit(iris.data[train, :], iris.target[train])
	>>> icp.calibrate(iris.data[cal, :], iris.target[cal])
	>>> icp.predict(iris.data[test, :], significance=0.10)
	...             # doctest: +SKIP
	array([[ True, False, False],
		[False,  True, False],
		...,
		[False,  True, False],
		[False,  True, False]], dtype=bool)
	"""
	def __init__(self, nc_function, condition=None, smoothing=True):
		super(IcpClassifier, self).__init__(nc_function, condition)
		self.classes = None
		self.smoothing = smoothing

	def _calibrate_hook(self, x, y, increment=False):
		self._update_classes(y, increment)

	def _update_classes(self, y, increment):
		if self.classes is None or not increment:
			self.classes = np.unique(y)
		else:
			self.classes = np.unique(np.hstack([self.classes, y]))

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float or None
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then the p-values
			are output rather than the predictions.

		Returns
		-------
		p : numpy array of shape [n_samples, n_classes]
			If significance is ``None``, then p contains the p-values for each
			sample-class pair; if significance is a float between 0 and 1, then
			p is a boolean array denoting which labels are included in the
			prediction sets.
		"""
		# TODO: if x == self.last_x ...
		n_test_objects = x.shape[0]
		p = np.zeros((n_test_objects, self.classes.size))

		ncal_ngt_neq = self._get_stats(x)

		for i in range(len(self.classes)):
			for j in range(n_test_objects):
				p[j, i] = calc_p(ncal_ngt_neq[j, i, 0],
				                 ncal_ngt_neq[j, i, 1],
				                 ncal_ngt_neq[j, i, 2],
				                 self.smoothing)

		if significance is not None:
			return p > significance
		else:
			return p

	def _get_stats(self, x):
		n_test_objects = x.shape[0]
		ncal_ngt_neq = np.zeros((n_test_objects, self.classes.size, 3))
		for i, c in enumerate(self.classes):
			test_class = np.zeros(x.shape[0], dtype=self.classes.dtype)
			test_class.fill(c)

			# TODO: maybe calculate p-values using cython or similar
			# TODO: interpolated p-values

			# TODO: nc_function.calc_nc should take X * {y1, y2, ... ,yn}
			test_nc_scores = self.nc_function.score(x, test_class)
			for j, nc in enumerate(test_nc_scores):
				cal_scores = self.cal_scores[self.condition((x[j, :], c))][::-1]
				n_cal = cal_scores.size

				idx_left = np.searchsorted(cal_scores, nc, 'left')
				idx_right = np.searchsorted(cal_scores, nc, 'right')

				ncal_ngt_neq[j, i, 0] = n_cal
				ncal_ngt_neq[j, i, 1] = n_cal - idx_right
				ncal_ngt_neq[j, i, 2] = idx_right - idx_left

		return ncal_ngt_neq

	def predict_conf(self, x):
		"""Predict the output values for a set of input patterns, using
		the confidence-and-credibility output scheme.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		Returns
		-------
		p : numpy array of shape [n_samples, 3]
			p contains three columns: the first column contains the most
			likely class for each test pattern; the second column contains
			the confidence in the predicted class label, and the third column
			contains the credibility of the prediction.
		"""
		p = self.predict(x, significance=None)
		label = p.argmax(axis=1)
		credibility = p.max(axis=1)
		for i, idx in enumerate(label):
			p[i, idx] = -np.inf
		confidence = 1 - p.max(axis=1)

		return np.array([label, confidence, credibility]).T


# -----------------------------------------------------------------------------
# Inductive conformal regressor
# -----------------------------------------------------------------------------
class IcpRegressor(BaseIcp, RegressorMixin):
	"""Inductive conformal regressor.

	Parameters
	----------
	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity of
		calibration examples and test patterns. Should implement ``fit(x, y)``,
		``calc_nc(x, y)`` and ``predict(x, nc_scores, significance)``.

	Attributes
	----------
	cal_x : numpy array of shape [n_cal_examples, n_features]
		Inputs of calibration set.

	cal_y : numpy array of shape [n_cal_examples]
		Outputs of calibration set.

	nc_function : BaseScorer
		Nonconformity scorer object used to calculate nonconformity scores.

	See also
	--------
	IcpClassifier

	References
	----------
	.. [1] Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).
		Inductive confidence machines for regression. In Machine Learning: ECML
		2002 (pp. 345-356). Springer Berlin Heidelberg.

	.. [2] Papadopoulos, H., & Haralambous, H. (2011). Reliable prediction
		intervals with regression neural networks. Neural Networks, 24(8),
		842-851.

	Examples
	--------
	>>> import numpy as np
	>>> from sklearn.datasets import load_boston
	>>> from sklearn.tree import DecisionTreeRegressor
	>>> from nonconformist.base import RegressorAdapter
	>>> from nonconformist.icp import IcpRegressor
	>>> from nonconformist.nc import RegressorNc, AbsErrorErrFunc
	>>> boston = load_boston()
	>>> idx = np.random.permutation(boston.target.size)
	>>> train = idx[:int(idx.size / 3)]
	>>> cal = idx[int(idx.size / 3):int(2 * idx.size / 3)]
	>>> test = idx[int(2 * idx.size / 3):]
	>>> model = RegressorAdapter(DecisionTreeRegressor())
	>>> nc = RegressorNc(model, AbsErrorErrFunc())
	>>> icp = IcpRegressor(nc)
	>>> icp.fit(boston.data[train, :], boston.target[train])
	>>> icp.calibrate(boston.data[cal, :], boston.target[cal])
	>>> icp.predict(boston.data[test, :], significance=0.10)
	...     # doctest: +SKIP
	array([[  5. ,  20.6],
		[ 15.5,  31.1],
		...,
		[ 14.2,  29.8],
		[ 11.6,  27.2]])
	"""
	def __init__(self, nc_function, local, k, significance=0.1, condition=None):
		super(IcpRegressor, self).__init__(nc_function, local, k, significance, condition)

	def predict(self, x, significance=None):
		"""Predict the output values for a set of input patterns.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of patters for which to predict output values.

		significance : float
			Significance level (maximum allowed error rate) of predictions.
			Should be a float between 0 and 1. If ``None``, then intervals for
			all significance levels (0.01, 0.02, ..., 0.99) are output in a
			3d-matrix.

		Returns
		-------
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99}
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		# TODO: interpolated p-values

		n_significance = (99 if significance is None
		                  else np.array(significance).size)

		if n_significance > 1:
			prediction = np.zeros((x.shape[0], 2, n_significance))
		else:
			prediction = np.zeros((x.shape[0], 2))

		condition_map = np.array([self.condition((x[i, :], None))
		                          for i in range(x.shape[0])])
		for condition in self.categories:
			idx = condition_map == condition
			if np.sum(idx) > 0:
				p = self.nc_function.predict(x[idx, :], self.cal_scores[condition], significance)
				if n_significance > 1:
					prediction[idx, :, :] = p
				else:
					prediction[idx, :] = p

		return prediction


class OobCpClassifier(IcpClassifier):
	def __init__(self,
	             nc_function,
	             condition=None,
	             smoothing=True):
		super(OobCpClassifier, self).__init__(nc_function,
		                                      condition,
		                                      smoothing)

	def fit(self, x, y):
		super(OobCpClassifier, self).fit(x, y)
		super(OobCpClassifier, self).calibrate(x, y, False)

	def calibrate(self, x, y, increment=False):
		# Should throw exception (or really not be implemented for oob)
		pass


class OobCpRegressor(IcpRegressor):
	def __init__(self,
				 nc_function,
				 condition=None):
		super(OobCpRegressor, self).__init__(nc_function,
											  condition)

	def fit(self, x, y):
		super(OobCpRegressor, self).fit(x, y)
		super(OobCpRegressor, self).calibrate(x, y, False)

	def calibrate(self, x, y, increment=False):
		# Should throw exception (or really not be implemented for oob)
		pass
