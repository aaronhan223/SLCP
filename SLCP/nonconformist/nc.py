#!/usr/bin/env python

"""
Nonconformity functions.
"""

from __future__ import division

import abc
import numpy as np
import sklearn.base
import config
import pdb
from nonconformist.base import ClassifierAdapter, RegressorAdapter
from nonconformist.base import OobClassifierAdapter, OobRegressorAdapter

# -----------------------------------------------------------------------------
# Error functions
# -----------------------------------------------------------------------------


class ClassificationErrFunc(object):
	"""Base class for classification model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(ClassificationErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass


class RegressionErrFunc(object):
	"""Base class for regression model error functions.
	"""

	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(RegressionErrFunc, self).__init__()

	@abc.abstractmethod
	def apply(self, prediction, y):#, norm=None, beta=0):
		"""Apply the nonconformity function.

		Parameters
		----------
		prediction : numpy array of shape [n_samples, n_classes]
			Class probability estimates for each sample.

		y : numpy array of shape [n_samples]
			True output labels of each sample.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of the samples.
		"""
		pass

	@abc.abstractmethod
	def apply_inverse(self, nc, significance):#, norm=None, beta=0):
		"""Apply the inverse of the nonconformity function (i.e.,
		calculate prediction interval).

		Parameters
		----------
		nc : numpy array of shape [n_calibration_samples]
			Nonconformity scores obtained for conformal predictor.

		significance : float
			Significance level (0, 1).

		Returns
		-------
		interval : numpy array of shape [n_samples, 2]
			Minimum and maximum interval boundaries for each prediction.
		"""
		pass


class InverseProbabilityErrFunc(ClassificationErrFunc):
	"""Calculates the probability of not predicting the correct class.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		1 - \hat{P}(y_i | x) \, .
	"""

	def __init__(self):
		super(InverseProbabilityErrFunc, self).__init__()

	def apply(self, prediction, y):
		prob = np.zeros(y.size, dtype=np.float32)
		for i, y_ in enumerate(y):
			if y_ >= prediction.shape[1]:
				prob[i] = 0
			else:
				prob[i] = prediction[i, int(y_)]
		return 1 - prob


class MarginErrFunc(ClassificationErrFunc):
	"""
	Calculates the margin error.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		0.5 - \dfrac{\hat{P}(y_i | x) - max_{y \, != \, y_i} \hat{P}(y | x)}{2}
	"""

	def __init__(self):
		super(MarginErrFunc, self).__init__()

	def apply(self, prediction, y):
		prob = np.zeros(y.size, dtype=np.float32)
		for i, y_ in enumerate(y):
			if y_ >= prediction.shape[1]:
				prob[i] = 0
			else:
				prob[i] = prediction[i, int(y_)]
				prediction[i, int(y_)] = -np.inf
		return 0.5 - ((prob - prediction.max(axis=1)) / 2)


class AbsErrorErrFunc(RegressionErrFunc):
	"""Calculates absolute error nonconformity for regression problems.

		For each correct output in ``y``, nonconformity is defined as

		.. math::
			| y_i - \hat{y}_i |
	"""

	def __init__(self):
		super(AbsErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return np.abs(prediction - np.squeeze(y))

	def apply_inverse(self, nc, significance):
		nc = np.sort(nc)[::-1]
		border = int(np.floor(significance * (nc.size + 1))) - 1
		# TODO: should probably warn against too few calibration examples
		border = min(max(border, 0), nc.size - 1)
		return np.vstack([nc[border], nc[border]])


class SignErrorErrFunc(RegressionErrFunc):
	"""Calculates signed error nonconformity for regression problems.

	For each correct output in ``y``, nonconformity is defined as

	.. math::
		y_i - \hat{y}_i

	References
	----------
	.. [1] Linusson, Henrik, Ulf Johansson, and Tuve Lofstrom.
		Signed-error conformal regression. Pacific-Asia Conference on Knowledge
		Discovery and Data Mining. Springer International Publishing, 2014.
	"""

	def __init__(self):
		super(SignErrorErrFunc, self).__init__()

	def apply(self, prediction, y):
		return (prediction - y)

	def apply_inverse(self, nc, significance):
        
		err_high = -nc
		err_low = nc
        
		err_high = np.reshape(err_high, (nc.shape[0],1))
		err_low = np.reshape(err_low, (nc.shape[0],1))
        
		nc = np.concatenate((err_low,err_high),1)
        
		nc = np.sort(nc,0)
		index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
		index = min(max(index, 0), nc.shape[0] - 1)
		return np.vstack([nc[index,0], nc[index,1]])

# CQR symmetric error function
class QuantileRegErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression error.
    
    For each correct output in ``y``, nonconformity is defined as
    
    .. math::
        max{\hat{q}_low - y, y - \hat{q}_high}
    
    """
    def __init__(self):
        super(QuantileRegErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        error_low = y_lower - np.squeeze(y)
        error_high = np.squeeze(y) - y_upper
        err = np.maximum(error_high, error_low)
        return err

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index], nc[index]])


class QuantileRegAsymmetricErrFunc(RegressionErrFunc):
    """Calculates conformalized quantile regression asymmetric error function.
    
    For each correct output in ``y``, nonconformity is defined as
    
    .. math::
        E_low = \hat{q}_low - y
        E_high = y - \hat{q}_high
    
    """
    def __init__(self):
        super(QuantileRegAsymmetricErrFunc, self).__init__()

    def apply(self, prediction, y):
        y_lower = prediction[:, 0]
        y_upper = prediction[:, -1]
        
        error_high = np.squeeze(y) - y_upper 
        error_low = y_lower - np.squeeze(y)

        err_high = np.reshape(error_high, (y_upper.shape[0],1))
        err_low = np.reshape(error_low, (y_lower.shape[0],1))

        return np.concatenate((err_low, err_high), 1)

    def apply_inverse(self, nc, significance):
        nc = np.sort(nc, 0)
        index = int(np.ceil((1 - significance / 2) * (nc.shape[0] + 1))) - 1
        index = min(max(index, 0), nc.shape[0] - 1)
        return np.vstack([nc[index,0], nc[index,1]])
    
# -----------------------------------------------------------------------------
# Base nonconformity scorer
# -----------------------------------------------------------------------------
class BaseScorer(sklearn.base.BaseEstimator):
	__metaclass__ = abc.ABCMeta

	def __init__(self):
		super(BaseScorer, self).__init__()

	@abc.abstractmethod
	def fit(self, x, y):
		pass

	@abc.abstractmethod
	def score(self, x, y=None):
		pass


class RegressorNormalizer(BaseScorer):
	def __init__(self, base_model, normalizer_model, err_func):
		super(RegressorNormalizer, self).__init__()
		self.base_model = base_model
		self.normalizer_model = normalizer_model
		self.err_func = err_func

	def fit(self, x, y):
		residual_prediction = self.base_model.predict(x)
		residual_error = np.abs(self.err_func.apply(residual_prediction, y))

		######################################################################
		# Optional: use logarithmic function as in the original implementation
		# available in https://github.com/donlnz/nonconformist
		#
		# CODE:
		# residual_error += 0.00001 # Add small term to avoid log(0)
		# log_err = np.log(residual_error)
		######################################################################

		log_err = residual_error
		self.normalizer_model.fit(x, log_err)

	def score(self, x, y=None):

		######################################################################
		# Optional: use logarithmic function as in the original implementation
		# available in https://github.com/donlnz/nonconformist
		#
		# CODE:
		# norm = np.exp(self.normalizer_model.predict(x))
		######################################################################

		norm = np.abs(self.normalizer_model.predict(x))
		return norm


class NcFactory(object):
	@staticmethod
	def create_nc(model, err_func=None, normalizer_model=None, oob=False):
		if normalizer_model is not None:
			normalizer_adapter = RegressorAdapter(normalizer_model)
		else:
			normalizer_adapter = None

		if isinstance(model, sklearn.base.ClassifierMixin):
			err_func = MarginErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_decision_function_'):
					adapter = OobClassifierAdapter(model)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                      'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = ClassifierAdapter(model)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return ClassifierNc(adapter, err_func, normalizer)
			else:
				return ClassifierNc(adapter, err_func)

		elif isinstance(model, sklearn.base.RegressorMixin):
			err_func = AbsErrorErrFunc() if err_func is None else err_func
			if oob:
				c = sklearn.base.clone(model)
				c.fit([[0], [1]], [0, 1])
				if hasattr(c, 'oob_prediction_'):
					adapter = OobRegressorAdapter(model)
				else:
					raise AttributeError('Cannot use out-of-bag '
					                     'calibration with {}'.format(
						model.__class__.__name__
					))
			else:
				adapter = RegressorAdapter(model)

			if normalizer_adapter is not None:
				normalizer = RegressorNormalizer(adapter,
				                                 normalizer_adapter,
				                                 err_func)
				return RegressorNc(adapter, err_func, normalizer)
			else:
				return RegressorNc(adapter, err_func)


class BaseModelNc(BaseScorer):
	"""Base class for nonconformity scorers based on an underlying model.

	Parameters
	----------
	model : ClassifierAdapter or RegressorAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc or RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.
	"""
	def __init__(self, model, local, k, err_func, mean=True, rbf_kernel=False, alpha=0.1, normalizer=None, beta=1e-6, model_2=None, gamma=1.):
		super(BaseModelNc, self).__init__()
		self.err_func = err_func
		self.model = model
		self.normalizer = normalizer
		self.beta = beta
		self.local = local
		self.k = k
		self.alpha = alpha
		self.model_2 = model_2
		self.gamma = gamma
		self.kernel = rbf_kernel
		self.mean = mean

		# If we use sklearn.base.clone (e.g., during cross-validation),
		# object references get jumbled, so we need to make sure that the
		# normalizer has a reference to the proper model adapter, if applicable.
		if (self.normalizer is not None and
			hasattr(self.normalizer, 'base_model')):
			self.normalizer.base_model = self.model

		self.last_x, self.last_y = None, None
		self.last_prediction = None
		self.clean = False

	def fit(self, x, y):
		"""Fits the underlying model of the nonconformity scorer.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for fitting the underlying model.

		y : numpy array of shape [n_samples]
			Outputs of examples for fitting the underlying model.

		Returns
		-------
		None
		"""
		self.model.fit(x, y)
		if self.normalizer is not None:
			self.normalizer.fit(x, y)
		if self.model_2 is not None:
			self.model_2.fit(x, y)
		self.clean = False
		if self.local:
			self.x_ref = x
			self.error_ref = self.score(x, y)
			return self.error_ref

	def score(self, x, y=None):
		"""Calculates the nonconformity score of a set of samples.

		Parameters
		----------
		x : numpy array of shape [n_samples, n_features]
			Inputs of examples for which to calculate a nonconformity score.

		y : numpy array of shape [n_samples]
			Outputs of examples for which to calculate a nonconformity score.

		Returns
		-------
		nc : numpy array of shape [n_samples]
			Nonconformity scores of samples.
		"""
		prediction = self.model.predict(x)
		if self.local and prediction.ndim == 1:
			prediction = np.transpose(np.vstack([prediction] * 2))
		n_test = x.shape[0]
		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		if self.model_2 is not None:
			prediction_2 = self.model_2.predict(x)
			prediction_2 = np.transpose(np.vstack([prediction_2] * 2))
			prediction = (1 - self.gamma) * prediction + self.gamma * prediction_2

		if prediction.ndim > 1:
		    ret_val = self.err_func.apply(prediction, y)
		else:
			ret_val = self.err_func.apply(prediction, y) / norm
		return ret_val


# -----------------------------------------------------------------------------
# Classification nonconformity scorers
# -----------------------------------------------------------------------------
class ClassifierNc(BaseModelNc):
	"""Nonconformity scorer using an underlying class probability estimating
	model.

	Parameters
	----------
	model : ClassifierAdapter
		Underlying classification model used for calculating nonconformity
		scores.

	err_func : ClassificationErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : ClassifierAdapter
		Underlying model object.

	err_func : ClassificationErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	RegressorNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
	             err_func=MarginErrFunc(),
	             normalizer=None,
	             beta=1e-6):
		super(ClassifierNc, self).__init__(model,
		                                   err_func,
		                                   normalizer,
		                                   beta)


# -----------------------------------------------------------------------------
# Regression nonconformity scorers
# -----------------------------------------------------------------------------
class RegressorNc(BaseModelNc):
	"""Nonconformity scorer using an underlying regression model.

	Parameters
	----------
	model : RegressorAdapter
		Underlying regression model used for calculating nonconformity scores.

	err_func : RegressionErrFunc
		Error function object.

	normalizer : BaseScorer
		Normalization model.

	beta : float
		Normalization smoothing parameter. As the beta-value increases,
		the normalized nonconformity function approaches a non-normalized
		equivalent.

	Attributes
	----------
	model : RegressorAdapter
		Underlying model object.

	err_func : RegressionErrFunc
		Scorer function used to calculate nonconformity scores.

	See also
	--------
	ProbEstClassifierNc, NormalizedRegressorNc
	"""
	def __init__(self,
	             model,
				 local,
				 k,
	             err_func,
				 mean=True,
				 rbf_kernel=False,
				 alpha=0.1,
	             normalizer=None,
	             beta=1e-6,
				 model_2=None,
				 gamma=1.):
		super(RegressorNc, self).__init__(model,
										  local,
										  k,
		                                  err_func,
										  mean,
										  rbf_kernel,
										  alpha,
		                                  normalizer,
		                                  beta,
										  model_2,
										  gamma)
		self.alpha = alpha
		self.kernel = rbf_kernel
		self.mean = mean

	def get_kernel(self):
		return self.kernel
	
	def get_mean(self):
		return self.mean

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

	def slcp_rbf_weights(self, x):
		alpha_hi = 1 - config.ConformalParams.alpha / 2
		alpha_lo = 1 - config.ConformalParams.alpha / 2
		idx, weights = self.kernel_smoothing(x)
		if self.mean:
			err_ref_q = self.compute_mean(weights, self.error_ref[idx])
		else:
			err_ref_q = self.compute_quantile(alpha_hi, alpha_lo, weights, self.error_ref[idx])
		return err_ref_q

	def predict(self, x, nc, significance=None):
		"""Constructs prediction intervals for a set of test examples.

		Predicts the output of each test pattern using the underlying model,
		and applies the (partial) inverse nonconformity function to each
		prediction, resulting in a prediction interval for each test pattern.

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
		p : numpy array of shape [n_samples, 2] or [n_samples, 2, 99]
			If significance is ``None``, then p contains the interval (minimum
			and maximum boundaries) for each test pattern, and each significance
			level (0.01, 0.02, ..., 0.99). If significance is a float between
			0 and 1, then p contains the prediction intervals (minimum and
			maximum	boundaries) for the set of test patterns at the chosen
			significance level.
		"""
		n_test = x.shape[0]
		prediction = self.model.predict(x)
		if self.model_2 is not None:
			prediction_2 = self.model_2.predict(x)
			prediction_2 = np.transpose(np.vstack([prediction_2] * 2))
			prediction = (1 - self.gamma) * prediction + self.gamma * prediction_2

		if self.normalizer is not None:
			norm = self.normalizer.score(x) + self.beta
		else:
			norm = np.ones(n_test)

		if significance:
			intervals = np.zeros((x.shape[0], 2))
			# err_dist = self.err_func.apply_inverse(nc, significance) # FIXME: assymetric
			if self.local:
				if self.get_kernel():
					err_ref_q = self.slcp_rbf_weights(x)
				else:
					err_ref_q = self.slcp_equal_weights(x)
				# this is for symmetric case
				ErrFunc = QuantileRegErrFunc()
				d = ErrFunc.apply_inverse(nc=nc, significance=significance)
				err_dist = np.vstack([d[0, 0] + err_ref_q[:, 0], d[1, 0] + err_ref_q[:, 1]])
			else:
				err_dist = self.err_func.apply_inverse(nc, significance)
				err_dist = np.hstack([err_dist] * n_test)

			if prediction.ndim > 1: # CQR, quantile high and low predictions
				intervals[:, 0] = prediction[:,0] - err_dist[0, :]
				intervals[:, 1] = prediction[:,-1] + err_dist[1, :]
			else: # regular conformal prediction
				err_dist *= norm
				intervals[:, 0] = prediction - err_dist[0, :]
				intervals[:, 1] = prediction + err_dist[1, :]

			return intervals
		else: # Not tested for CQR
			significance = np.arange(0.01, 1.0, 0.01)
			intervals = np.zeros((x.shape[0], 2, significance.size))

			for i, s in enumerate(significance):
				err_dist = self.err_func.apply_inverse(nc, s)
				err_dist = np.hstack([err_dist] * n_test)
				err_dist *= norm

				intervals[:, 0, i] = prediction - err_dist[0, :]
				intervals[:, 1, i] = prediction + err_dist[0, :]

			return intervals
