from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.cp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from sklearn.ensemble import RandomForestRegressor
import config
import copy
import numpy as np
from tqdm import tqdm
import pdb


class ConformalPred:
    """ Wrapper of all conformal prediciton method

    Parameters
    ----------
    ratio: float, proportion of training data used to calibrate
    """
    def __init__(self, model, method, data_name, ratio, x_train, x_test, y_train, y_test, model_2=None, gamma=1., k=300) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

        if method == 'slcp-knn':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'slcp-rbf':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), mean=False, rbf_kernel=True, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'slcp-mean':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), rbf_kernel=True, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'cqr':
            local = False
            nc = RegressorNc(model, local, k, err_func=QuantileRegErrFunc(), alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'cqr-asy':
            local = False
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=config.ConformalParams.alpha)
        elif method == 'lacp':
            local = False
            normalizer_adapter = copy.deepcopy(model)
            normalizer = RegressorNormalizer(model, normalizer_adapter, AbsErrorErrFunc())
            nc = RegressorNc(model, local, k, err_func=AbsErrorErrFunc(), alpha=config.ConformalParams.alpha, normalizer=normalizer, beta=1)
        else:
            local = False
            nc = RegressorNc(model, local, k, err_func=AbsErrorErrFunc(), alpha=config.ConformalParams.alpha)
        self.icp = IcpRegressor(nc, local, k, significance=config.ConformalParams.alpha)
        
        if 'simulation' in data_name:
            n_train = config.DataParams.n_train
        else:
            n_train = x_train.shape[0]
        idx = np.random.permutation(n_train)
        n_half = int(np.floor(n_train * ratio))
        self.idx_train, self.idx_cal = idx[:n_half], idx[n_half:]

    def fit(self):
        self.icp.fit(self.x_train[self.idx_train], self.y_train[self.idx_train])
        self.icp.calibrate(self.x_train[self.idx_cal], self.y_train[self.idx_cal])

    def predict(self):
        predictions = self.icp.predict(self.x_test, significance=config.ConformalParams.alpha)
        y_lower = predictions[:, 0]
        y_upper = predictions[:, 1]
        return y_lower, y_upper
