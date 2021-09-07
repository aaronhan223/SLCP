from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from sklearn.ensemble import RandomForestRegressor
import config
import numpy as np
from tqdm import tqdm
import pdb


class ConformalPred:
    """ Wrapper of all conformal prediciton method

    Parameters
    ----------
    ratio: float, proportion of training data used to ocalibrate
    """
    def __init__(self, model, method, ratio, x_train, x_test, y_train, y_test, k=300) -> None:
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.k = k

        if method == 'ldcp':
            local = True
            nc = RegressorNc(model, local, self.k, err_func=QuantileRegAsymmetricErrFunc(), alpha=config.ConformalParams.alpha)
        elif method == 'cqr':
            local = False
            nc = RegressorNc(model, local, self.k, err_func=QuantileRegErrFunc(), alpha=config.ConformalParams.alpha)
        else:
            local = False
            nc = RegressorNc(model, local, self.k, err_func=AbsErrorErrFunc(), alpha=config.ConformalParams.alpha)
        self.icp = IcpRegressor(nc, local, self.k, significance=config.ConformalParams.alpha)
        
        idx = np.random.permutation(config.UtilsParams.n_train)
        n_half = int(np.floor(config.UtilsParams.n_train * ratio))
        self.idx_train, self.idx_cal = idx[:n_half], idx[n_half:]

    def fit(self):
        self.icp.fit(self.x_train[self.idx_train], self.y_train[self.idx_train])
        self.icp.calibrate(self.x_train[self.idx_cal], self.y_train[self.idx_cal])

    def predict(self):
        predictions = self.icp.predict(self.x_test, significance=config.ConformalParams.alpha)
        y_lower = predictions[:, 0]
        y_upper = predictions[:, 1]
        return y_lower, y_upper