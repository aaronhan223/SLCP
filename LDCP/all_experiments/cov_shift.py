import config
import numpy as np
import pandas as pd
from datasets import datasets
from conformal import ConformalPred
from utils import plot_cov_shift, set_seed
from cqr import helper
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pdb
base_dataset_path = './datasets/'


def cov_shift_study(a, b, random_seed, k=100):
    
    set_seed(random_seed)
    X_train, X_test, y_train, y_test = datasets.GetDataset('cov_shift', base_dataset_path, random_seed, 0.5, a=a, b=b)
    quantile_estimator = helper.QLR_RegressorAdapter(model=None, in_shape=1)
    quantile_estimator_local = helper.QLR_RegressorAdapter(model=None, in_shape=1)
    mean_estimator = helper.MSELR_RegressorAdapter(model=None, in_shape=1)

    cp = ConformalPred(model=quantile_estimator, 
                       method='cqr', 
                       data_name='cov_shift',
                       ratio=0.5, 
                       x_train=X_train, 
                       x_test=X_test, 
                       y_train=y_train, 
                       y_test=y_test)

    cp_local = ConformalPred(model=quantile_estimator_local, 
                             method='slcp-rbf', 
                             data_name='cov_shift',
                             ratio=0.5, 
                             x_train=X_train, 
                             x_test=X_test, 
                             y_train=y_train, 
                             y_test=y_test, 
                             k=k)

    split = ConformalPred(
        model=mean_estimator,
        method='split',
        data_name='cov_shift',
        ratio=0.5,
        x_train=X_train, 
        x_test=X_test, 
        y_train=y_train, 
        y_test=y_test
        )

    cp.fit()
    cp_local.fit()
    split.fit()

    y_lower, y_upper = cp.predict()
    y_lower_local, y_upper_local = cp_local.predict()
    y_lower_split, y_upper_split = split.predict()

    coverage_local, length_local = helper.compute_coverage(y_test, y_lower_local, y_upper_local, config.ConformalParams.alpha, 'SLCP' + " Linear Regr")
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_test, y_lower, y_upper, config.ConformalParams.alpha, 'CQR' + " Linear Regr")
    coverage_split, length_split = helper.compute_coverage(y_test, y_lower_split, y_upper_split, config.ConformalParams.alpha, 'Split' + " Linear Regr")
    return coverage_local, coverage_cp_qnet, coverage_split


def run_cov_shift(k=100):

    a_list = np.linspace(0, 1, 11)
    b_list = np.linspace(0, 1, 11)
    coverage_local_all, coverage_cp_all, coverage_split_all = [], [], []
    for a, b in zip(a_list, b_list):
        if a == 0.:
            a += .01
            b += .01
        a, b = np.round(a, 2), np.round(b, 2)
        coverage_local, coverage_cp, coverage_split = cov_shift_study(a, b, config.UtilsParams.seed)
        coverage_local_all.append(coverage_local)
        coverage_cp_all.append(coverage_cp)
        coverage_split_all.append(coverage_split)
    plot_cov_shift(coverage_cp_all, coverage_local_all, coverage_split_all, a_list, k, config.DataParams.n_train)
