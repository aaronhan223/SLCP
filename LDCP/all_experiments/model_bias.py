import config
from sklearn.ensemble import RandomForestRegressor
from datasets import datasets
from utils import plot_model_bias, set_seed
from conformal import ConformalPred
from tqdm import tqdm
from cqr import helper
import numpy as np


base_dataset_path = './datasets/'


def model_bias_study(gamma, random_seed):
    
    set_seed(random_seed)
    X_train, X_test, y_train, y_test = datasets.GetDataset('simulation', base_dataset_path)
    mean_estimator = RandomForestRegressor(n_estimators=config.RandomForecastParams.n_estimators, 
                                            min_samples_leaf=config.RandomForecastParams.min_samples_leaf,
                                            max_features=config.RandomForecastParams.max_features, 
                                            random_state=config.RandomForecastParams.random_state)

    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None, 
                                                               fit_params=None, 
                                                               quantiles=config.ConformalParams.quantiles, 
                                                               params=config.RandomForecastParams)
                                                               
    cp = ConformalPred(model=quantile_estimator, 
                       method='cqr', 
                       ratio=0.5, 
                       x_train=X_train, 
                       x_test=X_test, 
                       y_train=y_train, 
                       y_test=y_test, 
                       model_2=mean_estimator, 
                       gamma=gamma)

    cp_local = ConformalPred(model=quantile_estimator, 
                             method='slcp', 
                             ratio=0.5, 
                             x_train=X_train, 
                             x_test=X_test, 
                             y_train=y_train, 
                             y_test=y_test, 
                             model_2=mean_estimator, 
                             gamma=gamma, 
                             k=config.ConformalParams.k)
    cp.fit()
    cp_local.fit()
    y_lower, y_upper = cp.predict()
    y_lower_local, y_upper_local = cp_local.predict()
    in_range_local = np.sum((y_test >= y_lower_local) & (y_test <= y_upper_local)) / len(y_test) * 100
    in_range = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100

    length = np.mean(y_upper - y_lower)
    length_local = np.mean(y_upper_local - y_lower_local)
    return length, length_local, in_range, in_range_local


def run_model_bias(k=100):

    gamma_vals = np.linspace(0, 1, 11)
    length_vals, length_local_vals = [], []
    coverage_vals, coverage_local_vals = [], []
    for gamma in tqdm(gamma_vals):
        length, length_local, in_range, in_range_local = model_bias_study(gamma, config.UtilsParams.seed)
        length_vals.append(length)
        length_local_vals.append(length_local)
        coverage_vals.append(in_range)
        coverage_local_vals.append(in_range_local)
    plot_model_bias(length_vals, length_local_vals, gamma_vals, k, config.DataParams.n_train)
