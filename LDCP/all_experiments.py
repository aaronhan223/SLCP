import os
import sys
import config
import numpy as np
import pandas as pd
import torch
import random
from datasets import datasets
from conformal import ConformalPred
from utils import plot_pred, plot_model_bias, plot_cov_shift, set_seed
from cqr import helper
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import pdb


pd.set_option('precision', 3)
base_dataset_path = './datasets/'


def run_pred_experiment(dataset_name, model_name, method_name, random_seed):

    set_seed(random_seed)
    try:
        X_train, X_test, y_train, y_test = datasets.GetDataset(dataset_name, base_dataset_path)
    except:
        print("CANNOT LOAD DATASET!")
        return

    if model_name == 'random_forest':
        if method_name == 'split':
            model = RandomForestRegressor(n_estimators=config.RandomForecastParams.n_estimators, 
                                          min_samples_leaf=config.RandomForecastParams.min_samples_leaf,
                                          max_features=config.RandomForecastParams.max_features, 
                                          random_state=config.RandomForecastParams.random_state)
        else:
            model = helper.QuantileForestRegressorAdapter(model=None, 
                                                          fit_params=None, 
                                                          quantiles=config.ConformalParams.quantiles, 
                                                          params=config.RandomForecastParams)
    elif model_name == 'linear_regression':
        if method_name == 'split':
            model = helper.MSELR_RegressorAdapter(model=None)
        else:
            model = helper.Linear_RegressorAdapter(model=None)

    elif model_name == 'neural_network':
        if method_name == 'split':
            model = helper.MSENet_RegressorAdapter(model=None)
        else:
            model = helper.AllQNet_RegressorAdapter(model=None)

    cp = ConformalPred(model=model, method=method_name, ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, k=300)
    cp.fit()
    y_lower, y_upper = cp.predict()
    pred = model.predict(X_test)
    plot_pred(x=X_test, y=y_test, y_u=y_upper, y_l=y_lower, pred=pred, shade_color=config.UtilsParams.cqr_color, method_name=method_name + ":", title="",
                filename=os.path.join('./results', method_name + '_' + dataset_name), save_figures=config.UtilsParams.save_figures)
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    print(method_name + " Random Forests: Percentage in the range (expecting " + str(100 * (1 - config.ConformalParams.alpha)) + "%):", in_the_range / len(y_test) * 100)
    length_cqr_rf = y_upper - y_lower
    print(method_name + " Random Forests: Average length:", np.mean(length_cqr_rf))


def model_bias_study(gamma, random_seed):
    
    set_seed(random_seed)
    X_train, X_test, y_train, y_test = datasets.GetDataset('simulation', base_dataset_path)
    mean_estimator = RandomForestRegressor(n_estimators=config.RandomForecastParams.n_estimators, 
                                            min_samples_leaf=config.RandomForecastParams.min_samples_leaf,
                                            max_features=config.RandomForecastParams.max_features, 
                                            random_state=config.RandomForecastParams.random_state)
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None, fit_params=None, quantiles=config.ConformalParams.quantiles, params=config.RandomForecastParams)
    cp = ConformalPred(model=quantile_estimator, method='cqr', ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, model_2=mean_estimator, gamma=gamma)
    cp_local = ConformalPred(model=quantile_estimator, method='ldcp', ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, model_2=mean_estimator, gamma=gamma, k=100)
    cp.fit()
    cp_local.fit()
    y_lower, y_upper = cp.predict()
    y_lower_local, y_upper_local = cp_local.predict()
    in_range_local = np.sum((y_test >= y_lower_local) & (y_test <= y_upper_local)) / len(y_test) * 100
    in_range = np.sum((y_test >= y_lower) & (y_test <= y_upper)) / len(y_test) * 100

    length = np.mean(y_upper - y_lower)
    length_local = np.mean(y_upper_local - y_lower_local)
    return length, length_local, in_range, in_range_local


def cov_shift_study(a, b, random_seed, k=100):
    
    set_seed(random_seed)
    X_train, X_test, y_train, y_test = datasets.GetDataset('cov_shift', base_dataset_path, a, b)
    quantile_estimator = helper.Linear_RegressorAdapter(model=None)
    quantile_estimator_local = helper.Linear_RegressorAdapter(model=None)
    cp = ConformalPred(model=quantile_estimator, method='cqr', ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test)
    cp_local = ConformalPred(model=quantile_estimator_local, method='ldcp', ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, k=k)
    cp.fit()
    cp_local.fit()
    y_lower, y_upper = cp.predict()
    y_lower_local, y_upper_local = cp_local.predict()
    coverage_local, length_local = helper.compute_coverage(y_test, y_lower_local, y_upper_local, config.ConformalParams.alpha, 'LDCP' + " Linear Regr")
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_test, y_lower, y_upper, config.ConformalParams.alpha, 'CQR' + " Linear Regr")
    return coverage_local, coverage_cp_qnet


def run_cov_shift(k=100):
    a_list = np.linspace(0, 1, 11)
    b_list = np.linspace(0, 1, 11)
    coverage_local_all, coverage_cp_all = [], []
    for a, b in zip(a_list, b_list):
        if a == 0.:
            a += .01
            b += .01
        a, b = np.round(a, 2), np.round(b, 2)
        coverage_local, coverage_cp = cov_shift_study(a, b, config.UtilsParams.seed)
        coverage_local_all.append(coverage_local)
        coverage_cp_all.append(coverage_cp)
    plot_cov_shift(coverage_cp_all, coverage_local_all, a_list, k, config.DataParams.n_train)


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
