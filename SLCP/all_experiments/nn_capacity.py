import os
import sys
import config
import numpy as np
import pandas as pd
from datasets import datasets
from conformal import ConformalPred
from utils import set_seed, plot_nn_capacity
from cqr import helper
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import logging
import copy
import pdb

pd.set_option('precision', 3)
base_dataset_path = './datasets/'
logger = logging.getLogger('SLCP.nn_capacity')


def model_capacity_study(hidden_size, dataset_name, random_seed):
    
    set_seed(random_seed)
    try:
        X_train, X_test, y_train, y_test = datasets.GetDataset(dataset_name, base_dataset_path, random_seed, config.DataParams.test_ratio)
    except:
        logger.info("CANNOT LOAD DATASET!")
        return

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    in_shape = X_train.shape[1]
    n_train = X_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * config.ConformalParams.valid_ratio))
    idx_train = idx[:n_half]

    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)

    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train) / mean_ytrain
    y_test = np.squeeze(y_test) / mean_ytrain

    mean_model = helper.MSENet_RegressorAdapter(model=None, 
                                                in_shape=in_shape,
                                                hidden_size=hidden_size,
                                                epochs=config.NeuralNetParams.epochs,
                                                lr=config.NeuralNetParams.lr,
                                                batch_size=config.NeuralNetParams.batch_size,
                                                dropout=config.NeuralNetParams.dropout,
                                                wd=config.NeuralNetParams.wd,
                                                test_ratio=config.NeuralNetParams.test_ratio,
                                                random_state=config.NeuralNetParams.random_state)

    quantile_model = helper.AllQNet_RegressorAdapter(model=None, 
                                                     in_shape=in_shape,
                                                     hidden_size=hidden_size,
                                                     epochs=config.NeuralNetParams.epochs,
                                                     lr=config.NeuralNetParams.lr,
                                                     batch_size=config.NeuralNetParams.batch_size,
                                                     dropout=config.NeuralNetParams.dropout,
                                                     wd=config.NeuralNetParams.wd,
                                                     test_ratio=config.NeuralNetParams.test_ratio,
                                                     random_state=config.NeuralNetParams.random_state)

    cqr_model = copy.deepcopy(quantile_model)

    split = ConformalPred(model=mean_model, 
                          method='split', 
                          data_name=dataset_name, 
                          ratio=config.ConformalParams.valid_ratio, 
                          x_train=X_train, 
                          x_test=X_test, 
                          y_train=y_train, 
                          y_test=y_test)

    cqr = ConformalPred(model=cqr_model, 
                        method='cqr', 
                        data_name=dataset_name, 
                        ratio=config.ConformalParams.valid_ratio, 
                        x_train=X_train, 
                        x_test=X_test, 
                        y_train=y_train, 
                        y_test=y_test)

    slcp = ConformalPred(model=quantile_model, 
                         method='slcp-rbf', 
                         data_name=dataset_name, 
                         ratio=config.ConformalParams.valid_ratio, 
                         x_train=X_train, 
                         x_test=X_test, 
                         y_train=y_train, 
                         y_test=y_test, 
                         k=config.ConformalParams.k)
    
    split.fit()
    cqr.fit()
    slcp.fit()
    
    y_lower_split, y_upper_split = split.predict()
    y_lower_slcp, y_upper_slcp = slcp.predict()
    y_lower_cqr, y_upper_cqr = cqr.predict()
    
    in_the_range_split = np.sum((y_test >= y_lower_split) & (y_test <= y_upper_split))
    in_the_range_cqr = np.sum((y_test >= y_lower_cqr) & (y_test <= y_upper_cqr))
    in_the_range_slcp = np.sum((y_test >= y_lower_slcp) & (y_test <= y_upper_slcp))
    
    cov_rate_split = round(in_the_range_split / len(y_test) * 100, 2)
    cov_rate_cqr = round(in_the_range_cqr / len(y_test) * 100, 2)
    cov_rate_slcp = round(in_the_range_slcp / len(y_test) * 100, 2)

    length_split = y_upper_split - y_lower_split
    length_cqr = y_upper_cqr - y_lower_cqr
    length_slcp = y_upper_slcp - y_lower_slcp

    ave_length_split = round(np.mean(length_split), 2)
    ave_length_cqr = round(np.mean(length_cqr), 2)
    ave_length_slcp = round(np.mean(length_slcp), 2)

    cov_rate_results = {'split': cov_rate_split, 'cqr': cov_rate_cqr, 'slcp': cov_rate_slcp}
    ave_length_results = {'split': ave_length_split, 'cqr': ave_length_cqr, 'slcp': ave_length_slcp}

    return cov_rate_results, ave_length_results


def run_nn_capacity(random_seed, dataset_name):

    hidden_size = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    split_length, split_cov_rate = [], []
    cqr_length, cqr_cov_rate = [], []
    slcp_length, slcp_cov_rate = [], []

    for size in tqdm(hidden_size):
        cov_rate_results, ave_length_results = model_capacity_study(size, dataset_name, random_seed)
        split_length.append(ave_length_results['split'])
        split_cov_rate.append(cov_rate_results['split'])
        cqr_length.append(ave_length_results['cqr'])
        cqr_cov_rate.append(cov_rate_results['cqr'])
        slcp_length.append(ave_length_results['slcp'])
        slcp_cov_rate.append(cov_rate_results['slcp'])

    all_length = {'split': split_length, 'cqr': cqr_length, 'slcp': slcp_length}
    all_cov_rate = {'split': split_cov_rate, 'cqr': cqr_cov_rate, 'slcp': slcp_cov_rate}
    plot_nn_capacity(all_length, all_cov_rate, hidden_size, dataset_name)