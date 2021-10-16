import os
import sys
import config
import numpy as np
import pandas as pd
from datasets import datasets
from conformal import ConformalPred
from utils import plot_pred, plot_toy_ave_length, plot_toy_cov_rate
from cqr import helper
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import logging
import copy
import pdb


pd.set_option('precision', 3)
base_dataset_path = './datasets/'
logger = logging.getLogger('SLCP.experiment')


def plot_toy_example():

    # X_train, X_test, y_train, y_test = datasets.GetDataset('simulation_2', base_dataset_path, config.UtilsParams.seed, config.DataParams.test_ratio)
    X_train = np.random.uniform(0, 5.0, size=config.DataParams.n_train).astype(np.float32)
    X_test = np.random.uniform(0, 5.0, size=config.DataParams.n_test).astype(np.float32)
    sim = datasets.simulation(2)
    y_train = sim.generate(X_train)
    y_test, noise = sim.generate(X_test, gt=True)
    y_test_upper = y_test + 0.15 * np.quantile(noise, 0.95) * X_test
    y_test_lower = y_test + 0.15 * np.quantile(noise, 0.05) * X_test
    y_test += 0.15 * noise * X_test

    X_train = np.reshape(X_train, (config.DataParams.n_train, 1))
    X_test = np.reshape(X_test, (config.DataParams.n_test, 1))

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    in_shape = X_train.shape[1]

    n_train = config.DataParams.n_train
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * config.ConformalParams.valid_ratio))
    idx_train = idx[:n_half]

    # TODO: don't need to scale here?
    mean_model = helper.MSENet_RegressorAdapter(model=None, 
                                                in_shape=in_shape,
                                                hidden_size=config.NeuralNetParams.hidden_size,
                                                epochs=config.NeuralNetParams.epochs,
                                                lr=config.NeuralNetParams.lr,
                                                batch_size=config.NeuralNetParams.batch_size,
                                                dropout=config.NeuralNetParams.dropout,
                                                wd=config.NeuralNetParams.wd,
                                                test_ratio=config.NeuralNetParams.test_ratio,
                                                random_state=config.NeuralNetParams.random_state)

    quantile_model = helper.AllQNet_RegressorAdapter(model=None, 
                                                     in_shape=in_shape,
                                                     hidden_size=config.NeuralNetParams.hidden_size,
                                                     epochs=config.NeuralNetParams.epochs,
                                                     lr=config.NeuralNetParams.lr,
                                                     batch_size=config.NeuralNetParams.batch_size,
                                                     dropout=config.NeuralNetParams.dropout,
                                                     wd=config.NeuralNetParams.wd,
                                                     test_ratio=config.NeuralNetParams.test_ratio,
                                                     random_state=config.NeuralNetParams.random_state)
    split = ConformalPred(model=mean_model, 
                          method='split', 
                          data_name='simulation_2', 
                          ratio=config.ConformalParams.valid_ratio, 
                          x_train=X_train, 
                          x_test=X_test, 
                          y_train=y_train, 
                          y_test=y_test, 
                          k=config.ConformalParams.k)

    slcp = ConformalPred(model=quantile_model, 
                          method='slcp', 
                          data_name='simulation_2', 
                          ratio=config.ConformalParams.valid_ratio, 
                          x_train=X_train, 
                          x_test=X_test, 
                          y_train=y_train, 
                          y_test=y_test, 
                          k=config.ConformalParams.k)
    
    split.fit()
    slcp.fit()

    y_lower_split, y_upper_split = split.predict()
    y_lower_slcp, y_upper_slcp = slcp.predict()

    plot_pred(
        x=X_test,
        y=y_test,
        y_u=y_upper_slcp,
        y_l=y_lower_slcp,
        y_u_2=y_upper_split,
        y_l_2=y_lower_split,
        gt_u=y_test_upper,
        gt_l=y_test_lower,
        shade_color=config.UtilsParams.cqr_color, 
        method_name='slcp' + ":", 
        title="Conformal Prediction Intervals",
        filename='./results/slcp_toy_plot.pdf', 
        save_figures=config.UtilsParams.save_figures
    )

    plot_toy_ave_length(
        x=X_test,
        y_u=y_upper_slcp,
        y_l=y_lower_slcp,
        y_u_2=y_upper_split,
        y_l_2=y_lower_split,
        gt_u=y_test_upper,
        gt_l=y_test_lower,
        filename='./results/slcp_toy_length.pdf'
    )

    plot_toy_cov_rate(
        x=X_test,
        y=y_test,
        y_u=y_upper_slcp,
        y_l=y_lower_slcp,
        y_u_2=y_upper_split,
        y_l_2=y_lower_split,
        gt_u=y_test_upper,
        gt_l=y_test_lower,
        filename='./results/slcp_toy_cov.pdf'
    )