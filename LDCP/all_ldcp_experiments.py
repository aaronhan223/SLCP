import os
import sys
import config
import numpy as np
import pandas as pd
import torch
import random
from datasets import datasets
from conformal import ConformalPred
from utils import plot_func
from cqr import helper
import pdb


pd.set_option('precision', 3)
base_dataset_path = './datasets/'


def run_pred_experiment(dataset_name, model_name, method_name, random_seed):

    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    try:
        X_train, X_test, y_train, y_test = datasets.GetDataset(dataset_name, base_dataset_path)
    except:
        print("CANNOT LOAD DATASET!")
        return

    if model_name == 'random_forest':
        model = helper.QuantileForestRegressorAdapter(model=None, fit_params=None, quantiles=config.ConformalParams.quantiles, params=config.RandomForecastParams)
        cp = ConformalPred(model=model, method=method_name, ratio=0.5, x_train=X_train, x_test=X_test, y_train=y_train, y_test=y_test, k=300)
        cp.fit()
        y_lower, y_upper = cp.predict()
        pred = model.predict(X_test)
        plot_func(x=X_test, y=y_test, y_u=y_upper, y_l=y_lower, pred=pred, shade_color=config.UtilsParams.cqr_color, method_name=method_name + ":", title="",
                  filename=os.path.join('./results', method_name + '_' + dataset_name), save_figures=config.UtilsParams.save_figures)
        in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
        print(method_name + " Random Forests: Percentage in the range (expecting " + str(100 * (1 - config.ConformalParams.alpha)) + "%):", in_the_range / len(y_test) * 100)
        length_cqr_rf = y_upper - y_lower
        print(method_name + " Random Forests: Average length:", np.mean(length_cqr_rf))


if __name__ == '__main__':
    run_pred_experiment(dataset_name='simulation_3', model_name='random_forest', method_name='cqr', random_seed=config.UtilsParams.seed)