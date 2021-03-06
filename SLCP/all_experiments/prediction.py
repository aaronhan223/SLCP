import os
from age_regression import AgeModel
import config
import numpy as np
import pandas as pd
from datasets import datasets
from conformal import ConformalPred
from utils import plot_pred, set_seed
from cqr import helper
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from collections import OrderedDict
import logging
import torch
import torch.nn as nn
import copy
import sys
logger = logging.getLogger('SLCP.prediction')
base_dataset_path = './datasets/'
pd.set_option('precision', 3)
import pdb


def run_pred_experiment(dataset_name, model_name, method_name, random_seed, conformal):

    set_seed(random_seed)
    try:
        X_train, X_test, y_train, y_test = datasets.GetDataset(dataset_name, 
                                                                base_dataset_path, 
                                                                random_seed, 
                                                                config.DataParams.test_ratio)
    except:
        logger.info("CANNOT LOAD DATASET!")
        return

    X_train, X_test = X_train.astype(np.float32), X_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.float32), y_test.astype(np.float32)
    in_shape = X_train.shape[1]

    if 'simulation' in dataset_name:
        n_train = config.DataParams.n_train
    else:
        n_train = X_train.shape[0]
    idx = np.random.permutation(n_train)
    n_half = int(np.floor(n_train * config.ConformalParams.valid_ratio))
    idx_train, idx_cal = idx[:n_half], idx[n_half:]

    # zero mean and unit variance scaling of the train and test features
    scalerX = StandardScaler()
    scalerX = scalerX.fit(X_train[idx_train])
    X_train = scalerX.transform(X_train)
    X_test = scalerX.transform(X_test)

    # scale the labels by dividing each by the mean absolute response
    mean_ytrain = np.mean(np.abs(y_train[idx_train]))
    y_train = np.squeeze(y_train)/mean_ytrain
    y_test = np.squeeze(y_test)/mean_ytrain

    ############
    save_path = '/home/xhan/SLCP/SLCP/datasets/saved_data'
    x_train, x_cal = X_train[idx_train], X_train[idx_cal]
    y_train, y_cal = y_train[idx_train], y_train[idx_cal]
    if len(y_train.shape) == 1:
        y_train = np.expand_dims(y_train, 1)
        y_cal = np.expand_dims(y_cal, 1)
        y_test = np.expand_dims(y_test, 1)
    np.savez(os.path.join(save_path, dataset_name + '.npz'), x_train=x_train, x_cal=x_cal, x_test=X_test, y_train=y_train, y_cal=y_cal, y_test=y_test)
    # sys.exit()

    if model_name == 'random_forest':
        if conformal and method_name in ['split', 'lacp', 'slcp-mean']:
            model = RandomForestRegressor(n_estimators=config.RandomForecastParams.n_estimators, 
                                          min_samples_leaf=config.RandomForecastParams.min_samples_leaf,
                                          max_features=config.RandomForecastParams.max_features, 
                                          random_state=config.RandomForecastParams.random_state)
        else:
            model = helper.QuantileForestRegressorAdapter(model=None, 
                                                          fit_params=None, 
                                                          quantiles=config.ConformalParams.quantiles, 
                                                          params=config.RandomForecastParams)

    elif model_name == 'linear':
        if conformal and method_name in ['split', 'lacp', 'slcp-mean']:
            model = helper.MSELR_RegressorAdapter(model=None, 
                                                  in_shape=in_shape,
                                                  epochs=config.LinearParams.epochs,
                                                  lr=config.LinearParams.lr,
                                                  batch_size=config.LinearParams.batch_size,
                                                  wd=config.LinearParams.wd,
                                                  test_ratio=config.LinearParams.test_ratio,
                                                  random_state=config.LinearParams.random_state)
        else:
            model = helper.QLR_RegressorAdapter(model=None, 
                                                in_shape=in_shape,
                                                epochs=config.LinearParams.epochs,
                                                lr=config.LinearParams.lr,
                                                batch_size=config.LinearParams.batch_size,
                                                wd=config.LinearParams.wd,
                                                test_ratio=config.LinearParams.test_ratio,
                                                random_state=config.LinearParams.random_state)

    elif model_name == 'neural_net':
        if conformal and method_name in ['split', 'lacp', 'slcp-mean']:
            model = helper.MSENet_RegressorAdapter(model=None, 
                                                   in_shape=in_shape,
                                                   hidden_size=config.NeuralNetParams.hidden_size,
                                                   epochs=config.NeuralNetParams.epochs,
                                                   lr=config.NeuralNetParams.lr,
                                                   batch_size=config.NeuralNetParams.batch_size,
                                                   dropout=config.NeuralNetParams.dropout,
                                                   wd=config.NeuralNetParams.wd,
                                                   test_ratio=config.NeuralNetParams.test_ratio,
                                                   random_state=config.NeuralNetParams.random_state)
        else:
            model = helper.AllQNet_RegressorAdapter(model=None, 
                                                    in_shape=in_shape,
                                                    hidden_size=config.NeuralNetParams.hidden_size,
                                                    epochs=config.NeuralNetParams.epochs,
                                                    lr=config.NeuralNetParams.lr,
                                                    batch_size=config.NeuralNetParams.batch_size,
                                                    dropout=config.NeuralNetParams.dropout,
                                                    wd=config.NeuralNetParams.wd,
                                                    test_ratio=config.NeuralNetParams.test_ratio,
                                                    random_state=config.NeuralNetParams.random_state)

    elif model_name == 'kde':
        if conformal and method_name in ['split', 'lacp', 'slcp-mean']:
            model = helper.MSEConst_RegressorAdapter()
        else:
            model = helper.QConst_RegressorAdapter()

    if conformal:
        cp = ConformalPred(model=model, 
                            method=method_name, 
                            data_name=dataset_name, 
                            ratio=config.ConformalParams.valid_ratio, 
                            x_train=X_train, 
                            x_test=X_test, 
                            y_train=y_train, 
                            y_test=y_test, 
                            k=config.ConformalParams.k)
        cp.fit()
        y_lower, y_upper = cp.predict()
    else:
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        y_lower, y_upper = predictions[:, 0], predictions[:, 1]

    # name_map = {'slcp-rbf': 'SLCP', 'cqr': 'CQR', 'split': 'Split Conformal'}
    color_map = {'slcp-knn': 'gray', 'slcp-rbf': 'gray', 'slcp-mean': 'gray', 'cqr': 'lightblue', 'cqr-asy': 'lightblue', 'split': 'tomato', 'lacp': 'tomato', 'qr': 'tomato'}
    model_map = {'random_forest': 'Random Forest', 'linear': 'Linear Regression', 'neural_net': 'Neural Network', 'kde': 'Constant'}
    estimator_map = {'slcp-knn': 'quantile', 'slcp-rbf': 'quantile', 'slcp-mean': 'mean', 'cqr': 'quantile', 'cqr-asy': 'quantile', 'split': 'mean', 'lacp': 'mean', 'qr': 'quantile'}
    pred = model.predict(X_test)
    if 'simulation' in dataset_name:
        plot_pred(x=X_test, 
                  y=y_test, 
                  y_u=y_upper, 
                  y_l=y_lower, 
                  pred=pred, 
                  shade_color=color_map[method_name], 
                  method_name=method_name + ":", 
                  title=f"{method_name} {model_map[model_name]} ({estimator_map[method_name]} regression)",
                  filename=os.path.join('./results', method_name + '_' + model_name + '_' + dataset_name + '.pdf'), 
                  save_figures=config.UtilsParams.save_figures)

    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    logger.info(f'{method_name} {model_name} : Coverage rate (expecting {100 * (1 - config.ConformalParams.alpha)} %): {round(in_the_range / len(y_test) * 100, 2)}')
    interval_length = y_upper - y_lower
    logger.info(f'{method_name} {model_name} : Average length: {round(np.mean(interval_length), 2)}')
    return round(in_the_range / len(y_test) * 100, 2), round(np.mean(interval_length), 2)


def run_age_pred(dataset_name, model_name, method_name, random_seed):
    set_seed(random_seed)
    path_to_model = './saved_models/best_model.pt'
    checkpoint = torch.load(path_to_model)
    embd_model = AgeModel(embedding=True)
    model = AgeModel()
    head_model = AgeModelHead()
    model.load_state_dict(checkpoint['model_state_dict'])
    model_dict = model.state_dict()
    embd_dict_keys = embd_model.state_dict().keys()
    embd_model_dict, head_model_dict = OrderedDict(), OrderedDict()
    head_model_dict['agemodelhead.1.weight'] = model_dict['agemodel.18.weight']
    head_model_dict['agemodelhead.1.bias'] = model_dict['agemodel.18.bias']
    for key in embd_dict_keys:
        embd_model_dict[key] = model_dict[key]
    embd_model.load_state_dict(embd_model_dict)
    head_model.load_state_dict(head_model_dict)
    embd_model.cuda()
    head_model.cuda()
    cp = ConformalPred(model=head_model, method=method_name, data_name=dataset_name, ratio=config.ConformalParams.valid_ratio, embd_model=embd_model)
    cp.fit()
    y_lower, y_upper, y_test = cp.predict()
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    logger.info(f'{method_name} {model_name} : Coverage rate (expecting {100 * (1 - config.ConformalParams.alpha)} %): {round(in_the_range / len(y_test) * 100, 2)}')
    interval_length = y_upper - y_lower
    logger.info(f'{method_name} {model_name} : Average length: {round(np.mean(interval_length), 2)}')
    return round(in_the_range / len(y_test) * 100, 2), round(np.mean(interval_length), 2)


class AgeModelHead(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.ReLU(inplace=True)]
        layers += [nn.Linear(16,1)]
        self.agemodelhead = nn.Sequential(*layers)

    def forward(self, x):
        return self.agemodelhead(x).squeeze(-1)