from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.cp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from utils import ImageLoader
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.ensemble import RandomForestRegressor
import config
import torch
import copy
import numpy as np
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import time
import logging
import pdb
logger = logging.getLogger('SLCP.conformal')


class ConformalPred:
    """ Wrapper of all conformal prediciton method

    Parameters
    ----------
    ratio: float, proportion of training data used to calibrate
    """
    def __init__(self, model, method, data_name, ratio, embd_model=None, x_train=None, x_test=None, y_train=None, y_test=None, model_2=None, gamma=1., k=300) -> None:
        
        self.image_data = False

        if x_train is None:
            self.image_data = True
            ImageDataset_train = ImageLoader(annotations_file='./datasets/img_labels_train.csv',
                                            img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                            transform=torch.nn.Sequential(
                                                                transforms.RandomRotation(degrees=10),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomApply(
                                                                    torch.nn.ModuleList([
                                                                        transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                                        ]), p=0.7),
                                                                transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                                transforms.Resize(128),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                )
                                                            )
            ImageDataset_valid = ImageLoader(annotations_file='./datasets/img_labels_valid.csv',
                                            img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                            transform=torch.nn.Sequential(
                                                                transforms.RandomRotation(degrees=10),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomApply(
                                                                    torch.nn.ModuleList([
                                                                        transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                                        ]), p=0.7),
                                                                transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                                transforms.Resize(128),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                )
                                                            )
            ImageDataset_test = ImageLoader(annotations_file='./datasets/img_labels_test.csv',
                                            img_dir='/mnt/d64c1162-08cc-4571-90a3-04c60b6f6f66/xing/wiki_utk',
                                            transform=torch.nn.Sequential(
                                                                transforms.RandomRotation(degrees=10),
                                                                transforms.RandomHorizontalFlip(),
                                                                transforms.RandomApply(
                                                                    torch.nn.ModuleList([
                                                                        transforms.ColorJitter(brightness=(0.3, 0.60), contrast=(0.5, 2)),
                                                                        ]), p=0.7),
                                                                transforms.RandomCrop(size=600, pad_if_needed=True, padding_mode='edge'),
                                                                transforms.Resize(128),
                                                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                                )
                                                            )
            bs = 32
            train_loader = DataLoader(ImageDataset_train, batch_size=bs, num_workers=0, shuffle=False)
            valid_loader = DataLoader(ImageDataset_valid, batch_size=bs, num_workers=0, shuffle=False)
            test_loader = DataLoader(ImageDataset_test, batch_size=bs, num_workers=0, shuffle=False)
            model = model.cuda()
            model.eval()
            embd_model.eval()
            self.img_train = torch.zeros((len(ImageDataset_train), 16))
            self.lbl_train = torch.zeros(len(ImageDataset_train))
            self.img_valid = torch.zeros((len(ImageDataset_valid), 16))
            self.lbl_valid = torch.zeros(len(ImageDataset_valid))
            self.img_test = torch.zeros((len(ImageDataset_test), 16))
            self.lbl_test = torch.zeros(len(ImageDataset_test))
            with torch.no_grad():
                for i, batch_train in enumerate(train_loader):
                    inputs_train = torch.Tensor(batch_train['image'].float())
                    results_train = torch.Tensor(batch_train['label'].float())
                    inputs_train, results_train = inputs_train.cuda(), results_train.cuda()
                    self.img_train[i * bs: i * bs + len(results_train)] = embd_model(inputs_train)
                    self.lbl_train[i * bs: i * bs + len(results_train)] = results_train

                for i, batch_valid in enumerate(valid_loader):
                    inputs_valid = torch.Tensor(batch_valid['image'].float())
                    results_valid = torch.Tensor(batch_valid['label'].float())
                    inputs_valid, results_valid = inputs_valid.cuda(), results_valid.cuda()
                    self.img_valid[i * bs: i * bs + len(results_valid)] = embd_model(inputs_valid)
                    self.lbl_valid[i * bs: i * bs + len(results_valid)] = results_valid

                for i, batch_test in enumerate(test_loader):
                    inputs_test = torch.Tensor(batch_test['image'].float())
                    results_test = torch.Tensor(batch_test['label'].float())
                    inputs_test, results_test = inputs_test.cuda(), results_test.cuda()
                    self.img_test[i * bs: i * bs + len(results_test)] = embd_model(inputs_test)
                    self.lbl_test[i * bs: i * bs + len(results_test)] = results_test
        else:
            self.x_train = x_train
            self.x_test = x_test
            self.y_train = y_train
            self.y_test = y_test

        if method == 'slcp-knn':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), image=self.image_data, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'slcp-rbf':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), image=self.image_data, mean=False, rbf_kernel=True, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'slcp-mean':
            local = True
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), image=self.image_data, rbf_kernel=True, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'cqr':
            local = False
            nc = RegressorNc(model, local, k, err_func=QuantileRegErrFunc(), image=self.image_data, alpha=config.ConformalParams.alpha, model_2=model_2, gamma=gamma)
        elif method == 'cqr-asy':
            local = False
            nc = RegressorNc(model, local, k, err_func=QuantileRegAsymmetricErrFunc(), image=self.image_data, alpha=config.ConformalParams.alpha)
        elif method == 'lacp':
            local = False
            normalizer_adapter = copy.deepcopy(model)
            normalizer = RegressorNormalizer(model, normalizer_adapter, AbsErrorErrFunc())
            nc = RegressorNc(model, local, k, err_func=AbsErrorErrFunc(), alpha=config.ConformalParams.alpha, normalizer=normalizer, beta=1, image=self.image_data)
        else:
            local = False
            nc = RegressorNc(model, local, k, err_func=AbsErrorErrFunc(), alpha=config.ConformalParams.alpha)
        self.icp = IcpRegressor(nc, local, k, significance=config.ConformalParams.alpha)
        
        if not self.image_data:
            if 'simulation' in data_name:
                n_train = config.DataParams.n_train
            else:
                n_train = self.x_train.shape[0]
            idx = np.random.permutation(n_train)
            n_half = int(np.floor(n_train * (1 - ratio)))
            self.idx_train, self.idx_cal = idx[:n_half], idx[n_half:]

    def fit(self):
        if not self.image_data:
            self.icp.fit(self.x_train[self.idx_train], self.y_train[self.idx_train])
            start = time.time()
            self.icp.calibrate(self.x_train[self.idx_cal], self.y_train[self.idx_cal])
            time_length = time.time() - start
            logger.info(f"Calibration time: {np.round(time_length, 4)}")
        else:
            self.lbl_train = self.lbl_train.numpy()
            self.img_valid = self.img_valid.numpy()
            self.lbl_valid = self.lbl_valid.numpy()
            self.icp.fit(self.img_train, self.lbl_train)
            start = time.time()
            self.icp.calibrate(self.img_valid, self.lbl_valid)
            time_length = time.time() - start
            logger.info(f"Calibration time: {np.round(time_length, 4)}")

    def predict(self):
        if not self.image_data:
            predictions = self.icp.predict(self.x_test, significance=config.ConformalParams.alpha)
        else:
            predictions = self.icp.predict(self.img_test.numpy(), significance=config.ConformalParams.alpha)
        y_lower = predictions[:, 0]
        y_upper = predictions[:, 1]
        if not self.image_data:
            return y_lower, y_upper
        else:
            return y_lower, y_upper, self.lbl_test.numpy()
