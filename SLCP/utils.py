import matplotlib.pyplot as plt
import numpy as np
import random
import config
import torch
import logging
import pandas as pd
import seaborn as sns 
import config
import os
from torchvision.io import read_image
from torch.utils.data import Dataset
import pdb


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_logger(log_path):
    '''Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `save/history.log`.
    Example:
    logging.info('Starting training...')
    Args:
        log_path: (string) where to log
    '''
    _logger = logging.getLogger('SLCP')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)


class ImageLoader(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, index_col=0)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        print(image.shape)
        print(label)
        print(img_path)
        print('idx', idx)
        print('-------')
        return image, label


def plot_pred(x, y, y_u=None, y_l=None, pred=None, y_u_2=None, y_l_2=None, gt_u=None, gt_l=None, shade_color="", method_name="", title="", filename=None, save_figures=True):
    
    """ Scatter plot of (x,y) points along with the constructed prediction interval 
    
    Parameters
    ----------
    x : numpy array, corresponding to the feature of each of the n samples
    y : numpy array, target response variable (length n)
    pred : numpy array, the estimated prediction. It may be the conditional mean,
           or low and high conditional quantiles.
    shade_color : string, desired color of the prediciton interval
    method_name : string, name of the method
    title : string, the title of the figure
    filename : sting, name of the file to save the figure
    save_figures : boolean, save the figure (True) or not (False)
    
    """
    
    x_ = x[:config.DataParams.max_show]
    y_ = y[:config.DataParams.max_show]
    if y_u is not None:
        y_u_ = y_u[:config.DataParams.max_show]
    if y_l is not None:
        y_l_ = y_l[:config.DataParams.max_show]
    if y_u_2 is not None:
        y_u_2_ = y_u_2[:config.DataParams.max_show]
    if y_l_2 is not None:
        y_l_2_ = y_l_2[:config.DataParams.max_show]
    if gt_u is not None:
        gt_u_ = gt_u[:config.DataParams.max_show]
    if gt_l is not None:
        gt_l_ = gt_l[:config.DataParams.max_show]  
    if pred is not None:
        pred_ = pred[:config.DataParams.max_show]

    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds,:], y_[inds], 'k.', alpha=.2, markersize=10, fillstyle='none', label=u'Observations')
    
    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x_[inds], x_[inds][::-1]]),
                 np.concatenate([y_u_[inds], y_l_[inds][::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label='SLCP' + ' prediction interval')
    
    if (y_u_2 is not None) and (y_l_2 is not None):
        plt.plot(x_[inds,:], y_u_2_[inds], '#F75C2F', lw=2, alpha=0.9, label=u'Split Conformal Interval')
        plt.plot(x_[inds,:], y_l_2_[inds], '#F75C2F', lw=2, alpha=0.9)
        plt.plot(x_[inds,:], gt_u_[inds], '#1B813E', ls='dashed', lw=2, alpha=0.9, label=u'Ground Truth Interval')
        plt.plot(x_[inds,:], gt_l_[inds], '#1B813E', ls='dashed', lw=2, alpha=0.9)

    if pred is not None:
        if pred_.ndim == 2:
            plt.plot(x_[inds,:], pred_[inds,0], 'k', lw=2, alpha=0.9, label=u'Predicted low and high quantiles')
            plt.plot(x_[inds,:], pred_[inds,1], 'k', lw=2, alpha=0.9)
        else:
            plt.plot(x_[inds,:], pred_[inds], 'k--', lw=2, alpha=0.9, label=u'Predicted value')
    
    plt.ylim([-2.5, 7])
    plt.xlabel('$X$', fontsize=22)
    plt.ylabel('$Y$', fontsize=22)
    plt.legend(loc='best', fontsize=14)
    plt.title(title, fontsize=20)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_toy_cov_rate(x, y, y_u, y_l, y_u_2, y_l_2, gt_u, gt_l, filename):
    x_ = x[:config.DataParams.max_show]
    y_ = y[:config.DataParams.max_show]
    y_u_ = y_u[:config.DataParams.max_show]
    y_l_ = y_l[:config.DataParams.max_show]
    y_u_2_ = y_u_2[:config.DataParams.max_show]
    y_l_2_ = y_l_2[:config.DataParams.max_show]
    gt_u_ = gt_u[:config.DataParams.max_show]
    gt_l_ = gt_l[:config.DataParams.max_show]

    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    x_ = x_[inds,:]
    y_ = y_[inds]
    length = len(inds)

    y_upper_split = y_u_2_[inds]
    y_lower_split = y_l_2_[inds]
    y_upper_slcp = y_u_[inds]
    y_lower_slcp = y_l_[inds]
    y_upper_gt = gt_u_[inds]
    y_lower_gt = gt_l_[inds]

    cov_diff_split = np.absolute(y_upper_gt - y_upper_split) + np.absolute(y_lower_gt - y_lower_split)
    cov_diff_slcp = np.absolute(y_upper_gt - y_upper_slcp) + np.absolute(y_lower_gt - y_lower_slcp)

    plt.plot(x_, cov_diff_split, '#F75C2F', lw=2, alpha=0.9, label=u'Split Conformal')
    plt.plot(x_, cov_diff_slcp, '#2EA9DF', lw=2, alpha=0.9, label=u'SLCP')
    plt.xlabel('$X$', fontsize=22)
    plt.ylabel('$\Delta C$', fontsize=22)
    plt.legend(loc='best', fontsize=18)
    plt.title('Local Difference of Conformal Band', fontsize=20)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_toy_ave_length(x, y_u, y_l, y_u_2, y_l_2, gt_u, gt_l, filename):
    x_ = x[:config.DataParams.max_show]
    y_u_ = y_u[:config.DataParams.max_show]
    y_l_ = y_l[:config.DataParams.max_show]
    y_u_2_ = y_u_2[:config.DataParams.max_show]
    y_l_2_ = y_l_2[:config.DataParams.max_show]
    gt_u_ = gt_u[:config.DataParams.max_show]
    gt_l_ = gt_l[:config.DataParams.max_show]
    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))

    y_upper_split = y_u_2_[inds]
    y_lower_split = y_l_2_[inds]
    y_upper_slcp = y_u_[inds]
    y_lower_slcp = y_l_[inds]
    y_upper_gt = gt_u_[inds]
    y_lower_gt = gt_l_[inds]

    interval_length_split = y_upper_split - y_lower_split
    interval_length_slcp = y_upper_slcp - y_lower_slcp
    interval_length_gt = y_upper_gt - y_lower_gt
    plt.plot(x_[inds,:], interval_length_split, '#F75C2F', lw=2, alpha=0.9, label=u'Split Conformal')
    plt.plot(x_[inds,:], interval_length_gt, '#1B813E', ls='dotted', lw=2, alpha=0.9, label=u'Ground Truth')
    plt.plot(x_[inds,:], interval_length_slcp, '#2EA9DF', lw=2, alpha=0.9, label=u'SLCP')
    plt.xlabel('$X$', fontsize=22)
    plt.ylabel('Average Length', fontsize=22)
    plt.legend(loc='best', fontsize=18)
    plt.title('Local Length of Conformal Band', fontsize=22)
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_model_bias(length_vals, length_local_vals, gamma_vals, k, n_sample):
    keys = ['c', 'd']
    colors = {'c': '#D05A6E', 'd': '#3A8FB7'}
    legends = {'c': 'CQR', 'd': 'SLCP'}
    markers = {'c': 'o', 'd': 'o'}
    linestyles = {'c': 'solid', 'd': 'solid'}
    plt.rcParams["figure.figsize"] = (10, 8.5)

    data = {'c': length_vals, 'd': length_local_vals}
    for key in keys:
        plt.plot(gamma_vals, data[key], label=legends[key], color=colors[key], lw=8, ls=linestyles[key], zorder=1)
    plt.legend(loc='upper left', fontsize=27, fancybox=True)
    plt.xlim([-.01, 1.01])
    plt.xticks(np.linspace(0, 1, 6), fontsize=23)
    plt.ylim([1.79, 3.41])
    plt.yticks(np.linspace(1.8, 3.4, 5), fontsize=23)
    plt.grid()
    plt.xlabel(r'$\gamma$', fontsize=26)
    plt.ylabel('Interval Length', fontsize=26)
    plt.title('Impact of Model Bias'.format(k, n_sample), fontsize=30)
    plt.savefig('./results/model_bias={}_n={}.pdf'.format(k, n_sample))
    plt.close()


def plot_cov_shift(cov_cqr, cov_slcp, cov_split, gamma_vals, k, n_sample):
    keys = ['c', 'sl', 'sp']
    colors = {'c': '#2EA9DF', 'sl': '#90B44B', 'sp': '#F75C2F'}
    legends = {'c': 'CQR', 'sl': 'SLCP', 'sp': 'Split'}
    markers = {'c': '^', 'sl': 'D', 'sp': 'o'}
    linestyles = {'c': 'dashed', 'sl': 'dashed', 'sp': 'dashed'}
    plt.rcParams["figure.figsize"] = (10, 8.5)

    data = {'c': cov_cqr, 'sl': cov_slcp, 'sp': cov_split}
    for key in keys:
        plt.plot(gamma_vals[::-1], data[key], label=legends[key], color=colors[key], lw=3, ls=linestyles[key], fillstyle='none', marker=markers[key],
                 mec=colors[key], mew=2, ms=18)
    plt.legend(loc='best', fontsize=27, fancybox=True)
    plt.xlim([-.01, 1.01])
    plt.xticks(np.linspace(0, 1, 6), fontsize=23)
    plt.ylim([74, 96])
    plt.yticks(np.linspace(75, 95, 5), fontsize=23)
    plt.grid()
    plt.xlabel('Degree of Covariate Shift', fontsize=26)
    plt.ylabel('Coverage Rate', fontsize=26)
    plt.title('Impact of Covariate Shift'.format(k, n_sample), fontsize=30)
    plt.savefig('./results/cov_k={}_n={}.pdf'.format(k, n_sample))
    plt.close()


def plot_nn_capacity(lengths, cov_rates, hidden_size, dataset_name):
    keys = ['s', 'c', 'l']
    colors = {'s': '#D05A6E', 'c': '#3A8FB7', 'l': '#24936E'}
    legends = {'s': 'Split', 'c': 'CQR', 'l': 'SLCP'}
    names = {'s': 'split', 'c': 'cqr', 'l': 'slcp'}
    markers = {'s': 'o', 'c': '^', 'l': 'D'}
    linestyles = {'s': 'dashed', 'c': 'dashed', 'l': 'dashed'}
    plt.rcParams["figure.figsize"] = (10, 8.5)
    desired_rate = (1 - config.ConformalParams.alpha) * 100

    data = {'s': lengths['split'], 'c': lengths['cqr'], 'l': lengths['slcp']}
    for key in keys:
        # below = np.array(cov_rates[names[key]]) < desired_rate
        # above = np.array(cov_rates[names[key]]) >= desired_rate
        # x_below, x_above = np.array(hidden_size)[below], np.array(hidden_size)[above]
        # y_below, y_above = np.array(data[key])[below], np.array(data[key])[above]
        # plt.scatter(x_below, y_below, facecolors='none', edgecolors=colors[key], s=100, marker=markers[key])
        # plt.scatter(x_above, y_above, label=legends[key], c=colors[key], s=100, marker=markers[key])
        plt.plot(hidden_size, data[key], label=legends[key], c=colors[key], lw=3, ls=linestyles[key], fillstyle='none',
                 marker=markers[key], mec=colors[key], mew=2, ms=18)
    plt.legend(loc='best', fontsize=27, fancybox=True)
    plt.grid()
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xscale('log', basex=2) 
    plt.xlabel('NN Hidden Size', fontsize=26)
    plt.ylabel('Average Length', fontsize=26)
    plt.title('Ave. Length v.s. Model Capacity', fontsize=28)
    plt.savefig(f'./results/nn_capacity_{dataset_name}.pdf')
    plt.close()
