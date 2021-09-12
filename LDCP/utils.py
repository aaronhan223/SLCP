import matplotlib.pyplot as plt
import numpy as np
import random
import config
import torch
import logging


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
    _logger = logging.getLogger('LDCP')
    _logger.setLevel(logging.INFO)

    fmt = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', '%Y-%m-%d %H:%M:%S')

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    _logger.addHandler(file_handler)


def plot_pred(x, y, y_u=None, y_l=None, pred=None, shade_color="", method_name="", title="", filename=None, save_figures=True):
    
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
    if pred is not None:
        pred_ = pred[:config.DataParams.max_show]

    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds,:], y_[inds], 'k.', alpha=.2, markersize=10, fillstyle='none', label=u'Observations')
    
    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x_[inds], x_[inds][::-1]]),
                 np.concatenate([y_u_[inds], y_l_[inds][::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' prediction interval')
    
    if pred is not None:
        if pred_.ndim == 2:
            plt.plot(x_[inds,:], pred_[inds,0], 'k', lw=2, alpha=0.9, label=u'Predicted low and high quantiles')
            plt.plot(x_[inds,:], pred_[inds,1], 'k', lw=2, alpha=0.9)
        else:
            plt.plot(x_[inds,:], pred_[inds], 'k--', lw=2, alpha=0.9, label=u'Predicted value')
    
    plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()


def plot_model_bias(length_vals, length_local_vals, gamma_vals, k, n_sample):
    keys = ['c', 'd']
    colors = {'c': '#D05A6E', 'd': '#3A8FB7'}
    legends = {'c': 'CQR', 'd': 'LDCP'}
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


def plot_cov_shift(length_vals, length_local_vals, gamma_vals, k, n_sample):
    keys = ['c', 'd']
    colors = {'c': '#D05A6E', 'd': '#3A8FB7'}
    legends = {'c': 'CQR', 'd': 'LDCP'}
    markers = {'c': 'o', 'd': 'o'}
    linestyles = {'c': 'solid', 'd': 'solid'}
    plt.rcParams["figure.figsize"] = (10, 8.5)

    data = {'c': length_vals, 'd': length_local_vals}
    for key in keys:
        plt.plot(gamma_vals[::-1], data[key], label=legends[key], color=colors[key], lw=8, ls=linestyles[key], zorder=1)
    plt.legend(loc='upper left', fontsize=27, fancybox=True)
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