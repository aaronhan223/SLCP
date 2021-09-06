from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from sklearn.ensemble import RandomForestRegressor
import config
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pdb


class simulation:
    """ Functions for generating 1 dimensional simulation
    Parameters
    ----------
    rank: int, the number of simulation function
    """
    def __init__(self, rank) -> None:
        self.rank = rank

    def f_1(self, x):
        return np.sin(x) ** 2 + 0.1 + 0.6 * np.sin(2 * x) * np.random.randn(1)

    def f_2(self, x):
        return 2 * np.sin(x) ** 2 + 0.1 + 0.15 * x * np.random.randn(1)

    def f_3(self, x):
        x = np.random.poisson(np.sin(x) ** 2 + 0.1) + 0.08 * x * np.random.randn(1)
        x += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return x

    def generate(self, data):
        y = 0 * data
        for i in range(len(data)):
            if self.rank == 1:
                y[i] = self.f_1(data[i])
            elif self.rank == 2:
                y[i] = self.f_2(data[i])
            else:
                y[i] = self.f_3(data[i])
        return y.astype(np.float32)


def plot_func(x,
              y,
              y_u=None,
              y_l=None,
              pred=None,
              shade_color="",
              method_name="",
              title="",
              filename=None,
              save_figures=True):
    
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
    
    x_ = x[:config.UtilsParams.max_show]
    y_ = y[:config.UtilsParams.max_show]
    if y_u is not None:
        y_u_ = y_u[:config.UtilsParams.max_show]
    if y_l is not None:
        y_l_ = y_l[:config.UtilsParams.max_show]
    if pred is not None:
        pred_ = pred[:config.UtilsParams.max_show]

    fig = plt.figure()
    inds = np.argsort(np.squeeze(x_))
    plt.plot(x_[inds,:], y_[inds], 'k.', alpha=.2, markersize=10,
             fillstyle='none', label=u'Observations')
    
    if (y_u is not None) and (y_l is not None):
        plt.fill(np.concatenate([x_[inds], x_[inds][::-1]]),
                 np.concatenate([y_u_[inds], y_l_[inds][::-1]]),
                 alpha=.3, fc=shade_color, ec='None',
                 label = method_name + ' prediction interval')
    
    if pred is not None:
        if pred_.ndim == 2:
            plt.plot(x_[inds,:], pred_[inds,0], 'k', lw=2, alpha=0.9,
                     label=u'Predicted low and high quantiles')
            plt.plot(x_[inds,:], pred_[inds,1], 'k', lw=2, alpha=0.9)
        else:
            plt.plot(x_[inds,:], pred_[inds], 'k--', lw=2, alpha=0.9,
                     label=u'Predicted value')
    
    plt.ylim([-2.5, 7])
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.legend(loc='upper right')
    plt.title(title)
    if save_figures and (filename is not None):
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.close()

np.random.seed(config.UtilsParams.seed)

# training features
x_train = np.random.uniform(0, 5.0, size=config.UtilsParams.n_train).astype(np.float32)

# test features
x_test = np.random.uniform(0, 5.0, size=config.UtilsParams.n_test).astype(np.float32)

# generate labels
sim = simulation(3)
y_train = sim.generate(x_train)
y_test = sim.generate(x_test)

# reshape the features
x_train = np.reshape(x_train,(config.UtilsParams.n_train, 1))
x_test = np.reshape(x_test,(config.UtilsParams.n_test, 1))

# divide the data into proper training set and calibration set
idx = np.random.permutation(config.UtilsParams.n_train)
n_half = int(np.floor(config.UtilsParams.n_train / 2))
idx_train, idx_cal = idx[:n_half], idx[n_half: 2 * n_half]


def run_experiments_cqr(quantile_estimator, local, method, k=50):
    # define the CQR object, computing the absolute residual error of points 
    # located outside the estimated QRF band 
    if method == 'LDCP':
        nc = RegressorNc(quantile_estimator, local, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=config.ConformalParams.alpha)
    else:
        nc = RegressorNc(quantile_estimator, local, k, err_func=QuantileRegErrFunc(), alpha=config.ConformalParams.alpha)

    # build the split CQR object
    icp = IcpRegressor(nc, local, k, significance=config.ConformalParams.alpha)

    # fit the conditional quantile regression to the proper training data
    icp.fit(x_train[idx_train], y_train[idx_train])

    # compute the absolute errors on calibration data
    icp.calibrate(x_train[idx_cal], y_train[idx_cal])

    # produce predictions for the test set, with confidence equal to significance
    predictions = icp.predict(x_test, significance=config.ConformalParams.alpha)
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute the low and high conditional quantile estimation
    pred = quantile_estimator.predict(x_test)

    # display the results
    plot_func(x=x_test, y=y_test, y_u=y_upper, y_l=y_lower, pred=pred, shade_color=config.UtilsParams.cqr_color,
        method_name=method+":",title="",
        filename="./results/" + method + "_sim_3.png", save_figures=config.UtilsParams.save_figures)

    # compute and display the average coverage
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    print(method + " Random Forests: Percentage in the range (expecting " + str(100*(1 - config.ConformalParams.alpha)) + "%):",
        in_the_range / len(y_test) * 100)

    # compute length of the conformal interval per each test point
    length_cqr_rf = y_upper - y_lower

    # compute and display the average length
    print(method + " Random Forests: Average length:", np.mean(length_cqr_rf))

model = helper.QuantileForestRegressorAdapter(model=None, fit_params=None, quantiles=config.ConformalParams.quantiles, params=config.RandomForecastParams)
# model = helper.Linear_RegressorAdapter(model=None)
run_experiments_cqr(quantile_estimator=model, local=False, method='CQR')
run_experiments_cqr(quantile_estimator=model, local=True, method='LDCP', k=300)
