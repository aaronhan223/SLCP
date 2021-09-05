from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pdb


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
    
    x_ = x[:max_show]
    y_ = y[:max_show]
    if y_u is not None:
        y_u_ = y_u[:max_show]
    if y_l is not None:
        y_l_ = y_l[:max_show]
    if pred is not None:
        pred_ = pred[:max_show]

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


def f(x):
    ''' Construct data (1D example)
    '''
    ax = 0*x
    for i in range(len(x)):
        # ax[i] = np.sin(x[i])**2+0.1 + 0.6*np.sin(2*x[i])*np.random.randn(1)
        # ax[i] = 2*np.sin(x[i])**2+0.1 + 0.15*x[i]*np.random.randn(1)
        ax[i] = np.random.poisson(np.sin(x[i])**2+0.1) + 0.08*x[i]*np.random.randn(1)
        ax[i] += 25*(np.random.uniform(0,1,1)<0.01)*np.random.randn(1)
    return ax.astype(np.float32)

split_color = 'tomato'
local_color = 'gray'
cqr_color = 'lightblue'

np.random.seed(1)

# desired miscoverage error
alpha = 0.1
# low and high target quantiles
quantiles = [5, 95]

# maximal number of testpoints to plot
max_show = 1000

# save figures?
save_figures = False

# parameters of random forests
n_estimators = 100
min_samples_leaf = 40
max_features = 1 # 1D signal
random_state = 0

# number of training examples
n_train = 2000
# number of test examples (to evaluate average coverage and length)
n_test = 5000
# k = 100

# training features
x_train = np.random.uniform(0, 5.0, size=n_train).astype(np.float32)

# test features
x_test = np.random.uniform(0, 5.0, size=n_test).astype(np.float32)

# generate labels
y_train = f(x_train)
y_test = f(x_test)

# reshape the features
x_train = np.reshape(x_train,(n_train,1))
x_test = np.reshape(x_test,(n_test,1))

# divide the data into proper training set and calibration set
idx = np.random.permutation(n_train)
n_half = int(np.floor(n_train/2))
idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

# define quantile random forests (QRF) parameters
params_qforest = dict()
params_qforest["n_estimators"] = n_estimators
params_qforest["min_samples_leaf"] = min_samples_leaf
params_qforest["max_features"] = max_features
params_qforest["CV"] = True
params_qforest["coverage_factor"] = 0.9
params_qforest["test_ratio"] = 0.1
params_qforest["random_state"] = random_state
params_qforest["range_vals"] = 10
params_qforest["num_vals"] = 4


def run_experiments_cqr(quantile_estimator, local, method, k=50):
    # define the CQR object, computing the absolute residual error of points 
    # located outside the estimated QRF band 
    if method == 'LDCP':
        nc = RegressorNc(quantile_estimator, local, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=alpha)
    else:
        nc = RegressorNc(quantile_estimator, local, k, err_func=QuantileRegErrFunc(), alpha=alpha)

    # build the split CQR object
    icp = IcpRegressor(nc, local, k, significance=alpha)

    # fit the conditional quantile regression to the proper training data
    icp.fit(x_train[idx_train], y_train[idx_train])

    # compute the absolute errors on calibration data
    icp.calibrate(x_train[idx_cal], y_train[idx_cal])

    # produce predictions for the test set, with confidence equal to significance
    predictions = icp.predict(x_test, significance=alpha)
    y_lower = predictions[:,0]
    y_upper = predictions[:,1]

    # compute the low and high conditional quantile estimation
    pred = quantile_estimator.predict(x_test)

    # display the results
    plot_func(x=x_test,y=y_test,y_u=y_upper,y_l=y_lower,pred=pred,shade_color=cqr_color,
        method_name=method+":",title="",
        filename="./results/" + method + "_sim_3.png", save_figures=True)

    # compute and display the average coverage
    in_the_range = np.sum((y_test >= y_lower) & (y_test <= y_upper))
    print(method + " Random Forests: Percentage in the range (expecting " + str(100*(1-alpha)) + "%):",
        in_the_range / len(y_test) * 100)

    # compute length of the conformal interval per each test point
    length_cqr_rf = y_upper - y_lower

    # compute and display the average length
    print(method + " Random Forests: Average length:", np.mean(length_cqr_rf))

model = helper.QuantileForestRegressorAdapter(model=None, fit_params=None, quantiles=quantiles, params=params_qforest)
# model = helper.Linear_RegressorAdapter(model=None)
run_experiments_cqr(quantile_estimator=model, local=False, method='CQR')
run_experiments_cqr(quantile_estimator=model, local=True, method='LDCP', k=300)
