from cqr import helper
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc, RegressorNormalizer, AbsErrorErrFunc
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import pdb

def f(x):
    ''' Construct data (1D example)
    '''
    ax = 0*x
    for i in range(len(x)):
        ax[i] = np.random.poisson(np.sin(x[i])**2+0.1) + 0.03*x[i]*np.random.randn(1)
        ax[i] += 25*(np.random.uniform(0,1,1)<0.01)*np.random.randn(1)
    return ax.astype(np.float32)

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
n_train = 1900
# number of test examples (to evaluate average coverage and length)
n_test = 5000
k = 100

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

def model_bias_study(k, gamma):
    mean_estimator = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=min_samples_leaf, max_features=max_features, random_state=random_state)
    quantile_estimator = helper.QuantileForestRegressorAdapter(model=None, fit_params=None, quantiles=quantiles, params=params_qforest)

    nc_local = RegressorNc(quantile_estimator, True, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=alpha, model_2=mean_estimator, gamma=gamma)
    nc = RegressorNc(quantile_estimator, False, k, err_func=QuantileRegErrFunc(), alpha=alpha, model_2=mean_estimator, gamma=gamma)

    icp_local = IcpRegressor(nc_local, True, k, significance=alpha)
    icp = IcpRegressor(nc, False, k, significance=alpha)

    icp_local.fit(x_train[idx_train], y_train[idx_train])
    icp.fit(x_train[idx_train], y_train[idx_train])

    icp_local.calibrate(x_train[idx_cal], y_train[idx_cal])
    icp.calibrate(x_train[idx_cal], y_train[idx_cal])

    predictions_local = icp_local.predict(x_test, significance=alpha)
    predictions = icp.predict(x_test, significance=alpha)

    in_range_local = np.sum((y_test >= predictions_local[:, 0]) & (y_test <= predictions_local[:, 1])) / len(y_test) * 100
    in_range = np.sum((y_test >= predictions[:, 0]) & (y_test <= predictions[:, 1])) / len(y_test) * 100

    length = np.mean(predictions[:, 1] - predictions[:, 0])
    length_local = np.mean(predictions_local[:, 1] - predictions_local[:, 0])

    return length, length_local, in_range, in_range_local

gamma_vals = np.linspace(0, 1, 11)
# k_vals = np.arange(20, 1001, 20)
# length_all, length_local_all = [], []
# for k in tqdm(k_vals):
length_vals, length_local_vals = [], []
coverage_vals, coverage_local_vals = [], []
for gamma in tqdm(gamma_vals):
    length, length_local, in_range, in_range_local = model_bias_study(k, gamma)
    length_vals.append(length)
    length_local_vals.append(length_local)
    coverage_vals.append(in_range)
    coverage_local_vals.append(in_range_local)
    # length_all.append(length_vals)
    # length_local_all.append(length_local_vals)

# def plot_curve(length_vals, length_local_vals, coverage_vals, coverage_local_vals, gamma_vals, k, n_sample):
#     cqr = np.vstack([length_vals, coverage_vals])
#     ldcp = np.vstack([length_local_vals, coverage_local_vals])
#     plt.rcParams["figure.figsize"] = (10, 8.5)
#     cm_1 = plt.cm.get_cmap('cool')
#     cm_2 = plt.cm.get_cmap('Wistia')
#     sc_1 = plt.scatter(cqr[0], cqr[1], s=300, c=gamma_vals, marker='o', cmap=cm_1, label='CQR')
#     sc_2 = plt.scatter(ldcp[0], ldcp[1], s=300, c=gamma_vals, marker='D', cmap=cm_2, label='LDCP')
#     # cbar_1 = plt.colorbar(sc_1)
#     # cbar_2 = plt.colorbar(sc_2)
#     plt.legend(loc='upper right', fontsize=27, fancybox=True)
#     plt.title('Impact of Model Bias', fontsize=30)
#     plt.xlabel('Average Interval Length', fontsize=26)
#     plt.ylabel('Average Coverage Rate', fontsize=26)
#     plt.xlim([1.3, 1.93])
#     plt.xticks(np.linspace(1.35, 1.85, 6), fontsize=23)
#     plt.ylim([88.1, 91.3])
#     plt.yticks(np.linspace(88.2, 91.2, 6), fontsize=23)
#     # cbar_1.ax.tick_params(labelsize=24)
#     # cbar_2.ax.tick_params(labelsize=24)
#     plt.savefig('./model_bias={}_n={}.pdf'.format(k, n_sample))
#     plt.close()

def plot_curve(length_vals, length_local_vals, gamma_vals, k, n_sample):
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

# for i, k in enumerate(k_vals):
plot_curve(length_vals, length_local_vals, gamma_vals, k, n_train)