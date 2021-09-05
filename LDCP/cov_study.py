from scipy.stats import beta
import numpy as np
from cqr import helper
import matplotlib.pyplot as plt
from nonconformist.nc import RegressorNc
from nonconformist.icp import IcpRegressor
from nonconformist.nc import QuantileRegErrFunc, QuantileRegAsymmetricErrFunc

L, R = -1, 1
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

class GaussianDataGenerator(object):
    def __init__(self, px_model, mu_model, sigma_model):
        self.px_model = px_model
        self.mu_model = mu_model
        self.sigma_model = sigma_model
    
    def generate(self, size, **kwargs):
        if 'a' in kwargs:
            a = kwargs.pop('a')
            b = kwargs.pop('b')
            X = self.px_model(size, a=a, b=b)
        else:
            X = self.px_model(size)
        Y = self.mu_model(X) + self.sigma_model(X) * np.random.randn(size)
        return X, Y

def px_model(size, **kwargs):
    if 'a' in kwargs:
        a = kwargs.pop('a')
        b = kwargs.pop('b')
        return L + (R - L) * np.expand_dims(beta.rvs(a, b, size=size), 1)
    return L + (R - L) * np.random.rand(size, 1)

def mu_model(x):
    k = [1.0]
    b = -0.0
    return np.sum(k * x, axis=-1) + b 

def sigma_model(x):
    x_abs = np.abs(x)
    return (x_abs / (x_abs + 1)).reshape(-1)

def run_exp_cov_shift(a, b, k):
    quantile_estimator = helper.Linear_RegressorAdapter(model=None)
    quantile_estimator_local = helper.Linear_RegressorAdapter(model=None)

    # define a CQR object, computes the absolute residual error of points 
    # located outside the estimated quantile neural network band 
    nc_local = RegressorNc(quantile_estimator_local, True, k, err_func=QuantileRegAsymmetricErrFunc(), alpha=alpha)
    nc = RegressorNc(quantile_estimator, False, k, err_func=QuantileRegErrFunc(), alpha=alpha)

    x_test, y_test = data_model.generate(n_test, a=a, b=b)

    # run CQR procedure
    y_lower_local, y_upper_local = helper.run_icp(nc_local, x_train, y_train, x_test, idx_train, idx_cal, alpha, True, k)

    y_lower, y_upper = helper.run_icp(nc, x_train, y_train, x_test, idx_train, idx_cal, alpha, False, k)
    # compute and print average coverage and average length
    coverage_local, length_local = helper.compute_coverage(y_test, y_lower_local, y_upper_local, alpha, 'DDCP' + " Linear Regr")
    coverage_cp_qnet, length_cp_qnet = helper.compute_coverage(y_test, y_lower, y_upper, alpha, 'CQR' + " Linear Regr")
    return coverage_local, coverage_cp_qnet

def plot_curve(length_vals, length_local_vals, gamma_vals, k, n_sample):
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

data_model = GaussianDataGenerator(px_model, mu_model, sigma_model)
n_train, n_test = 2000, 5000
x_train, y_train = data_model.generate(n_train)
idx = np.random.permutation(n_train)
n_half = int(np.floor(n_train/2))
idx_train, idx_cal = idx[:n_half], idx[n_half:2*n_half]

a_list = np.linspace(0, 1, 11)
b_list = np.linspace(0, 1, 11)

coverage_local_all, coverage_cp_all = [], []
k = 100
for a, b in zip(a_list, b_list):
    if a == 0.:
        a += .01
        b += .01
    a, b = np.round(a, 2), np.round(b, 2)
    coverage_local, coverage_cp = run_exp_cov_shift(a, b, k)
    coverage_local_all.append(coverage_local)
    coverage_cp_all.append(coverage_cp)

plot_curve(coverage_cp_all, coverage_local_all, a_list, k, n_train)
