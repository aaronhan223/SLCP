import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.model_selection import train_test_split
import config


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
    
    def f(self, x):
        x = np.random.poisson(np.sin(x) ** 2 + 0.1) + 0.03 * x * np.random.randn(1)
        x += 25 * (np.random.uniform(0, 1, 1) < 0.01) * np.random.randn(1)
        return x

    def generate(self, data):
        y = 0 * data
        for i in range(len(data)):
            if self.rank == 1:
                y[i] = self.f_1(data[i])
            elif self.rank == 2:
                y[i] = self.f_2(data[i])
            elif self.rank == 3:
                y[i] = self.f_3(data[i])
            else:
                y[i] = self.f(data[i])
        return y.astype(np.float32)


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
        return config.DataParams.left_interval + (config.DataParams.right_interval - config.DataParams.left_interval) * np.expand_dims(beta.rvs(a, b, size=size), 1)
    return config.DataParams.left_interval + (config.DataParams.right_interval - config.DataParams.left_interval) * np.random.rand(size, 1)


def mu_model(x):
    k = [1.0]
    b = -0.0
    return np.sum(k * x, axis=-1) + b 


def sigma_model(x):
    x_abs = np.abs(x)
    return (x_abs / (x_abs + 1)).reshape(-1)


def GetDataset(name, base_path, seed, test_ratio, a=1., b=1.):

    if 'simulation' in name:
        x_train = np.random.uniform(0, 5.0, size=config.DataParams.n_train).astype(np.float32)
        x_test = np.random.uniform(0, 5.0, size=config.DataParams.n_test).astype(np.float32)

        if '1' in name:
            sim = simulation(1)
        elif '2' in name:
            sim = simulation(2)
        elif '3' in name:
            sim = simulation(3)
        else:
            sim = simulation(4)

        y_train = sim.generate(x_train)
        y_test = sim.generate(x_test)
        x_train = np.reshape(x_train, (config.DataParams.n_train, 1))
        x_test = np.reshape(x_test, (config.DataParams.n_test, 1))
    
    if name == 'cov_shift':
        data_model = GaussianDataGenerator(px_model, mu_model, sigma_model)
        x_train, y_train = data_model.generate(config.DataParams.n_train)
        x_test, y_test = data_model.generate(config.DataParams.n_test, a=a, b=b)

    if name == "community":
        # https://github.com/vbordalo/Communities-Crime/blob/master/Crime_v1.ipynb
        attrib = pd.read_csv(base_path + 'communities_attributes.csv', delim_whitespace = True)
        data = pd.read_csv(base_path + 'communities.data', names = attrib['attributes'])
        data = data.drop(columns=['state','county', 'community','communityname', 'fold'], axis=1)
        data = data.replace('?', np.nan)
        
        # Impute mean values for samples with missing values        
        from sklearn.impute import SimpleImputer
        
        imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
        
        imputer = imputer.fit(data[['OtherPerCap']])
        data[['OtherPerCap']] = imputer.transform(data[['OtherPerCap']])
        data = data.dropna(axis=1)
        X = data.iloc[:, 0:100].values
        y = data.iloc[:, 100].values
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=seed)
    return x_train, x_test, y_train, y_test