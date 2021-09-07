import pandas as pd
import numpy as np
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


def GetDataset(name, path):

    if 'simulation' in name:
        x_train = np.random.uniform(0, 5.0, size=config.UtilsParams.n_train).astype(np.float32)
        x_test = np.random.uniform(0, 5.0, size=config.UtilsParams.n_test).astype(np.float32)

        if '1' in name:
            sim = simulation(1)
        if '2' in name:
            sim = simulation(2)
        else:
            sim = simulation(3)

        y_train = sim.generate(x_train)
        y_test = sim.generate(x_test)
        x_train = np.reshape(x_train,(config.UtilsParams.n_train, 1))
        x_test = np.reshape(x_test,(config.UtilsParams.n_test, 1))
        
    return x_train, x_test, y_train, y_test