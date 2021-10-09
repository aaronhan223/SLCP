
class ConformalParams:

    alpha = 0.1 # desired miscoverage error
    quantiles = [5, 95] # low and high target quantiles
    k = 300
    valid_ratio = 0.5


class RandomForecastParams:

    n_estimators = 1000 # [10, 40, 70, 200, 500]
    min_samples_leaf = 1 
    max_features = 'auto' # ['auto', 'sqrt', 'log2', None, 0.5]
    random_state = 1
    cross_valid = True
    coverage_factor = 0.85
    test_ratio = 0.05
    range_vals = 30
    num_vals = 10


class NeuralNetParams:

    epochs = 1000
    lr = 0.0005
    hidden_size = 64
    batch_size = 64
    dropout = 0.1
    wd = 1e-6
    test_ratio = 0.05
    random_state = 1


class LinearParams:

    epochs = 100
    lr = 0.01
    batch_size = 64
    wd = 1e-6
    test_ratio = 0.05
    random_state = 1


class UtilsParams:

    save_figures = True
    seed = 1
    experiment = 'prediction'
    split_color = 'tomato'
    local_color = 'gray'
    cqr_color = 'lightblue'


class DataParams:

    n_train = 2000 # number of training examples
    n_test = 5000 # number of test examples (to evaluate average coverage and length)
    max_show = 1000 # maximal number of testpoints to plot
    test_ratio = 0.2 # train test split ratio
    left_interval = -1
    right_interval = 1
