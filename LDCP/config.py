
class ConformalParams:

    alpha = 0.1 # desired miscoverage error
    quantiles = [5, 95] # low and high target quantiles
    k = 300


class RandomForecastParams:

    n_estimators = 100
    min_samples_leaf = 40
    max_features = 1
    random_state = 0
    cross_valid = True
    coverage_factor = 0.9
    test_ratio = 0.1
    range_vals = 10
    num_vals = 4


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
    left_interval = -1
    right_interval = 1
