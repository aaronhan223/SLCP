from all_experiments import run_pred_experiment, run_cov_shift, run_model_bias
import config


if __name__ == '__main__':
    # TODO: 1. add local conformal, 2. add and test real data from original dataset.py file, 3. assymmetric quantile error function
    if config.UtilsParams.experiment == 'prediction':
        dataset_list = ['simulation_1', 'simulation_2', 'simulation_3', 'star', 'facebook_1', 'facebook_2', 'bio', 'blog_data', 'bike', 'concrete', 'community']
        model_list = ['random_forest', 'linear_regression', 'neural_net']
        method_name = ['ldcp', 'cqr', 'cqr-asy', 'split', 'lacp']
        run_pred_experiment(dataset_name='bio', model_name='linear_regression', method_name='ldcp', random_seed=config.UtilsParams.seed)
        run_pred_experiment(dataset_name='bio', model_name='linear_regression', method_name='cqr', random_seed=config.UtilsParams.seed)
        run_pred_experiment(dataset_name='bio', model_name='linear_regression', method_name='lacp', random_seed=config.UtilsParams.seed)
        run_pred_experiment(dataset_name='bio', model_name='linear_regression', method_name='split', random_seed=config.UtilsParams.seed)
    if config.UtilsParams.experiment == 'cov_shift':
        run_cov_shift()
    if config.UtilsParams.experiment == 'model_bias':
        run_model_bias()