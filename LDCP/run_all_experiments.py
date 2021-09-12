from all_experiments import run_pred_experiment, run_cov_shift, run_model_bias
import config


if __name__ == '__main__':

    if config.UtilsParams.experiment == 'prediction':
        dataset_list = ['simulation_1', 'simulation_2', 'simulation_3', 'star', 'facebook_1', 'facebook_2', 'bio', 'blog_data', 'bike', 'concrete', 'community']
        model_list = ['random_forest', 'linear', 'neural_net']
        method_name = ['ldcp', 'cqr', 'cqr-asy', 'split', 'lacp', 'qr']

        for data in dataset_list:
            for model in model_list:
                for method in method_name:
                    is_cp = False if method == 'qr' else True
                    run_pred_experiment(dataset_name=data, 
                                        model_name=model, 
                                        method_name=method, 
                                        random_seed=config.UtilsParams.seed,
                                        conformal=is_cp)

    if config.UtilsParams.experiment == 'cov_shift':
        run_cov_shift()
    if config.UtilsParams.experiment == 'model_bias':
        run_model_bias()