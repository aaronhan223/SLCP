from all_experiments.prediction import run_pred_experiment, run_age_pred
from all_experiments.cov_shift import run_cov_shift
from all_experiments.model_bias import run_model_bias
from all_experiments.nn_capacity import run_nn_capacity
from all_experiments.toy_plot import plot_toy_example
from datasets.datasets import GaussianDataGenerator, px_model, mu_model, sigma_model
from tqdm import tqdm
import numpy as np
import utils
import config
import pandas as pd
import logging
import os
import pdb

logger = logging.getLogger('SLCP.main')


if __name__ == '__main__':

    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))
    utils.set_logger(os.path.join('./results', f'history_{config.UtilsParams.experiment}_{config.ConformalParams.k}_{config.ConformalParams.valid_ratio}.log'))    
    logger.info('\n\n<---------------NEW RUN--------------->')

    if config.UtilsParams.experiment == 'prediction':
        logger.info('Running conformal prediction task.')
        dataset_list = [
                        # 'simulation_1', 
                        # 'simulation_2', 
                        # 'simulation_3', 
                        # 'star',
                        # 'meps_19', 
                        # 'meps_20', 
                        # 'meps_21', 
                        # 'facebook_1', 
                        # 'facebook_2', 
                        # 'bio', 
                        # 'blog_data', 
                        # 'bike', 
                        # 'concrete', 
                        # 'community',
                        'age'
                        ]
        model_list = [
                    #   'random_forest', 
                    #   'linear', 
                    #   'neural_net',
                    #   'kde',
                      'age_regr'
                      ]
        method_name = [
                       'slcp-knn', 
                       'slcp-rbf', 
                       'slcp-mean',
                    #    'cqr', 
                    #    'cqr-asy', 
                    #    'split', 
                    #    'lacp', 
                    #    'qr'
                       ]

        all_cov_rate = np.zeros((len(model_list), len(method_name)))
        all_length = np.zeros((len(model_list), len(method_name)))
        for data in tqdm(dataset_list):
            for i, model in enumerate(model_list):
                for j, method in enumerate(method_name):
                    logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    logger.info(f'Dataset: {data} | Model: {model} | Method: {method}.')
                    is_cp = False if method == 'qr' else True
                    if data != 'age':
                        cov_rate, length = run_pred_experiment(dataset_name=data, 
                                                                model_name=model, 
                                                                method_name=method, 
                                                                random_seed=config.UtilsParams.seed,
                                                                conformal=is_cp)
                    else:
                        cov_rate, length = run_age_pred(dataset_name=data, 
                                                        model_name=model, 
                                                        method_name=method, 
                                                        random_seed=config.UtilsParams.seed)
                    all_cov_rate[i, j] += cov_rate
                    all_length[i, j] += length
                logger.info('=======================================================================')

        all_cov_rate /= len(dataset_list)
        all_length /= len(dataset_list)
        cov_rate_result = pd.DataFrame(data=all_cov_rate, index=model_list, columns=method_name)
        length_result = pd.DataFrame(data=all_length, index=model_list, columns=method_name)
        cov_rate_result.to_csv(f'./results/cov_rate_plot_{config.ConformalParams.k}_{config.ConformalParams.valid_ratio}.csv')
        length_result.to_csv(f'./results/length_plot_{config.ConformalParams.k}_{config.ConformalParams.valid_ratio}.csv')

    if config.UtilsParams.experiment == 'cov_shift':
        logger.info('Running covariate shift experiment.')
        run_cov_shift()

    if config.UtilsParams.experiment == 'model_bias':
        logger.info('Running model bias experiment.')
        run_model_bias()
    
    if config.UtilsParams.experiment == 'toy_plot':
        logger.info('Running toy plot experiment.')
        plot_toy_example()

    if config.UtilsParams.experiment == 'nn_capacity':
        logger.info('Running NN capacity experiment.')
        dataset = 'simulation_2'
        run_nn_capacity(config.UtilsParams.seed, dataset)
        
    logger.info('Program done!')