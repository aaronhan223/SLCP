from all_experiments import run_pred_experiment, run_cov_shift, run_model_bias
from tqdm import tqdm
import utils
import config
import logging
import os

logger = logging.getLogger('LDCP.main')


if __name__ == '__main__':

    if not os.path.exists(os.path.join(os.getcwd(), 'results')):
        os.mkdir(os.path.join(os.getcwd(), 'results'))
    utils.set_logger(os.path.join('./results', 'history.log'))    
    logger.info('\n\n<---------------NEW RUN--------------->')

    if config.UtilsParams.experiment == 'prediction':
        logger.info('Running conformal prediction task.')
        dataset_list = ['simulation_1', 
                        'simulation_2', 
                        'simulation_3', 
                        'star', 
                        'meps_19', 
                        'meps_20', 
                        'meps_21', 
                        'facebook_1', 
                        'facebook_2', 
                        'bio', 
                        'blog_data', 
                        'bike', 
                        'concrete', 
                        'community']
        model_list = ['random_forest', 'linear', 'neural_net']
        method_name = ['ldcp', 'cqr', 'cqr-asy', 'split', 'lacp', 'qr']

        for data in tqdm(dataset_list):
            for model in tqdm(model_list):
                for method in tqdm(method_name):
                    logger.info('~~~~~~~~~~~~~~~~~~~~~~~~~~')
                    logger.info(f'Dataset: {data} | Model: {model} | Method: {method}.')
                    is_cp = False if method == 'qr' else True
                    run_pred_experiment(dataset_name=data, 
                                        model_name=model, 
                                        method_name=method, 
                                        random_seed=config.UtilsParams.seed,
                                        conformal=is_cp)
                logger.info('=======================================================================')

    if config.UtilsParams.experiment == 'cov_shift':
        logger.info('Running covariate shift experiment.')
        run_cov_shift()

    if config.UtilsParams.experiment == 'model_bias':
        logger.info('Running model bias experiment.')
        run_model_bias()
    
    logger.info('Program done!')