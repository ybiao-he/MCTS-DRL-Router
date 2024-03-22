import numpy as np
import tensorflow as tf
from roller import Roller
# from config import Params
from wrapper_env import env_wrapper

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from policy import Policy
from CREnv_v3 import CREnv
import csv


if __name__ == "__main__":

    import sys
    import os
    exp_id = sys.argv[1]
    board_id = sys.argv[2]

    env_version = 'env3'
    from importlib import import_module
    config_file = "config-{}".format(exp_id)
    config = import_module(config_file)
    params = config.Params()
    
    tf.random.set_seed(params.env.seed)
    np.random.seed(params.env.seed)

    env = CREnv(board_path="../simple_boards/board_{}.csv".format(board_id))
    env = env_wrapper(env, params.env)

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces

    roller = Roller(env, model, params.trainer.steps_per_epoch,
                    params.trainer.gamma, params.trainer.lam)               # Define Roller for genrating rollouts for training

    ppo = Policy(model=model, params=params)

    tensorboard_dir = './tensorboard/board_{}_{}'.format(env_version, exp_id)
    writer = tf.summary.create_file_writer(tensorboard_dir)

    results_dict = {'pi_loss':[], 'v_loss':[], 'entropy_loss':[], 'approx_ent':[], 'approx_kl':[], 'ep_rews':[]}
    with writer.as_default():

        best_rews = -900
        for epoch in range(params.trainer.epochs):
            
            rollouts, infos = roller.rollout(epoch)
            outs = ppo.update(rollouts)
            
            tf.summary.scalar('pi_loss', outs['pi_loss'], epoch)
            results_dict['pi_loss'].append(outs['pi_loss'])
            tf.summary.scalar('v_loss', outs['v_loss'], epoch)
            results_dict['v_loss'].append(outs['v_loss'])
            tf.summary.scalar('entropy_loss', outs['entropy_loss'], epoch)
            results_dict['entropy_loss'].append(outs['entropy_loss'])
            tf.summary.scalar('approx_ent', outs['approx_ent'], epoch)
            results_dict['approx_ent'].append(outs['approx_ent'])
            tf.summary.scalar('approx_kl', outs['approx_kl'], epoch)
            results_dict['approx_kl'].append(outs['approx_kl'])
            tf.summary.scalar('eps_reward', np.mean(infos['ep_rews']), epoch)
            results_dict['ep_rews'].append(np.mean(infos['ep_rews']))

            if np.mean(infos['ep_rews'])>best_rews:
                # model.save_weights( 'model/epoch_{}'.format(epoch), save_format='tf')
                model.save_weights( 'results_{}_{}/best_{}_{}'.format(env_version, board_id, board_id, exp_id), save_format='tf')
                best_rews = np.mean(infos['ep_rews'])

        model.save_weights( 'results_{}_{}/epoch_{}_{}_{}'.format(env_version, board_id, epoch, board_id, exp_id), save_format='tf')

    with open('results_{}_{}/learning_{}_{}.csv'.format(env_version, board_id, board_id, exp_id), 'w') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in results_dict.items():
           writer.writerow([key]+value)