import numpy as np
import tensorflow as tf
from roller import Roller
from config import Params
from wrapper_env import env_wrapper

from nn_architecures import network_builder
from models import CategoricalModel
from policy import Policy
from CREnv_v0 import CREnv

import time

if __name__ == "__main__":

    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)
    # board_idx = 'I1'

    # gpus = tf.config.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         # Currently, memory growth needs to be the same across GPUs
    #         for gpu in gpus:
    #           tf.config.experimental.set_memory_growth(gpu, True)
    #         logical_gpus = tf.config.list_logical_devices('GPU')
    #         print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #     except RuntimeError as e:
    #         # Memory growth must be set before GPUs have been initialized
    #         print(e)

    import sys
    import os
    boards_directory = sys.argv[1]
    board_idx = sys.argv[2]

    params = Params()          # Get Configuration | HORIZON = Steps per epoch
    board_name = "board_{}.csv".format(board_idx)
    
    tf.random.set_seed(params.env.seed)                                     # Set Random Seeds for np and tf
    np.random.seed(params.env.seed)

    env = CREnv(board_path=os.path.join(boards_directory, board_name))
    env = env_wrapper(env, params.env)

    network = network_builder(params.trainer.nn_architecure)(hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)    # Build Neural Network with Forward Pass

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces

    if params.trainer.load_model:
        print("loading a pretrained model..........")
        model.load_weights("./saved_model_old/II4_494")
    
    roller = Roller(env, model, params.trainer.steps_per_epoch,
                    params.trainer.gamma, params.trainer.lam)               # Define Roller for genrating rollouts for training

    ppo = Policy(model=model)            # Define PPO Policy with combined loss

    tensorboard_dir = './tensorboard'
    writer = tf.summary.create_file_writer(tensorboard_dir)
    with writer.as_default():

        for epoch in range(params.trainer.epochs):                              # Main training loop for n epochs

            start_time = time.time()
            rollouts, infos = roller.rollout()
            # print(rollouts['rews'])
            print("Running time for rollout is: {}".format(time.time()-start_time))
            outs = ppo.update(rollouts)                                         # Push rollout in ppo and update policy accordingly
            end_time = time.time()
            print("Running time for uodating policy is: {}".format(end_time-start_time))
            tf.summary.scalar('pi_loss', np.mean(outs['pi_loss']), epoch)
            tf.summary.scalar('v_loss', np.mean(outs['v_loss']), epoch)
            tf.summary.scalar('entropy_loss', np.mean(outs['entropy_loss']), epoch)
            tf.summary.scalar('approx_ent', np.mean(outs['approx_ent']), epoch)
            tf.summary.scalar('approx_kl', np.mean(outs['approx_kl']), epoch)
            tf.summary.scalar('eps_reward', np.mean(infos['ep_rews']), epoch)
            tf.summary.scalar('pi_loss', outs['pi_loss'], epoch)
            writer.flush()

            model.save_weights( 'saved_model/{}_{}'.format(board_idx, epoch), 
                                save_format='tf')
    # env.close()                                                             # Dont forget closing the environment
