# This script is just for checking if env is correct
import numpy as np
import tensorflow as tf
from config import Params
from wrapper_env import env_wrapper
import gym

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from policy import Policy
from copy import copy
from copy import deepcopy
import os

def draw_board(paths_x, paths_y, board, save_name):
    
    import matplotlib.pyplot as plt
    width, height = board.shape

    # create a 8" x 8" board
    fig = plt.figure(figsize=[8,8])
    # fig.patch.set_facecolor((1,1,.8))

    ax = fig.add_subplot(111)

    # draw the grid
    for x in range(30):
        ax.plot([x, x], [0,29], color=(0.5,0.5,0.5,1))
    for y in range(30):
        ax.plot([0, 29], [y,y], color=(0.5,0.5,0.5,1))

    # draw paths
    for p in range(len(paths_x)):

        ph = plt.subplot()
        ph.plot(paths_y[p], paths_x[p], linewidth=5, color='black')

    # draw obstacles
    x_axis = []
    y_axis = []
    nets = dict()
    for x in range(width):
        for y in range(height):
            if board[x, y]!=0:
                x_axis.append(y)
                y_axis.append(x)
                if board[x, y]!=1:
                    nets[(x,y)] = board[x, y]

    ax.scatter(x_axis, y_axis, marker='s', s=250, c='k')

    for xy in nets:
        ax.text(xy[1], xy[0], str(int(nets[xy])-1), fontsize=18, color='w',
                horizontalalignment='center', verticalalignment='center')

    # scale the axis area to fill the whole figure
    ax.set_position([0,0,1,1])

    # get rid of axes and everything (the figure background will show through)
    ax.set_axis_off()

    # scale the plot area conveniently (the board is in 0,0..18,18)
    ax.set_xlim(0,29)
    ax.set_ylim(0,29)
    
    fig.savefig(save_name, bbox_inches='tight')


# rollout using MCTS env
def rl_rollout(env, model, res_idx):

    res_folder_name = "route_results"
    if not os.path.isdir(res_folder_name):
        os.mkdir(res_folder_name)
    
    saved_fig_name = os.path.join(res_folder_name, "rl_route_board_{}.png".format(res_idx))

    env.reset()
    state = env
    done = False

    board = np.absolute(np.genfromtxt("./board.csv", delimiter=','))

    routed_paths = [state.head]
    while not done:

        obs_vec = np.expand_dims(state.board_embedding(), axis=0)
        obs_vis = None
        action_t, logp_t, value_t = model.get_action_logp_value({"vec_obs": obs_vec, "vis_obs": obs_vis})

        state.step(action_t)
        routed_paths.append(state.head)
        done = state.isTerminal()

    paths_x, paths_y = separate_paths(state, routed_paths)
    draw_board(paths_x, paths_y, board, saved_fig_name)

    return state.getReward()

def separate_paths(env, routed_paths):
    
    env.reset()
    ret_paths_x = []
    ret_paths_y = []
    single_path = [env.head]

    start_pins = list(env.start.values())

    for v in routed_paths:
        if v in start_pins:
            ret_paths_x.append([node[0] for node in single_path])
            ret_paths_y.append([node[1] for node in single_path])
            single_path = []
        single_path.append(v)

    ret_paths_x.append([node[0] for node in single_path])
    ret_paths_y.append([node[1] for node in single_path])

    return ret_paths_x, ret_paths_y

if __name__ == "__main__":

    from CREnv_v4 import CREnv
    # physical_devices = tf.config.list_physical_devices('GPU') 
    # tf.config.experimental.set_memory_growth(physical_devices[0], True)

    params = Params()          # Get Configuration | HORIZON = Steps per epoch

    tf.random.set_seed(params.env.seed)                                     # Set Random Seeds for np and tf
    np.random.seed(params.env.seed)

    env = CREnv()
    env = env_wrapper(env, params.env)

    network = network_builder(params.trainer.nn_architecure) \
        (hidden_sizes=params.policy.hidden_sizes, env_info=env.env_info)    # Build Neural Network with Forward Pass

    model = CategoricalModel if env.env_info.is_discrete else GaussianModel
    model = model(network=network, env_info=env.env_info)                   # Build Model for Discrete or Continuous Spaces

    for i in range(35):
        print('Loading Model ...')
        model.load_weights("./saved_model/allNets_vec_oneHit_{}".format(i))           # Load model if load_model flag set to true
        test_env = CREnv()
        ep_rew = rl_rollout(test_env, model, i)
