import numpy as np
import tensorflow as tf
from config import Params, EnvInfo
import gym

from nn_architecures import network_builder
from models import CategoricalModel, GaussianModel
from copy import copy
from copy import deepcopy
import os

from scipy.spatial import distance

from astar import Astar


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


def visualize_path(board, path, starts, rollout_idx):

    paths_x, paths_y = separate_paths(starts, path)

    res_folder_name = "route_results"
    if not os.path.isdir(res_folder_name):
        os.mkdir(res_folder_name)

    saved_fig_name = os.path.join(res_folder_name, "oneNet_board_{}.png".format(rollout_idx))

    draw_board(paths_x, paths_y, board, saved_fig_name)


def mcts_DFS_rollout(state, model):

    '''
    This function is DFS based rollout for MCTS, now it just rollout for one net
    '''
    paths = []
    ini_state = deepcopy(state)
    while not ini_state.isTerminal():

        # Following 3 line is to deal with the case where the selection of MCTS reaches the target
        possible_actions = ini_state.getPossibleActions()
        if len(possible_actions)>0:
            if possible_actions[0]==ini_state.finish[state.pairs_idx]:
                ini_state = ini_state.takeAction(possible_actions[0], is_tuple=True)
                paths += [ini_state.head, init_state.finish[state.pairs_idx]]
                continue
        
        path = prob_DFS(ini_state, model)
        # checking if the path exists, non-existence can be caused be a bad selection of mcts
        if path is None:
            print("DFS doesn't find a path!!!")
            return -90, [ini_state.head]

        path.append(ini_state.finish[ini_state.pairs_idx])
        print(path)
        paths += path
        for node in path[1:]:
            action = tuple(np.array(node)-np.array(ini_state.head))
            ini_state = ini_state.takeAction(action, is_tuple=True)
    # print("total reward is: {}".format(ini_state.getReward()))

    if len(paths)==0:
    	paths=[state.head]
    return ini_state.getReward(), paths

def block_other_nets(check_state, path):

    state = deepcopy(check_state)
    starts = state.start
    starts.pop(state.pairs_idx)
    finishs = state.finish
    finishs.pop(state.pairs_idx)
    maze = state.board
    for p in path:
        try:
            # skip the node not in the region of board
            maze[p] = 1
        except:
            print("The path with wrong nodes is: {}".format(path))
    pin_max = int(max(state.start.keys()))
    for i in range(state.pairs_idx+1, pin_max+1):
        maze[maze>1]=1
        maze[starts[i]] = 0
        maze[finishs[i]] = 0
        astar = Astar(maze)
        path_t = astar.run(starts[i], finishs[i])
        if path_t is None:
            return True        
    return False

def prob_DFS(state, model, deterministic=False):
    
    DFS_state = deepcopy(state)
    pair_index = state.pairs_idx
    
    states_queue = [DFS_state]
    path = [DFS_state.head]
    
    while DFS_state.pairs_idx==pair_index:
        
        obs_vec = np.expand_dims(DFS_state.board_embedding(), axis=0)
        obs_vis = None
        mask = DFS_state.compute_mask()
        # print(mask, len(states_queue))
        if not all(mask==0):
            if deterministic:
                probs = model.p_all({"vec_obs": obs_vec, "vis_obs": obs_vis})[0]
                new_dist = probs.numpy()*mask
                # print(new_dist)
                if sum(new_dist)==0:
                    new_dist = mask
                new_dist = new_dist/sum(new_dist)
                action = np.argmax(new_dist)
            else:
                action, _, _ = model.get_action_logp_value({"vec_obs": obs_vec, "vis_obs": obs_vis}, mask=mask)

                # let's try random action
                # actions = DFS.state.getPossibleActions()

            DFS_state = DFS_state.takeAction(action)
            states_queue.append(DFS_state)
            path.append(DFS_state.head)
        elif len(states_queue)>1:
            pop_state = states_queue.pop()
            pop_state.board[pop_state.head] = 1
            DFS_state = states_queue[-1]
            DFS_state.board = copy(pop_state.board)
            # do not have to do this for now, but for later try of CNN
            DFS_state.board[DFS_state.head] = DFS_state.head_value
            path.pop()
        else:
            return None

    # if path[-1]!=state.finish[state.pairs_idx]:
    #     print("mask is: {}".format(mask))
    #     draw_board([], [], DFS_state.board, "DFS_vis.png")
    # for connecting multi-net
    path.pop()
    return path

def MCTS_search(env, model, fig_idx=0, board_ID='II4', rollout_times=50):

    from mcts import mcts

    state = deepcopy(env)
    state.reset()
    # print(state.finish)
    pin_indices = list(state.start.keys())
    pin_indices.sort()
    board = copy(state.board)

    reward_type = 'best'
    node_select = 'best'
    # rollout_times = 20

    routed_paths = []

    MCTS = mcts(iterationLimit=rollout_times, rolloutPolicy=mcts_DFS_rollout, nn_model=model,
            rewardType=reward_type, nodeSelect=node_select, explorationConstant=5)

    path_length = []
    nets_distance = []
    for pin_idx in pin_indices:
    # for pin_idx in [2]:
        pin_idx = int(pin_idx)

        state.reset(state.board, pin_idx)
        net_path = [state.start[pin_idx]]
        net_path += MCTS.search(initialState=state)

        routed_paths += net_path
        # checking if the path of this net block any other nets
        if block_other_nets(state, net_path):
            print("MCTS did not find a good path to connect net {}".format(pin_idx))
            # break

        nets_distance.append(distance.cityblock(state.start[pin_idx], state.finish[pin_idx]))
        if net_path[-1] == state.finish[pin_idx]:
            path_length.append(len(net_path))
        else:
            path_length.append(0)

        for node in net_path[1:]:
            action = tuple(np.array(node)-np.array(state.head))
            state = state.takeAction(action, is_tuple=True)

    paths_x, paths_y = separate_paths(env.start, routed_paths)

    board = env.original_board

    visual_folder_name = "visual_results_{}".format(rollout_times)
    if not os.path.isdir(visual_folder_name):
        os.mkdir(visual_folder_name)

    len_folder_name = "path_length_results_{}".format(rollout_times)
    if not os.path.isdir(len_folder_name):
        os.mkdir(len_folder_name)
    # saving path length and distance
    len_save_folder = os.path.join(len_folder_name, "length_{}.csv".format(board_ID))
    np.savetxt(len_save_folder, [path_length, nets_distance], delimiter=",")
    
    saved_fig_name = os.path.join(visual_folder_name, "board_{}.png".format(board_ID))

    draw_board(paths_x, paths_y, board, saved_fig_name)

    return routed_paths

def separate_paths(starts, routed_paths):
    
    ret_paths_x = []
    ret_paths_y = []
    single_path = []

    start_pins = list(starts.values())

    for v in routed_paths:
        if v in start_pins:
            ret_paths_x.append([node[0] for node in single_path])
            ret_paths_y.append([node[1] for node in single_path])
            single_path = [v]
        single_path.append(v)

    ret_paths_x.append([node[0] for node in single_path])
    ret_paths_y.append([node[1] for node in single_path])

    return ret_paths_x, ret_paths_y