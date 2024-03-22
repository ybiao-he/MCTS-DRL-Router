import numpy as np
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

def paths_to_xys(paths):
    paths_x = []
    paths_y = []
    for p in paths:
        xs = [node[0] for node in p]
        ys = [node[1] for node in p]
        paths_x.append(xs)
        paths_y.append(ys)
    
    return paths_x, paths_y


def prune_paths(paths_array, state):

    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    # pruning path considering 3 directions
    length = len(paths_array)

    i = 1
    while i < length-1:
        current_node = paths_array[i]
        # checking current node if it is a turning node
        pre_direction = current_node - paths_array[i-1]
        post_direction = paths_array[i+1] - current_node
        if tuple(pre_direction)==tuple(post_direction):
            check_directions = [np.array(d) for d in directions if 0 not in np.array(d)+pre_direction]
        else:
            check_directions = [pre_direction, -post_direction]

        for cd in check_directions:
            paths_array = examine_one_direction(paths_array, i, cd, state)
        
        i += 1
        length = len(paths_array)

    # prune for the starting node of the path
    path_direction = paths_array[1] - paths_array[0]
    for cd in directions:
        if tuple(path_direction)!=cd:
            paths_array = examine_one_direction(paths_array, 0, cd, state)

    return paths_array


def examine_one_direction(path, node_id, checkD, env_state):

    node = path[node_id]
    board2D = env_state.board
    target_node = env_state.finish[env_state.pairs_idx]
    node = np.array(node)+np.array(checkD)
    board_shape = np.array(board2D.shape)
    explore_nodes = []
    while np.array(node<board_shape).all() and np.array(node>=0).all():
        if board2D[tuple(node.astype(int))]==0 or tuple(node)==target_node:
            check_node_in = [np.array_equal(node,pn) for pn in path[node_id:]]
            if True in check_node_in:
                current_idx = check_node_in.index(True)
                circle = path[node_id:node_id+current_idx+1] + explore_nodes
                if not check_net_in_circle(circle, env_state):
                    path = path[:node_id+1]+explore_nodes+path[node_id+current_idx:]
                # path = path[:node_id+1]+explore_nodes+path[node_id+current_idx:]
                break
            else:
                explore_nodes.append(node)
                node = np.array(node)+np.array(checkD)
        else:
            break
    return path


def check_net_in_circle(circle, env_state, check_node=None):

    from copy import copy
    board = copy(env_state.board)

    # find the max and min value of the region that the circle covers
    x_values = [node_tem[0] for node_tem in circle]
    y_values = [node_tem[1] for node_tem in circle]
    max_x = int(max(x_values))
    min_x = int(min(x_values))
    max_y = int(max(y_values))
    min_y = int(min(y_values))

    # find all the obstacles and net pins in the region
    obstacles_sub = []
    pins_sub = []
    for i in range(min_x, max_x+1):
        for j in range(min_y, max_y+1):
            if board[i,j] == 1:
                obstacles_sub.append(np.array([i,j]))
            elif board[i,j] > env_state.pairs_idx and (i,j)!=env_state.head:
                pins_sub.append(np.array([i,j]))

    # checking if pins in the region
    tem_board = np.zeros(board.shape)
    for node in circle:
        tem_board[tuple(node.astype(int))]=1

    outsider = None
    circle_tuple = [tuple(n) for n in circle]
    for corner in [(0,0), (0,board.shape[1]-1), (board.shape[0]-1,0), (board.shape[0]-1,board.shape[1]-1)]:
        if corner not in circle_tuple:
            outsider = corner
            break
    if outsider is None:
        return True

    astar = Astar(tem_board)
    # for node in check_node:
    for node in pins_sub:
        result = astar.run(outsider, tuple(node))
        if result is None:
            return True
    return False

# read board and paths

from MCTS_CREnv_v0 import MCTS_CREnv
import numpy as np

board_file = "./board0.csv"
path_file = "board0_path.csv"

env = MCTS_CREnv(board_path=board_file)
env.reset(pin_idx=4)

paths = np.genfromtxt(path_file, delimiter=',')
sep_paths = []
path_tem = []
for p in paths:
    if -1 in p:
        sep_paths.append(path_tem)
        path_tem = []
    else:
        path_tem.append(p)

# circle = np.array([[5,20], [4,20], [3,20], [3,21], [3,22], [4,22], [5,22], [5,21]])
# insider = np.array([4,25])
# print(check_net_in_circle(circle, env, [insider]))

new_path = prune_paths(sep_paths[2], env)
# new_path = sep_paths[2]
# print(new_path)
paths_x, paths_y = paths_to_xys([new_path])
# board = np.absolute(np.genfromtxt("./board_II4.csv", delimiter=','))
board = env.board

draw_board(paths_x, paths_y, board, "test.jpg")
