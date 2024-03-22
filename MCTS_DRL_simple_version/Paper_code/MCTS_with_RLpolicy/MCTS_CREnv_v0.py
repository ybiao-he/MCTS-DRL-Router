from __future__ import division

from copy import copy
from copy import deepcopy
from scipy.spatial import distance

import numpy as np
import os, random

from astar import Astar

####    Environment for MCTS    ####

class MCTS_CREnv():
    def __init__(self, board_path="./board.csv"):

        self.directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.board_path = board_path
        self.state_shape = (4,)
        self.n_actions = 4

        self.board = np.genfromtxt(self.board_path, delimiter=',')
        
        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if abs(self.board[i,j])>=2:
                    if self.board[i,j]<0:
                        self.finish[-int(self.board[i,j])] = (i,j)
                        self.board[i,j] = abs(self.board[i,j])
                    else:
                        self.start[int(self.board[i,j])] = (i,j)
        self.original_board = copy(self.board)

        # compute and store the shortest for each single net
        self.short_net_path = dict()
        for net_id in self.start.keys():
            path_t = self.find_shortest_path(net_id)
            self.short_net_path[net_id] = path_t

    def reset(self, board=None, pin_idx=2):

        if board is None:
            self.board = np.genfromtxt(self.board_path, delimiter=',')
        else:
            self.board = board

        self.head_value = 20
        self.path_length = 0
        self.connection = False
        self.collide = False
        self.terminal = False
        # initialize the action node
        self.pairs_idx = pin_idx
        self.max_pair = self.pairs_idx
        # self.max_pair = max(self.start.keys())
        self.head = self.start[self.pairs_idx]

        self.board[self.head] = self.head_value

        self.total_reward = 0

    def takeAction(self, action, is_tuple=False):

        newState = deepcopy(self)

        if is_tuple:
            action_tmp = action
        else:
            action_tmp = newState.directions[action]
        newState.connection = False
        newState.collide = False
        newState.pre_head = newState.head

        newState.path_length += 1

        # pre-determine new action node
        newState.head = (newState.head[0]+action_tmp[0], newState.head[1]+action_tmp[1])
        # check/adjust new action node and set its value
        x = newState.head[0]
        y = newState.head[1]

        if 0 <= x < newState.board.shape[0] and 0 <= y < newState.board.shape[1]:
            if newState.head == newState.finish[newState.pairs_idx]:
                newState.goto_new_net(True)
            elif newState.board[newState.head]!=0:
                newState.collide = True
                newState.goto_new_net(False)
            else:
                newState.board[newState.pre_head] = 1
                newState.board[newState.head] = newState.head_value
        else:
            newState.collide = True
            newState.goto_new_net(False, out_range=True)

        newState.total_reward += newState.step_reward()

        return newState

    def goto_new_net(self, connection_sign, out_range=False):

        self.board[self.pre_head] = 1
        self.connection = connection_sign
        self.pairs_idx += 1
        if not out_range:
            self.board[self.head] = 1
            self.pre_head = self.head
        if self.pairs_idx<=self.max_pair:
            self.head = self.start[self.pairs_idx]
            self.board[self.head] = self.head_value

    def isTerminal(self):

        if self.pairs_idx>self.max_pair:
            self.terminal = True
            return True

        return False

    def getReward(self):

        return self.total_reward

    def step_reward(self):

        # compute reward from other nets
        if self.connection or self.collide:
            # blocking other nets
            r_2 = self.reward_other_nets()
            # print("reward_other_nets is {} {}".format(r_2, self.total_reward))
            return r_2


        return -0.1

    def check_direction(self, direction):

        x = self.head[0] + direction[0]
        y = self.head[1] + direction[1]
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if (x,y) == self.finish[self.pairs_idx]:
                return 2
            elif self.board[(x,y)] == 0:
                return 1
        return 0

    def compute_mask(self):

        mask = np.zeros(self.n_actions)
        for act_i in range(self.n_actions):
            act_d = self.directions[act_i]
            check_sign = self.check_direction(act_d)
            if check_sign==2:
                mask = np.zeros(self.n_actions)
                mask[act_i] = 1
                return mask
            elif check_sign==1:
                mask[act_i] = 1     

        return mask

    def getPossibleActions(self):

        possible_actions = []
        for act_d in self.directions:
            check_sign = self.check_direction(act_d)
            if check_sign==2:
                return [act_d]
            elif check_sign==1:
                possible_actions.append(act_d)
        return possible_actions

    def board_embedding(self):

        if self.pairs_idx<=self.max_pair:
            dist_to_target = [i-j for i, j in zip(self.head, self.finish[self.pairs_idx])]
        else:
            dist_to_target = [i-j for i, j in zip(self.head, self.finish[self.pairs_idx-1])]
        # state = np.array(list(self.head)+list(self.finish[self.pairs_idx]))
        state = np.array(list(self.head)+dist_to_target)
        # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)

        return state

    def reward_other_nets(self):

        path_diff = 0
        pin_max = int(max(self.start.keys()))
        for i in range(self.pairs_idx, pin_max+1):
            path_t = self.find_shortest_path(i)
            if path_t is None:
                return -20
            # if len(path_t)<len(self.short_net_path[i]):
            #     print(i, path_t, self.short_net_path[i])
            path_diff += (len(path_t)-len(self.short_net_path[i]))
        return -path_diff/10
    
    def find_shortest_path(self, net_id):

        s_node = self.start[net_id]
        t_node = self.finish[net_id]
        maze = copy(self.board)
        maze[maze>1] = 1
        maze[s_node] = 0
        maze[t_node] = 0
        astar = Astar(maze)
        path = astar.run(s_node, t_node)
        return path