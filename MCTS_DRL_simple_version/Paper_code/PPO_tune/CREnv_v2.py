# The env in this script has 3 candudate actions: straight, clockwise, counter-clockwise
# and it also allows 90-degree bend
from __future__ import division

from copy import copy
from copy import deepcopy
from scipy.spatial import distance

import numpy as np
import gym
from gym import spaces
import os, random


class CREnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is an env for circuit routing. 
    """
    def __init__(self, board_path="./board.csv"):
        super(CREnv, self).__init__()

        self.all_directions = [(-1,0), (0,1), (1,0), (0,-1)]

        self.state_shape = (6,)

        self.board_path = board_path
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=30, shape=self.state_shape, dtype=np.float32)

    def reset(self):

        self.board = np.genfromtxt(self.board_path, delimiter=',')

        self.head_value = 20
        
        self.path_length = 0
        self.connection = False
        self.collide = False

        # parse the board and get the starts and ends
        self.start = {}
        self.finish = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if abs(self.board[i,j])>=2:
                    if self.board[i,j]<0:
                        self.finish[-self.board[i,j]] = (i,j)
                        self.board[i,j] = abs(self.board[i,j])
                    else:
                        self.start[self.board[i,j]] = (i,j)

        # initialize the head node
        self.max_pair = int(np.amax(self.board))
        self.pairs_idx = int(min(self.start.keys()))
        # self.pairs_idx = 3
        self.head = self.start[self.pairs_idx]

        self.pre_head = self.find_ini_prehead()

        self.last_node = self.head

        self.board[self.head] = self.head_value

        state = self.board_embedding()
        return state

    def step(self, action):

        action_tmp = self.get_directions_from_action(action)
        self.connection = False
        self.collide = False
        self.pre_head = self.head

        self.path_length += 1

        # pre-determine new action node
        self.head = (self.head[0]+action_tmp[0], self.head[1]+action_tmp[1])
        # check/adjust new action node and set its value
        x = self.head[0]
        y = self.head[1]

        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.head == self.finish[self.pairs_idx]:
                self.goto_new_net(True)
            elif self.board[self.head]!=0:
                self.collide = True
                self.goto_new_net(False)
            else:
                self.board[self.pre_head] = 1
                self.board[self.head] = self.head_value
        else:
            self.collide = True
            self.goto_new_net(False, out_range=True)

        reward = self.getReward()

        state = self.board_embedding()

        done = self.isTerminal()

        info = {}

        return state, reward, done, info

    def get_directions_from_action(self, act_idx):

        path_d = np.array(self.head)-np.array(self.pre_head)

        d_idx = (self.all_directions.index(tuple(path_d))+act_idx-1)%len(self.all_directions)

        return self.all_directions[d_idx]

    def find_ini_prehead(self):

        possible_ds = []
        for d in self.all_directions:
            if self.check_direction(d)>0:
                tem_target = (self.head[0]+d[0], self.head[1]+d[1])
                possible_ds.append(d)

        if len(possible_ds)>0:
            best_d = random.choice(possible_ds)
        else:
            best_d = random.choice(self.all_directions)

        return (self.head[0]-best_d[0], self.head[1]-best_d[1])

    def goto_new_net(self, connection_sign, out_range=False):

        self.board[self.pre_head] = 1
        self.connection = connection_sign
        self.pairs_idx += 1
        self.last_node = self.head
        if not out_range:
            self.board[self.head] = 1
        if self.pairs_idx<=self.max_pair:
            self.head = self.start[self.pairs_idx]
            self.pre_head = self.find_ini_prehead()
            self.board[self.head] = self.head_value

    def isTerminal(self):

        if self.pairs_idx>self.max_pair:
            return True

        return False

    def getReward(self):

        if self.connection:
            return 20
        if self.collide:
            left_dist = 5*np.linalg.norm(np.array(self.last_node) - np.array(self.finish[self.pairs_idx-1]))
            # distance.cityblock(self.last_node, self.finish[self.pairs_idx-1])
            return -left_dist/10

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

    def board_embedding(self):

        if self.pairs_idx<=self.max_pair:
            dist_to_target = [i-j for i, j in zip(self.head, self.finish[self.pairs_idx])]
        else:
            dist_to_target = [i-j for i, j in zip(self.head, self.finish[self.pairs_idx-1])]
        # state = np.array(list(self.action_node)+list(self.finish[self.pairs_idx]))
        state = np.array(list(self.head)+dist_to_target+list(self.pre_head))
        # state = np.concatenate(( state, np.array(nets_vector.tolist()+obs_vector.tolist()) ), axis=0)

        return state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()

        print("------")
        print(self.action_node)

    def close(self):
        pass