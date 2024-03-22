from __future__ import division

from copy import copy
from copy import deepcopy
from scipy.spatial import distance
import itertools

import functools
import numpy as np
import gym
from gym import spaces
import os, random

import matplotlib.pyplot as plt

class CREnv(gym.Env):
    """
        The env is for general routing problem on the generated circuits, 
        it has 3 candudate actions: straight, 90-degree clockwise and 90-degree counter-clockwise and allows 90-degree bend
    """

    def __init__(self, board_path="./board.csv"):
        super(CREnv, self).__init__()

        self.all_directions = [(1,0), (0,1), (-1,0), (0,-1)]

        self.state_shape = (6,)

        self.board_path = board_path
        n_actions = 3
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Box(low=0, high=30, shape=self.state_shape, dtype=np.float32)

    def reset(self):

        self.board = np.genfromtxt(self.board_path, delimiter=',')

        self.path_length = 0

        self.connection = False
        self.collide = False

        self.obstacles = []

        # parse the board and get pins of each net
        self.nets = {}
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if abs(self.board[i,j])>=2:
                    net_idx = abs(self.board[i,j])
                    if net_idx in self.nets:
                        self.nets[net_idx].append((i,j))
                    else:
                        self.nets[net_idx] = [(i,j)]
                elif self.board[i,j]==1:
                    self.obstacles.append((i,j))

        # initialize the action node and paths (empty)
        self.paths45 = dict()
        self.pairs_idx = int(min(self.nets.keys()))
        self.max_pair = int(max(self.nets.keys()))

        self.current_net = copy(self.nets[self.pairs_idx])
        self.other_nets = {key: value for key, value in self.nets.items() if key > self.pairs_idx}
        self.head = self.current_net[0]
        self.current_net.remove(self.head)
        
        self.current_path = [self.head]
        self.paths = {self.pairs_idx:[]}

        self.targets = self.find_targets()

        self.pre_head = self.find_ini_prehead()
        self.last_node = self.head

        state = self.extract_circuit_info()

        return state

    def find_targets(self):

        if len(self.paths[self.pairs_idx])!=0:
            return functools.reduce(lambda a, b: a+b, self.paths[self.pairs_idx])
        
        return copy(self.current_net)


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

        mid_node = (np.array(self.head)+np.array(self.pre_head))/2
        mid_node = tuple(mid_node)
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if self.paths45.get(mid_node):
                self.collide = True
                self.goto_new_net(False)
            else:
                if self.head in self.targets:
                    self.current_path.append(self.head)
                    if self.head in self.current_net:
                        self.current_net.remove(self.head)
                    self.goto_new_net(True)
                    self.paths45[mid_node] = True
                elif self.board[self.head]!=0:
                    self.collide = True
                    self.goto_new_net(False)
                else:
                    self.current_path.append(self.head)
                    self.board[self.pre_head] = 1
                    self.board[self.head] = 1
                    self.paths45[mid_node] = True
        else:
            self.collide = True
            self.goto_new_net(False, out_range=True)

        reward = self.getReward()
        # print(reward)

        state = self.extract_circuit_info()

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

        self.paths[self.pairs_idx].append(self.current_path)

        self.board[self.pre_head] = 1
        self.connection = connection_sign
        self.last_node = self.head

        # if not out_range:
        #     self.board[self.head] = 1


        if len(self.current_net)>0:
            self.head = self.current_net[0]
            self.board[self.head] = 1

            self.pre_head = self.find_ini_prehead()
            self.current_net.remove(self.head)

            self.current_path = [self.head]
            self.targets = self.find_targets()

        elif self.pairs_idx<self.max_pair:
            self.pairs_idx += 1
            self.current_net = copy(self.nets[self.pairs_idx])
            self.other_nets = {key: value for key, value in self.nets.items() if key > self.pairs_idx}
            self.head = self.current_net[0]

            self.paths[self.pairs_idx] = []
            self.board[self.head] = 1

            self.pre_head = self.find_ini_prehead()
            self.current_net.remove(self.head)

            self.current_path = [self.head]

            self.targets = self.find_targets()

        else:
            self.pairs_idx += 1
            # self.head=self.pre_head


    def isTerminal(self):

        if self.pairs_idx > self.max_pair:
            return True

        return False


    def getReward(self):

        if self.connection:
            return 10
        if self.collide:
            tem_head = np.array(self.head)
            min_euclid_distance = self.board.shape[0]*self.board.shape[1]
            for t in self.targets:
                tem_t = np.array(t)
                min_euclid_distance = min(min_euclid_distance, np.linalg.norm(tem_t - tem_head))
            return -min_euclid_distance/10

        # expand_length = np.linalg.norm(np.array(self.head) - np.array(self.pre_head)) 
        # return -expand_length/10
        return -0.1

    def check_direction(self, direction):

        x = self.head[0] + direction[0]
        y = self.head[1] + direction[1]
        mid_node = np.array(self.head)+np.array(direction)/2
        mid_node = tuple(mid_node)
        if 0 <= x < self.board.shape[0] and 0 <= y < self.board.shape[1]:
            if not self.paths45.get(mid_node):
                if (x,y) in self.targets:
                    return 2
                elif self.board[(x,y)] == 0:
                    return 1
        return 0

    def extract_circuit_info(self):

        state = []

        state.append(np.array(self.get_current_net(), dtype= np.float32))
        # state.append(np.array(self.nets_to_edgeFeature(), dtype= np.float32))
        # state.append(np.array(self.paths_to_edgeFeature(), dtype= np.float32))
        # state.append(np.array(self.obstacles, dtype= np.float32))

        return state
        # return np.array(self.get_current_net(), dtype= np.float32)

    def get_current_net(self):

        target_embed = np.zeros((2,))
        tem_head = np.array(self.head)
        sum_euclid_distance = 0
        for t in self.targets:
            tem_t = np.array(t)
            sum_euclid_distance += 1/np.linalg.norm(tem_t - tem_head)
        for t in self.targets:
            tem_t = np.array(t)
            ratio_dist = 1/np.linalg.norm(tem_t - tem_head)/sum_euclid_distance
            target_embed += tem_t*ratio_dist
        current_net_vector = np.array(list(self.head)+list(self.pre_head)+list(target_embed))

        return current_net_vector

    def nets_to_edgeFeature(self):

        nets = copy(self.other_nets)

        if len(nets)==0:
            return [[0,0,0,0]]

        min_idx = int(min(nets.keys()))
        max_idx = int(max(nets.keys()))

        features = []

        for i in range(min_idx, max_idx+1):
            net = nets[i]
            edges = list(itertools.combinations(net, 2))[0]
            edges = list(edges[0] + edges[1])
            features.append(edges)
            
        return features

    def paths_to_edgeFeature(self):

        min_pair_idx = int(min(self.nets.keys()))
        # print(self.paths)
        
        if len(self.paths[min_pair_idx])==0 and len(self.current_path)<2:
            return [[0,0,0,0]]

        path_edges = [a + b for a, b in zip(self.current_path[:-1], self.current_path[1:])]

        upper_index = min(self.pairs_idx, self.max_pair)
        for i in range(min_pair_idx, upper_index+1):
            for j in range(len(self.paths[i])):
                path_len = len(self.paths[i][j])
                if len(self.paths[i][j])>1:
                    path_edges += [a + b for a, b in zip(self.paths[i][j][:-1], self.paths[i][j][1:])]

        return path_edges


    def render(self):

        width, height = self.board.shape
        if self.board.shape==(30,30):
            self.fig = plt.figure(figsize=[8, 8])
        else:
        # create a 8" x 8" board
            self.fig = plt.figure(figsize=[width/4, height/4])

        save_name = "test_env_output/routing_step_{}.jpg".format(self.path_length)

        paths = functools.reduce(lambda a, b: a+b, self.paths.values())
        paths.append(self.current_path)

        paths_x = [[cood[0] for cood in pth] for pth in paths]
        paths_y = [[cood[1] for cood in pth] for pth in paths]

        board = np.absolute(np.genfromtxt(self.board_path, delimiter=','))
        # print(paths_x)

        width, height = board.shape

        ax = self.fig.add_subplot(111)

        # draw the grid
        for x in range(width):
            ax.plot([x, x], [0,height-1], color=(0.5,0.5,0.5,1))
        for y in range(height):
            ax.plot([0, width-1], [y,y], color=(0.5,0.5,0.5,1))

        # draw paths
        for p in range(len(paths_x)):

            ph = plt.subplot()
            ph.plot(paths_x[p], paths_y[p], linewidth=5, color='black')

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
        # print(nets)
        ax.scatter(y_axis, x_axis, marker='s', s=250, c='k')

        for xy in nets:
            ax.text(xy[0], xy[1], str(int(nets[xy])-1), fontsize=18, color='w',
                    horizontalalignment='center', verticalalignment='center')

        # scale the axis area to fill the whole figure
        ax.set_position([0,0,1,1])

        # get rid of axes and everything (the figure background will show through)
        ax.set_axis_off()

        # scale the plot area conveniently (the board is in 0,0..18,18)
        ax.set_xlim(0,width-1)
        ax.set_ylim(0,height-1)
        
        if self.isTerminal():
            self.fig.savefig(save_name)
        
        plt.draw()
        plt.pause(0.0001)
        plt.clf()