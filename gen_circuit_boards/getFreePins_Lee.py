# This file ims at generate the PCB board with the placement for all nets
# The parameters to be considered includes Grid size (30*20), #Obstacle, #Nets 
import numpy as np
import random

import sys
import copy
from LeeAlgm import Field

def get_free_pin_pairs(board, barriers, regions_nodes, num_pairs):

	# get the indices of two regions where two pains are connected
    num_regions = len(regions_nodes)
    pin_pairs = []
    region_pairs_idx = []
    paths = []
    for i in range(num_regions-1):
        for j in range(i+1, num_regions):
            region_pairs_idx.append((i,j))

    for n_p in range(num_pairs):

        random.shuffle(region_pairs_idx)
        for region_pair_idx in region_pairs_idx:

            path = get_pin_from_regions(barriers, region_pair_idx, regions_nodes)

            if path is None:
                continue
            else:
                paths.append(path)
                pair = [path[0], path[-1]]
                pin_pairs.append(pair)
                for node in path:
                    board[node] = 1
                    barriers.append(node)
                    for reg in regions_nodes:
                        if node in reg:
                            reg.remove(node)
                break

    return pin_pairs, board, paths

def get_pin_from_regions(barriers, region_pair_idx, regions_nodes):

    region_fir = copy.copy(regions_nodes[region_pair_idx[0]])
    random.shuffle(region_fir)

    region_sec = copy.copy(regions_nodes[region_pair_idx[1]])
    random.shuffle(region_sec)

    for pin1 in region_fir:
        for pin2 in region_sec:
            try:
                field = Field(len=30, start=pin1, finish=pin2, barriers=barriers)
                field.emit()
                path = list(field.get_path())
                # print(path)

            except:
                path = None
                # print("No path for free pin pairs")

            if path is not None:
                return path

    return None
