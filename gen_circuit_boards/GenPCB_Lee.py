# This file aims at generate the PCB board with the placement for all nets
# The parameters to be considered includes Grid size (30*30), #Obstacle, #Nets 
import numpy as np
import random

import sys
import copy
from LeeAlgm import Field
from getFreePins_Lee import get_free_pin_pairs

class PCBLayout(object):
    """docstring for ClassName"""
    def __init__(self, num_chips, num_pairs):
        # define constants
        self.height = 30
        self.width = 30
        self.num_chips = num_chips
        self.board = np.zeros((self.height, self.width))
        self.board_path = np.zeros((self.height, self.width))
        self.num_pairs = num_pairs
        self.num_obstacles = 90

        # define PCB variables
        self.barriers = []
        self.chip_regions = []
        self.pin_pairs = []
        self.all_paths = []
 
    def run(self, shuffle):

        seg_regions = self.get_chip_regions()
        # print(seg_regions)

        chips = self.get_chips(seg_regions)
        # print(chips)

        self.region_no_chip(seg_regions, chips)

        # print(self.chip_regions)
        chips_around = self.get_chips_around(chips)

        self.around_to_barriers(chips_around)
        # print(barriers)
        pin_return = self.get_pin_pairs(chips_around)
        self.pin_pairs += pin_return

        if len(pin_return)<self.num_pairs:
            print(str(len(pin_return))+" out of "+str(self.num_pairs)+" are found!!")
            free_pin_pairs, self.board_path, paths = get_free_pin_pairs(self.board_path, 
                self.barriers, self.chip_regions, self.num_pairs-len(pin_return))
            
            self.pin_pairs += free_pin_pairs
            self.all_paths += paths

        self.gen_board(shuffle)

    def region_no_chip(self, seg_regions, chips):

        for r in range(len(seg_regions)):
            chip_region = []
            for x in range(seg_regions[r][0][0]+2, seg_regions[r][1][0]-1):
                for y in range(seg_regions[r][0][1]+2, seg_regions[r][1][1]-1):
                    if chips[r][0][0]<=x<=chips[r][1][0] and chips[r][0][1]<=y<=chips[r][1][1]:
                        continue
                    else:    
                        chip_region.append((x,y))
            self.chip_regions.append(chip_region)

    def get_chip_regions(self):
        board_x, board_y = self.board.shape
        board_x -= 1
        board_y -= 1
        mid_x = int(board_x/2)
        mid_y = int(board_y/2)

        region_1 = [(0,0), (mid_x, mid_y)]
        region_2 = [(0, mid_y), (mid_x, board_y)]
        region_3 = [(mid_x, mid_y), (board_x, board_y)]
        region_4 = [(mid_x, 0), (board_x, mid_y)]
        regions = [region_1, region_2, region_3, region_4]

        if self.num_chips==2:
            if random.randint(1,2)==1:
                return [[(0,0), (board_x, mid_y)], [(0, mid_x), (board_x, board_y)]]
            else:
                return [[(0,0), (mid_x, board_y)], [(mid_x, 0), (board_x, board_y)]]
            
        if self.num_chips==3:
            layout_idx = random.randint(0,3)
            next_idx = (layout_idx+1)%4
            new_region = self.combine_regions(regions[layout_idx], regions[next_idx])
            ret_regions = [new_region]
            for i in range(4):
                if i!=layout_idx and i!= next_idx:
                    ret_regions.append(regions[i])
            return ret_regions

        if self.num_chips==4:
            return regions

    def combine_regions(self, region1, region2):
        front_region = region2
        behind_region = region1
        if region1[0][0]<=region2[0][0] and region1[0][1]<=region2[0][1]:
            front_region = region1
            behind_region = region2

        new_region = [front_region[0], behind_region[1]]
        return new_region

    def get_chips(self, seg_regions):

        chips = []
        num_chips = len(seg_regions)
        for region in seg_regions:
            chip_size_x = random.randint(6-num_chips, 9-num_chips)
            chip_size_y = random.randint(6-num_chips, 9-num_chips)

            shrink_region_x = [region[0][0]+2, region[1][0]-2]
            shrink_region_y = [region[0][1]+2, region[1][1]-2]

            upleft_place_x_range = [shrink_region_x[0], shrink_region_x[1]-chip_size_x]
            upleft_place_y_range = [shrink_region_y[0], shrink_region_y[1]-chip_size_y]

            upleft_place_x = random.randint(upleft_place_x_range[0], upleft_place_x_range[1])
            upleft_place_y = random.randint(upleft_place_y_range[0], upleft_place_y_range[1])

            chips.append([(upleft_place_x, upleft_place_y), (upleft_place_x+chip_size_x, upleft_place_y+chip_size_y)])

        for chip in chips:
            for i in range(chip[0][0], chip[1][0]+1):
                for j in range(chip[0][1], chip[1][1]+1):
                    self.board_path[i][j] = 1
                    self.board[i][j] = 1
                    self.barriers.append((i,j))

        return chips

    def get_chips_around(self, chips):

        chips_around = []
        for chip in chips:
            chip_ard = []
            for x in range(chip[0][0], chip[1][0]+1):
                chip_ard.append((x, chip[0][1]-1))
            for y in range(chip[0][1], chip[1][1]+1):
                chip_ard.append((chip[1][0]+1, y))
            for x in range(chip[1][0], chip[0][0]-1, -1):
                chip_ard.append((x, chip[1][1]+1))
            for y in range(chip[1][1], chip[0][1]-1, -1):
                chip_ard.append((chip[0][0]-1, y))
            chips_around.append(chip_ard)

        # for reg in chips_around:
        #     for node in reg:
        #         self.board[node] = 1
        return chips_around

    def get_pin_pairs(self, chips_around):

        num_chips = len(chips_around)
        pin_pairs = []
        chip_pairs_idx = []
        for i in range(num_chips-1):
            for j in range(i+1, num_chips):
                chip_pairs_idx.append((i,j))

        chips_around_wrap = [[chip_ard] for chip_ard in chips_around]

        for n_p in range(self.num_pairs):

            random.shuffle(chip_pairs_idx)
            for chip_pair_idx in chip_pairs_idx:

                path_and_new_around = self.get_pin_from_chip(chip_pair_idx, chips_around_wrap)

                if path_and_new_around is None:
                    continue
                else:
                    chips_around_wrap = path_and_new_around[0]
                    self.all_paths.append(path_and_new_around[1])
                    pair = [path_and_new_around[1][0], path_and_new_around[1][-1]]
                    pin_pairs.append(pair)
                    for node in path_and_new_around[1]:
                        self.board_path[node] = 1
                        self.barriers.append(node)
                        for reg in self.chip_regions:
                            if node in reg:
                                reg.remove(node)
                    break

        return pin_pairs

    def get_pin_from_chip(self, chip_pair_idx, chips_around_wrap):

        chip_ards_fir = chips_around_wrap[chip_pair_idx[0]]
        chip_ards_sec = chips_around_wrap[chip_pair_idx[1]]

        for ard_seg1_idx in range(len(chip_ards_fir)):
            # get first pin
            ard_seg1 = chip_ards_fir[ard_seg1_idx]
            pin1_idx = random.randint(0, len(ard_seg1)-1)
            pin1 = ard_seg1[pin1_idx]
            for ard_seg2_idx in range(len(chip_ards_sec)):
                # get second pin
                ard_seg2 = chip_ards_sec[ard_seg2_idx]
                pin2_idx = random.randint(0, len(ard_seg2)-1)
                pin2 = ard_seg2[pin2_idx]

                tem_barriers = copy.copy(self.barriers)
                tem_barriers.remove(pin1)
                tem_barriers.remove(pin2)
                try:
                    field = Field(len=30, start=pin1, finish=pin2, barriers=tem_barriers)
                    field.emit()
                    path = list(field.get_path())
                    # print(path)

                except:
                    path = None
                    # print("No path for pins attached to the chip")

                if path is not None:
                    chip_ards_fir.pop(ard_seg1_idx)
                    chip_ards_sec.pop(ard_seg2_idx)
                    new_segs1 = self.split_segment(ard_seg1, pin1_idx)
                    new_segs2 = self.split_segment(ard_seg2, pin2_idx)
                    chip_ards_fir += new_segs1
                    chip_ards_sec += new_segs2
                    chips_around_wrap[chip_pair_idx[0]] = chip_ards_fir
                    chips_around_wrap[chip_pair_idx[1]] = chip_ards_sec
                    return [chips_around_wrap, path]

        return None

    def around_to_barriers(self, chips_around):

        for ard in chips_around:
            self.barriers += ard

    def split_segment(self, segment, remove_idx):

        seg1 = segment[:remove_idx]
        seg2 = segment[remove_idx+1:]

        if abs(segment[0][0]-segment[-1][0])==1 and abs(segment[0][1]-segment[-1][1])==1:
            seg1.reverse()
            seg2.reverse()
            return [seg1+seg2]

        if len(seg1)!=0 and len(seg2)!=0:
            return [seg1, seg2]
        elif len(seg1)!=0 and len(seg2)==0:
            return [seg1]
        elif len(seg1)==0 and len(seg2)!=0:
            return [seg2]

        return []

    def gen_board(self, shuffle=False):

        # # adding obstacles
        # board_size = self.board.shape[0]*self.board.shape[1]
        # for i in range(self.board.shape[0]):
        #     for j in range(self.board.shape[1]):
        #         if self.board_path[i,j] == 0 and random.randint(1, board_size)<=self.num_obstacles:
        #             self.board[i,j] = 1
        #             self.board_path[i,j] = 1

        # rearrange pin indices
        if shuffle:
            random.shuffle(self.pin_pairs)
        idx = 2 
        for pp in self.pin_pairs:
            self.board[pp[0]] = idx
            self.board[pp[1]] = -idx
            self.board_path[pp[0]] = idx
            self.board_path[pp[1]] = -idx
            idx += 1

# if __name__== "__main__":

#     height = 30
#     width = 30
#     num_obstacles = 60

#     num_boards = int(sys.argv[1])
#     saved_path = sys.argv[2]

#     for i in range(num_boards):

#         num_chips = random.randint(2,4)
#         num_pairs = 10 # manually added to the csv file

#         board_gen = PCBLayout(num_chips, num_pairs)

#         board_gen.run()

#         np.savetxt(saved_path+'/board'+str(i)+'.csv', board_gen.board, delimiter=',')

#         np.savetxt(saved_path+'/board_path'+str(i)+'.csv', board_gen.board_path, delimiter=',')