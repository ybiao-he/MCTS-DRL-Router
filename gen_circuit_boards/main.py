import numpy as np
import random

import sys
from multiprocessing import Process, Manager
from GenPCB_Lee import PCBLayout
import multiprocessing as mp

def abs_pin(board):
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            board[i,j] = abs(board[i,j])
    return board

def gen_train_boards(board, paths):

    board = abs_pin(board)

    direction_map = {(-1, 0): 0, (0, 1): 1, (1, 0): 2, (0, -1): 3}
    saved_data = []
    for path in paths:

        path_lenght = len(path)
        smpl_idx = random.randint(1, path_lenght-2)
        path_idx = board[path[0]]

        for i in range(path_lenght):
            if i==smpl_idx:
                board[path[i]] = path_idx
                tem_board = np.ravel(board)
                action = (path[i+1][0]-path[i][0], path[i+1][1]-path[i][1])
                action_idx = direction_map[action]
                tem_board = np.append(tem_board, [action_idx])
                saved_data.append(tem_board)
                # np.savetxt(saved_name+str(path_idx)+'.csv', board_gen.board, delimiter=',')
                board[path[i]] = 1
            else:
                board[path[i]] = 1

    return saved_data

def multiprocess_worker(dataset, iters):

    for i in range(iters):

        num_chips = random.randint(2,4)
        num_pairs = random.randint(5,10) # manually added to the csv file

        board_gen = PCBLayout(num_chips, num_pairs)

        board_gen.run(False)

        training_board = gen_train_boards(board_gen.board, board_gen.all_paths)

        dataset += training_board

def multiprocess_run(num_p, iters):

    dataset = []
    with Manager() as manager:
        tem_dataset = manager.list()
        processes = []
        # Run processes
        for i in range(num_p):
            p = Process(target=multiprocess_worker, args=(tem_dataset, iters))
            p.start()
            processes.append(p)

        # Exit the completed processes
        for p in processes:
            p.join()

        # dataset += [q.get() for p in processes]
        dataset = [x for x in tem_dataset]
        
    return dataset

def single_process_worker(num_boards, saved_path, gen_raw_board):

    dataset = []

    for i in range(num_boards):

        num_chips = random.randint(2,4)
        num_pairs = random.randint(5,10) # manually added to the csv file

        board_gen = PCBLayout(num_chips, num_pairs)

        board_gen.run(False)

        if gen_raw_board:
            np.savetxt(saved_path+'/board'+str(i)+'.csv', board_gen.board, delimiter=',')
        else:
            # saved_name_first_part = saved_path+'/board'+str(i)+'_path'
            training_board = gen_train_boards(board_gen.board, board_gen.all_paths)
            dataset += training_board

    if not gen_raw_board:
        dataset = np.array(dataset)
        np.savetxt(saved_path+'/training_data.csv', dataset, delimiter=',')

        # np.savetxt(saved_path+'/board_path'+str(i)+'.csv', board_gen.board_path, delimiter=',')

if __name__== "__main__":

    num_boards = int(sys.argv[1])
    saved_path = sys.argv[2]
    gen_raw_board = eval(sys.argv[3])

    import os
    if not os.path.isdir(saved_path):
    	os.mkdir(saved_path)

    if gen_raw_board:

        single_process_worker(num_boards, saved_path, gen_raw_board)

    else:

        num_p = mp.cpu_count()
        np_remainder = num_boards%num_p
        iters = num_boards//num_p

        dataset = []
        # for i in range(iters):

        dataset += multiprocess_run(num_p, iters)

        if np_remainder!=0:

            dataset += multiprocess_run(np_remainder, 1)

        print(len(dataset))

        dataset = np.array(dataset)
        np.savetxt(saved_path+'/training_data.csv', dataset, delimiter=',')
