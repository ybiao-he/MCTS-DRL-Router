from astar import Astar
import os
import numpy as np

def read_board(file_name):

    board = np.genfromtxt(file_name, delimiter=',')

    # parse the board and get the starts and ends
    start = {}
    finish = {}
    for i in range(board.shape[0]):
        for j in range(board.shape[1]):
            if abs(board[i,j])>=2:
                if board[i,j]<0:
                    finish[-board[i,j]] = (i,j)
                    board[i,j] = abs(board[i,j])
                else:
                    start[board[i,j]] = (i,j)
    return board, start, finish

# rename board names in one folder
def rename_boards(directory, category):

    import re
    boards_name = os.listdir(directory)
    boards_ids = []
    for name in boards_name:
        idx = int(re.findall(r'\d+', name)[0])
        boards_ids.append(idx)

    boards_ids = sorted(boards_ids)
    # print(boards_ids)
    for new_i, i in enumerate(boards_ids):
        board_name = os.path.join(directory, "board{}.csv".format(i))
        new_board_name = os.path.join(directory, "board_{}{}.csv".format(category, new_i))
        os.rename(board_name, new_board_name)


# separate boards into two categories: A* can connect and A* can't connect
# This is the folder of all boards and also the folder to store first category of boards
board_folder = '/home/ybiao/workspace/RL_routing/boards_I'
# This is the folder to store seconds category of boards
boards_II_folder = "/home/ybiao/workspace/RL_routing/boards_II"
if not os.path.isdir(boards_II_folder):
    os.mkdir(boards_II_folder)

for board_idx in range(1000):
    board_path = os.path.join(board_folder,"board{}.csv".format(board_idx))
    board_backup, starts, finishes = read_board(board_path)

    for net_idx in range(2, int(max(starts))+1):
        board = np.copy(board_backup)
        board[board==net_idx]=0
        board[board>=1]=1

        start = starts[net_idx]
        end = finishes[net_idx]

        astar = Astar(board)
        result = astar.run(start, end)

        # print(result)
        if result is None:
            print("Board{} cannot be routed by A star".format(board_idx))
            os.rename(board_path, os.path.join(boards_II_folder, 'board{}.csv'.format(board_idx)))
            break
        else:
            for node in result:
                board_backup[tuple(node)]=1

# rename board names for each category
rename_boards(board_folder, 'I')
rename_boards(boards_II_folder, "II")