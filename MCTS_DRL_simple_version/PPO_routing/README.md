# circuit routing using PPO

To train RL for a board, just run `python3 main.py directory_to_board board_name`. For example, to train for board `board_II4`, the one in currecnt forlder, please run `python3 main.py ./ board_II4.csv`.

The learned policy will be saved in the folder "saved_model" after training. (will be used as the rollout policy of MCTS)