#!/bin/bash

mkdir output

for i in {0..53}
do
	nohup python3 main_mcts.py /proj/cloudincr-PG0/ppo/simple_boards_II/ II$i /proj/cloudincr-PG0/ppo/PPO_running_boards_II/ > output/output$i.out 2>&1 &
done
