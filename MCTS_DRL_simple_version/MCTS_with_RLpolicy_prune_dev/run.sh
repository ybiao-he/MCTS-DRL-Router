#!/bin/bash
while IFS=, read -r name; do

server=Ybiao@amd251.utah.cloudlab.us
name="${name%%[[:cntrl:]]}"

ssh -tt -p 22 $server << EOF
cd /proj/cloudincr-PG0/mcts/MCTS_with_DRLPolicy_env0/
nohup python3 main_mcts.py /proj/cloudincr-PG0/ppo/simple_boards/ $name /proj/cloudincr-PG0/ppo/PPO_running_boards_ent_01/ > output/output-50$name.out 2>&1 &
exit
EOF

done < select_boards_ids.csv