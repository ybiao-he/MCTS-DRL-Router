#!/bin/bash
# head -n50 select_boards_ids.csv 
while IFS=, read -r name server; do

server="${server%%[[:cntrl:]]}"
name="${name%%[[:cntrl:]]}"

# tune network structue, layes and neurons
# for n in 2 3 4
# do
# for i in 16 32 64
# do
# ssh -tt -p 22 $server << EOF
# cd /proj/cloudincr-PG0/ppo/PPO_tune_all_boards/
# nohup python3 main.py nn-$n-$i $name > output_nn-$n-$i.out 2>&1 &
# exit
# EOF
# done
# done

# tune entropy coefficient
# for ((i = 0; i < 5; i++))
# do
# ssh -tt -p 22 $server << EOF
# cd /proj/cloudincr-PG0/ppo/PPO_tune_all_boards/
# nohup python3 main.py ent-$i $name > output_ent-$i.out 2>&1 &
# exit
# EOF
# done

# tune learning rate
for ((i = 0; i < 4; i++))
do
ssh -tt -p 22 $server << EOF
cd /proj/cloudincr-PG0/ppo/PPO_tune_all_boards/
nohup python3 main.py lr-$i $name > output_lr-$i.out 2>&1 &
exit
EOF
done

# for ((i = 0; i < 6; i++))
# do
# ssh -tt -p 22 $server << EOF
# cd /proj/cloudincr-PG0/ppo/PPO_tune_all_boards/
# nohup python3 main.py steps-$i $name > output_steps-$i.out 2>&1 &
# exit
# EOF
# done

done < id_server.csv
