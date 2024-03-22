#!/bin/bash
# tune neural network structure
# for n in 1 2 3
# do
# for i in 16 32 64
# do
# 	if [ $n == 1 ]
# 	then
# 		sed "86s/.*/    hidden_sizes: list = _([$i])/" config.py > config-nn-$n-$i.py
# 	fi
# 	if [ $n == 2 ]
# 	then
# 		sed "86s/.*/    hidden_sizes: list = _([$i, $i])/" config.py > config-nn-$n-$i.py
# 	fi
# 	if [ $n == 3 ]
# 	then
# 		sed "86s/.*/    hidden_sizes: list = _([$i, $i, $i])/" config.py > config-nn-$n-$i.py
# 	fi
# done
# done

# for n in 1 2 3
# do
# for i in 16 32 64
# do
# 	nohup python3 main.py nn-$n-$i > output_nn-$n-$i.out 2>&1 &
# done
# done


# tune learning rate
# values=(0.01 0.001 0.0001 0.00001)

# for ((i = 0; i < 4; i++))
# do
# 	sed "84s/.*/    lr: float = ${values[i]}/" config.py > config-lr-$i.py
# done

# for ((i = 0; i < 4; i++))
# do
# 	nohup python3 main.py lr-$i > output_lr-$i.out 2>&1 &
# done


# tune entropy coefficient
values=(0 0.1 0.2 0.3 0.4)

for ((i = 0; i < 5; i++))
do
	sed "90s/.*/    ent_coef: float = ${values[i]}/" config.py > config-ent-$i.py
done

for ((i = 0; i < 5; i++))
do
	nohup python3 main.py ent-$i > output_ent-$i.out 2>&1 &
done

# tune steps per epoch
# values=(100 200 500 1000 2000 5000)

# for ((i = 0; i < 6; i++))
# do
# 	sed "71s/.*/    steps_per_epoch: int = ${values[i]}/" config.py > config-steps-$i.py
# done

# for ((i = 0; i < 6; i++))
# do
# 	nohup python3 main.py steps-$i > output_steps-$i.out 2>&1 &
# done