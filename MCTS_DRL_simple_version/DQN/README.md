# DQN-based Circuit Routing with via considered

1. In this version, we consider the number of via in the circuit routing problem. We set that the path can go through the obstacles and interact with other paths. So agent can extend the path to any direction until it reaches the target.

2. The reason we use DQN is that its repaly buffer can be used to combine MCTS.

3. Also, we will try to use both CNN and vanilla NN to performance the policy, respectively, and see what they can achieve.