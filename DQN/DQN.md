# DQN

## Modules in the agent

1. Two networks

There should be a target network(Estimate q_target) and a action-value network(Predict q_eval). The action-value network is newer. In reality, the action-value network is used to select action, and the target network is older version with same structure. The target network will be replaced by action-value network after few iterations.

In a step, select action by **action-value** network, calculate y = r_j + eta*max(Q)by **target** network, update parameter of **action-value** network by calculating loss of y and **action-value** network. After some steps, replace network.