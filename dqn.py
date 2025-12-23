import torch.nn.functional as F
import torch.nn as nn
import torch

"""
calculate an action value function parameterized by theta
create a replay buffer of size N from which batch of transitions will be extracted
pseudocode:
for n episodes:
  sample an episode using epsilon greedy policy, using the action value function parameterized by theta
  insert that in replay buffer
  sample a batch of transitions
  obtain the target for each transition in the batch using fixed target policy
  obtain the prediction of action value function for that timestep
  update the value function approximation
"""

class ReplayBuffer:
    def __init__(self, size=100):
        self.buffer = []