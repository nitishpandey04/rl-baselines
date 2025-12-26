import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import random
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
        self.size = size
        self.buffer = deque()

    def insert(self, transition):
        if len(self.buffer) == self.size:
            self.buffer.popleft()
        self.buffer.append(transition)

    def sample(self, batch_size=32):
        batch = random.sample(self.buffer, batch_size)
        batch_tensor = torch.tensor(batch)
        return batch_tensor

