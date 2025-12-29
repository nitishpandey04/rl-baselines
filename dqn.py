import torch.nn.functional as F
from collections import deque
import torch.nn as nn
import random
import torch

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


class QValueFunction(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(512 * 3 * 3, 10)
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)

# policy = Policy().to("cuda")
# image = torch.randn((1, 3, 255, 255), device="cuda")
# print(image.shape)
# out = policy(image)
# print(out.shape)

# take a replay buffer
# take a q-value function

env = None
value_function = QValueFunction()
replay_buffer = ReplayBuffer(size=1000)
num_episodes = 100
epsilon = 1e-5
for ep in range(num_episodes):
    state = env.reset()
    done = False
    while True:
        values = value_function(state)
        prob = random.random()
        if prob < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            action = values.argmax() # take optimal action
            
        # we have a state, i obtained the action value functions for that state
        # we will select an action from the action values using epsilon greedy approach
        # we will implement that action in the environment, obtain the reward and state

        reward, next_state, terminated, truncated, info = env.step(action)
        transition = (state, action, reward, next_state, terminated or truncated)
        replay_buffer.insert(transition)
        
        transitions_batch = replay_buffer.sample()
        # create the targets heres

        # use the targets and the actual to estimate l2 loss
        # perform gradient descent
        