import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import gymnasium as gym


# create a policy using function approximation
# so let's say i have an environment
# in that environment i will take an action
# the current state of environment will be fed to the policy network
# policy network will output an action

class Policy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 48),
            nn.ReLU(),
            nn.Linear(48, 48),
            nn.ReLU(),
            nn.Linear(48, 1),
            nn.Sigmoid()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        prob = self.layers(obs)
        action = torch.round(prob)
        return action


device = "cuda"
env = gym.make("CartPole-v1")
print(env.action_space)

policy = Policy().to(device)
optimizer = torch.optim.AdamW(policy.parameters())

# training loop
# batch size 1
# using r_t as training signal
steps = 1000
for step in range(steps):
    # a single episode
    observation, info = env.reset()
    total_reward = torch.tensor([0.0], device=device) # analogous to loss
    while True:
        observation = torch.tensor(observation, device=device)
        action = policy(observation) # sample action from policy
        observation, reward, terminated, truncated, info = env.step(action) # perform action in environment
        total_reward += reward * action.log() # TODO: improve the cost evaluator
        if terminated or truncated:
            break
    optimizer.zero_grad()
    total_reward.backward()
    optimizer.step()

