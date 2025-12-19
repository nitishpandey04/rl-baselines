import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# create a policy using function approximation

# so let's say i have an environment
# in that environment i will take an action
# the current state of environment will be fed to the policy network
# policy network will output an action

@dataclass
class TaskConfig:
    n_observation: int = 128
    n_hidden: int = 64
    n_action: int = 10


class Policy(nn.Module):
    def __init__(self, cfg: TaskConfig) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg.n_observation, cfg.n_hidden),
            nn.ReLU(),
            nn.Linear(cfg.n_hidden, cfg.n_hidden),
            nn.ReLU(),
            nn.Linear(cfg.n_hidden, cfg.n_action)
        )
        self.head = nn.Softmax(dim=-1)

    def forward(self, batch_obs: torch.Tensor) -> torch.Tensor:
        logits = self.layers(batch_obs)
        probs = self.head(logits)
        action = torch.argmax(probs, dim=-1)
        return action
    

env = None # assume you have the environment
policy = Policy(TaskConfig())
optimizer = torch.optim.AdamW(policy.parameters())

# training loop
steps = 100
for step in steps:
    # a single episode
    state = env.reset()
    trajectory = []
    while True:
        action = policy(state) # sample action from policy
        next_state, reward, terminated = env.step(action) # perform action in environment
        trajectory.append((state, action, reward, next_state))
        if terminated:
            break
    cost_function = 0
    for transition in trajectory:
        state, action, reward, next_state
        cost_function += -action.log() * reward # add to cost 
    optimizer.zero_grad()
    cost_function.backward()
    optimizer.step()
