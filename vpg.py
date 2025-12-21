from torch.distributions import Bernoulli
import gymnasium as gym
import torch.nn as nn
import torch


class Policy(nn.Module):
    def __init__(self,):
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
        probs = self.layers(obs)
        return probs


device = "cuda"
env = gym.make("CartPole-v1", render_mode="rgb_array")
policy = Policy().to(device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)


# TODO: discount factor
steps = 5000
for step in range(steps):
    # sample an episode
    observation, info = env.reset()

    all_log_probs = []
    all_rewards = []
    while True:
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
        probs = policy(obs)

        # Q1
        dist = Bernoulli(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_observation, reward, terminated, truncated, info = env.step(action.to(torch.int32).item())
        all_log_probs.append(log_prob)
        all_rewards.append(reward)

        if terminated or truncated:
            break
            
    # Q2
    cost_value = 0
    for i in range(len(all_rewards) - 2, -1, -1):
        all_rewards[i] += all_rewards[i + 1]

    for i, log_prob in enumerate(all_log_probs):
        cost_value += -log_prob * all_rewards[i]
    optimizer.zero_grad()
    cost_value.backward()
    optimizer.step()

    if step % 50 == 0:
        print(f"Episode {step + 1} | Total reward {all_rewards[0]:.2f}")







# Q1. why bernoulli distribution ? why not directly infer the action using prob and do log to get log prob
# Q2. is my way to calculate the final reward the correct way ? because that operation is not part of computation graph

