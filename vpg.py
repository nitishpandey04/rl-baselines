from torch.distributions import Bernoulli
import torch.nn.functional as F
import gymnasium as gym
import torch.nn as nn
import torch

class Policy(nn.Module):
    def __init__(self,):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        logits = self.layers(obs)
        return logits


device = "cuda"
env = gym.make("CartPole-v1", render_mode="rgb_array")
policy = Policy().to(device)
optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-2)


# TODO: batching, loss normalization, discount factor
steps = 500
batch_size = 4
for step in range(steps):
    # sample an episode

    log_probs_batch = []
    rewards_batch = []

    for i in range(batch_size):
        observation, info = env.reset()
        
        log_probs = []
        rewards = []
        while True:
            obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
            logits = policy(obs)

            dist = Bernoulli(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_observation, reward, terminated, truncated, info = env.step(action.to(torch.int32).item())
            log_probs.append(log_prob)
            rewards.append(reward)

            if terminated or truncated:
                break

        # compute rewards-to-go
        for i in range(len(rewards) - 2, -1, -1):
            rewards[i] += rewards[i + 1]

        log_probs_batch.extend(log_probs)
        rewards_batch.extend(rewards)

    # normalize rewards across batch
    rewards_batch = torch.tensor(rewards_batch, device=device)
    rewards_batch = (rewards_batch - rewards_batch.mean()) / (rewards_batch.std() + 1e-8)

    # compute loss
    log_probs_tensor = torch.stack(log_probs_batch).view(-1)
    loss = -(log_probs_tensor * rewards_batch).sum()

    # update policy
    optimizer.zero_grad()
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    optimizer.step()

    if step % 1 == 0:
        print(f"Episode {step} | Loss {loss:.5f} | Grad norm {grad_norm:.5f}")





# inference
env = gym.make("CartPole-v1", render_mode="human")
observation, info = env.reset()
while True:
    obs = torch.as_tensor(device=device, dtype=torch.float32)
    logit = policy(obs)
    prob = F.sigmoid(logit)
    action = torch.round()




# Q1. why bernoulli distribution ? why not directly infer the action using prob and do log to get log prob
# Q2. is my way to calculate the final reward the correct way ? because that operation is not part of computation graph
