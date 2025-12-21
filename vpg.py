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


class ReinforceTrainer:
    def __init__(self, steps: int=30, batch_size: int=32, device: str="cuda"):
        self.steps = steps
        self.batch_size = batch_size
        self.device = device
        self.policy = Policy().to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-2)
        self.env = gym.make("CartPole-v1", render_mode="rgb_array")

    def train(self):
        for step in range(self.steps):
            # sample an episode
            log_probs_batch = []
            rewards_batch = []

            for i in range(self.batch_size):
                observation, info = self.env.reset()
                
                log_probs = []
                rewards = []
                while True:
                    obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                    logits = self.policy(obs)

                    # bernoulli distribution because it makes sampling of log prob a differentiable operation
                    # otherwise taking probability from sigmoid layer, doing log prob
                    dist = Bernoulli(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    action = action.to(torch.int32).item()

                    observation, reward, terminated, truncated, info = self.env.step(action)
                    log_probs.append(log_prob)
                    rewards.append(reward)

                    if terminated or truncated:
                        break

                # compute rewards-to-go. TODO: add discount factor
                for i in range(len(rewards) - 2, -1, -1):
                    rewards[i] += rewards[i + 1]

                log_probs_batch.extend(log_probs)
                rewards_batch.extend(rewards)

            # normalize rewards across batch
            rewards_batch = torch.tensor(rewards_batch, device=self.device)
            max_reward_per_batch = rewards_batch.max()
            rewards_batch = (rewards_batch - rewards_batch.mean()) / (rewards_batch.std() + 1e-8)

            # compute loss
            log_probs_tensor = torch.stack(log_probs_batch).view(-1)
            loss = -(log_probs_tensor * rewards_batch).sum()

            # update policy
            self.optimizer.zero_grad()
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
            self.optimizer.step()

            if step % 1 == 0:
                print(f"Episode {step} | Loss {loss:.5f} | Grad norm {grad_norm:.5f} | Max reward per batch {max_reward_per_batch}")

        self.save_policy()

    def play(self):
        # inference policy
        env = gym.make("CartPole-v1", render_mode="human")
        observation, info = env.reset()
        total_reward = 0
        while True:
            obs = torch.as_tensor(observation, device=self.device, dtype=torch.float32)
            logit = self.policy(obs)
            prob = F.sigmoid(logit)
            action = torch.round(prob).to(torch.int32).item()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        env.close()

    def save_policy(self, checkpoint_path="agent.pt"):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load_policy(self, checkpoint_path="agent.pt"):
        state_dict = torch.load(checkpoint_path, weights_only=True)
        self.policy.load_state_dict(state_dict)
        

trainer = ReinforceTrainer()

# trainer.train()

# before training
trainer.play()
trainer.load_policy()
# after training
trainer.play()