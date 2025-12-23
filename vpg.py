import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class Policy(nn.Module):
    def __init__(self, n_observations, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_observations, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.layers(x)


class ReinforceTrainer:
    def __init__(self, env_id="CartPole-v1", steps=30, batch_size=32, gamma=0.99, device="cuda"):
        self.device = device
        self.gamma = gamma
        self.steps = steps
        self.batch_size = batch_size
        
        self.env = gym.make(env_id)
        self.n_obs = self.env.observation_space.shape[0]
        self.n_acts = self.env.action_space.n
        
        self.policy = Policy(self.n_obs, self.n_acts).to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=1e-2)

    def train(self):
        for step in range(self.steps):
            log_probs_batch = []
            returns_batch = []
            episode_rewards = []

            for _ in range(self.batch_size):
                observation, _ = self.env.reset()
                log_probs = []
                rewards = []
                
                done = False
                while not done:
                    state = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
                    logits = self.policy(state)

                    # sample action and get log_prob
                    if self.n_acts == 2:
                        dist = Bernoulli(logits=logits)
                    else:
                        dist = Categorical(logits=logits)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    
                    observation, reward, terminated, truncated, _ = self.env.step(action.item())
                    
                    log_probs.append(log_prob)
                    rewards.append(reward)
                    done = terminated or truncated

                # discounted rewards-to-go
                G = 0
                returns = []
                for r in reversed(rewards):
                    G = r + self.gamma * G
                    returns.insert(0, G)
                    
                log_probs_batch.extend(log_probs)
                returns_batch.extend(returns)
                episode_rewards.append(sum(rewards))

            # batch normalization
            returns_tensor = torch.tensor(returns_batch, device=self.device)
            returns_tensor = (returns_tensor - returns_tensor.mean()) / (returns_tensor.std() + 1e-8)

            # calculate loss
            log_probs_tensor = torch.stack(log_probs_batch)
            loss = -(log_probs_tensor * returns_tensor).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optimizer.step()

            avg_reward = sum(episode_rewards) / self.batch_size
            print(f"Step {step} | Avg Reward: {avg_reward:.2f} | Loss: {loss.item():.4f}")
            
    def play(self):
        env = gym.make(self.env.spec.id, render_mode="human")
        obs, _ = env.reset()
        done = False
        while not done:
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.policy(state)
                action = torch.argmax(logits).item() # greedy
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()

    def record(self, video_folder="./agent_video"):
        env = gym.make(self.env.spec.id, render_mode="rgb_array")
        env = RecordVideo(env, video_folder=video_folder)
        obs, _ = env.reset()
        done = False
        while not done:
            state = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                logits = self.policy(state)
                action = torch.argmax(logits).item() # greedy
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        env.close()

    def save_policy(self, checkpoint_path="agent.pt"):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def load_policy(self, checkpoint_path="agent.pt"):
        state_dict = torch.load(checkpoint_path, weights_only=True)
        self.policy.load_state_dict(state_dict)


# usage
trainer = ReinforceTrainer(env_id="CartPole-v1")
trainer.play()
trainer.train()
trainer.save_policy()
trainer.play()
trainer.record()
