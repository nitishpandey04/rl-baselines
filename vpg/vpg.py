import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


class Policy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )

    def forward(self, x):
        return self.layers(x)


class VPGAgent:
    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-2, gamma: float = 0.99, device: str = "cuda"):
        self.gamma = gamma
        self.device = device
        self.action_dim = action_dim

        self.policy = Policy(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.AdamW(self.policy.parameters(), lr=lr)

    def select_action(self, state):
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        logits = self.policy(state_t)

        dist = Bernoulli(logits=logits) if self.action_dim == 2 else Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, log_probs: list, returns: list) -> float:
        returns_t = torch.tensor(returns, device=self.device)
        returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        loss = -(torch.stack(log_probs) * returns_t).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save(self, path: str = "vpg_agent.pt"):
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str = "vpg_agent.pt"):
        self.policy.load_state_dict(torch.load(path, weights_only=True))


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "CartPole-v1",
    steps: int = 30,
    batch_size: int = 32,
    gamma: float = 0.99,
    lr: float = 1e-2,
    device: str = "cuda",
) -> VPGAgent:
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = VPGAgent(state_dim, action_dim, lr=lr, gamma=gamma, device=device)

    for step in range(steps):
        log_probs_batch, returns_batch, episode_rewards = [], [], []

        for _ in range(batch_size):
            obs, _ = env.reset()
            log_probs, rewards = [], []
            done = False

            while not done:
                action, log_prob = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                done = terminated or truncated

            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            log_probs_batch.extend(log_probs)
            returns_batch.extend(returns)
            episode_rewards.append(sum(rewards))

        loss = agent.update(log_probs_batch, returns_batch)
        avg_reward = sum(episode_rewards) / batch_size
        print(f"Step {step:3d} | Avg Reward: {avg_reward:8.2f} | Loss: {loss:.4f}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: VPGAgent, env_name: str = "CartPole-v1"):
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


def record(agent: VPGAgent, env_name: str = "CartPole-v1", video_folder: str = "./agent_video"):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = train(env_name="CartPole-v1", steps=30, batch_size=32)
    agent.save("vpg_cartpole.pt")
    play(agent, env_name="CartPole-v1")
