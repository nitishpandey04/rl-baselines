"""
Advantage Actor-Critic (A2C)
============================
A clean, intuitive PyTorch implementation.

A2C extends VPG (REINFORCE) by adding a learned value function V(s) as a baseline.
Instead of using raw returns G as the policy gradient signal, it uses the advantage:

    A(s, a) = G - V(s)

This keeps the gradient unbiased while reducing its variance — the value baseline
tells the agent "how good is this state on average", so the policy only gets credit
for doing *better* than expected.

Two losses are optimized jointly:
    L_actor  = -mean( log π(a|s) · A(s, a) )       ← policy gradient with advantage
    L_critic =  mean( (V(s) - G)² )                 ← value function regression
    L        =  L_actor + c_value · L_critic         ← combined

Reference: Mnih et al., "Asynchronous Methods for Deep Reinforcement Learning" (2016)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import gymnasium as gym
from gymnasium.wrappers import RecordVideo


# ---------------------------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------------------------

class PolicyNetwork(nn.Module):
    """Actor: maps states → action logits."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ValueNetwork(nn.Module):
    """Critic: maps states → scalar V(s)."""

    def __init__(self, state_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------

class A2CAgent:
    """
    A2C Agent.

    select_action returns (action, log_prob, value) — all three are needed
    during trajectory collection and stored for the update step.

    update takes the collected log_probs, Monte-Carlo returns, and value
    estimates, computes advantages, and jointly updates actor and critic.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        value_coef: float = 0.5,    # weight of critic loss in the combined loss
        device: str = "cuda",
    ):
        self.gamma = gamma
        self.value_coef = value_coef
        self.action_dim = action_dim
        self.device = device

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value  = ValueNetwork(state_dim).to(device)

        self.actor_optimizer  = torch.optim.AdamW(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.value.parameters(),  lr=critic_lr)

    def select_action(self, state):
        """Returns (action, log_prob, value_estimate)."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        logits = self.policy(state_t)
        dist   = Bernoulli(logits=logits) if self.action_dim == 2 else Categorical(logits=logits)
        action = dist.sample()

        value  = self.value(state_t)

        return action.item(), dist.log_prob(action), value

    def update(self, log_probs: list, returns: list, values: list) -> dict:
        """
        log_probs : list of scalar tensors, one per timestep
        returns   : list of floats (discounted Monte-Carlo returns)
        values    : list of scalar tensors V(s), one per timestep
        """
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        values_t  = torch.stack(values)

        # Advantage: how much better was this action than expected?
        advantages = returns_t - values_t.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_loss  = -(torch.stack(log_probs) * advantages).mean()
        critic_loss = F.mse_loss(values_t, returns_t)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {"actor_loss": actor_loss.item(), "critic_loss": critic_loss.item()}

    def save(self, path: str = "a2c_agent.pt"):
        torch.save({
            "policy": self.policy.state_dict(),
            "value":  self.value.state_dict(),
        }, path)

    def load(self, path: str = "a2c_agent.pt"):
        ckpt = torch.load(path, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "CartPole-v1",
    steps: int = 500,
    batch_size: int = 32,
    gamma: float = 0.99,
    actor_lr: float = 1e-3,
    critic_lr: float = 1e-3,
    device: str = "cuda",
) -> A2CAgent:
    env = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim, actor_lr=actor_lr, critic_lr=critic_lr, gamma=gamma, device=device)

    for step in range(steps):
        log_probs_batch, returns_batch, values_batch = [], [], []
        episode_rewards = []

        for _ in range(batch_size):
            obs, _ = env.reset()
            log_probs, rewards, values = [], [], []
            done = False

            while not done:
                action, log_prob, value = agent.select_action(obs)
                obs, reward, terminated, truncated, _ = env.step(action)

                log_probs.append(log_prob)
                rewards.append(reward)
                values.append(value)
                done = terminated or truncated

            # Discounted returns (Monte-Carlo)
            G = 0
            returns = []
            for r in reversed(rewards):
                G = r + gamma * G
                returns.insert(0, G)

            log_probs_batch.extend(log_probs)
            returns_batch.extend(returns)
            values_batch.extend(values)
            episode_rewards.append(sum(rewards))

        losses = agent.update(log_probs_batch, returns_batch, values_batch)
        avg_reward = sum(episode_rewards) / batch_size
        print(f"Step {step:3d} | Avg Reward: {avg_reward:8.2f} | "
              f"Actor Loss: {losses['actor_loss']:.4f} | Critic Loss: {losses['critic_loss']:.4f}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: A2CAgent, env_name: str = "CartPole-v1"):
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


def record(agent: A2CAgent, env_name: str = "CartPole-v1", video_folder: str = "./agent_video"):
    env = gym.make(env_name, render_mode="rgb_array")
    env = RecordVideo(env, video_folder=video_folder)
    obs, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = train(env_name="CartPole-v1", steps=500, batch_size=32)
    agent.save("a2c_cartpole.pt")
    play(agent, env_name="CartPole-v1")
