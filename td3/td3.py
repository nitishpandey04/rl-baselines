"""
Twin Delayed DDPG (TD3)
=======================
A clean, intuitive PyTorch implementation.

TD3 is a direct upgrade to DDPG that fixes its tendency to overestimate Q-values.
It introduces three changes on top of DDPG:

  1. Twin critics — two independent Q-networks; the Bellman target uses the minimum
     of both to avoid overestimation:
         y = r + γ · min(Q1_target(s', ã), Q2_target(s', ã))

  2. Delayed policy updates — the actor and target networks update every
     `policy_delay` critic steps (default 2), reducing variance in the policy gradient.

  3. Target policy smoothing — clipped Gaussian noise is added to target actions
     during critic updates, smoothing the value landscape and preventing the actor
     from exploiting sharp Q-function peaks:
         ã = clip(μ_target(s') + clip(ε, -c, c), -max_action, max_action)

Reference: Fujimoto et al., "Addressing Function Approximation Error in Actor-Critic Methods" (2018)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym


# ---------------------------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Deterministic policy: maps states → actions (bounded by max_action)."""

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh(),
        )
        self.max_action = max_action

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """Q-function: maps (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple FIFO experience replay buffer."""

    def __init__(self, capacity: int = 1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# TD3 Agent
# ---------------------------------------------------------------------------

class TD3Agent:
    """
    TD3 Agent.

    The learning loop works as follows:

    Every step:
      1. Actor picks action:   a = μ(s) + ε,  ε ~ N(0, σ)
      2. Environment returns:  s', r, done
      3. Store (s, a, r, s', done) in replay buffer
      4. Update twin critics:
           ã = clip(μ_target(s') + clip(ε, -c, c), -max_a, max_a)
           y = r + γ · min(Q1_target(s', ã), Q2_target(s', ã))
           L = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

    Every `policy_delay` steps:
      5. Update actor:   L_actor = -mean(Q1(s, μ(s)))
      6. Soft-update all target networks
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        exploration_noise: float = 0.1,   # std of Gaussian noise added during data collection
        policy_noise: float = 0.2,         # std of smoothing noise added to target actions
        noise_clip: float = 0.5,           # clipping range for smoothing noise
        policy_delay: int = 2,             # actor updates once per this many critic updates
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action
        self.exploration_noise = exploration_noise
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self._update_count = 0

        # ---- Twin critics ----
        self.critic1 = Critic(state_dim, action_dim).to(self.device)
        self.critic2 = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        # ---- Actor ----
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)

        # ---- Optimizers ----
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if explore:
            action += self.exploration_noise * np.random.randn(action.shape[0])

        return np.clip(action, -self.max_action, self.max_action)

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        self._update_count += 1
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ---- Update Twin Critics ----
        with torch.no_grad():
            # Target policy smoothing: add clipped noise to target actions
            noise = torch.FloatTensor(actions.shape).normal_(0, self.policy_noise).to(self.device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-self.max_action, self.max_action)

            # Conservative target: take the minimum of both critics
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q  = rewards + (1.0 - dones) * self.gamma * torch.min(target_q1, target_q2)

        critic1_loss = F.mse_loss(self.critic1(states, actions), target_q)
        critic2_loss = F.mse_loss(self.critic2(states, actions), target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        log = {"critic1_loss": critic1_loss.item(), "critic2_loss": critic2_loss.item()}

        # ---- Delayed Actor + Target Updates ----
        if self._update_count % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self._soft_update(self.actor,   self.actor_target)
            self._soft_update(self.critic1, self.critic1_target)
            self._soft_update(self.critic2, self.critic2_target)

            log["actor_loss"] = actor_loss.item()

        return log

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str = "td3_agent.pt"):
        torch.save({
            "actor":   self.actor.state_dict(),
            "critic1": self.critic1.state_dict(),
            "critic2": self.critic2.state_dict(),
        }, path)

    def load(self, path: str = "td3_agent.pt"):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.actor_target   = copy.deepcopy(self.actor)
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "Pendulum-v1",
    max_episodes: int = 200,
    max_steps: int = 200,
    warmup_steps: int = 1000,
    log_interval: int = 10,
    **agent_kwargs,
) -> TD3Agent:
    env = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(state_dim, action_dim, max_action, **agent_kwargs)

    total_steps = 0
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(terminated))

            if total_steps >= warmup_steps:
                agent.update()

            state = next_state
            episode_reward += reward
            total_steps += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode % log_interval == 0:
            avg = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode:4d} | Avg Reward (last {log_interval}): {avg:8.2f} | Steps: {total_steps}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: TD3Agent, env_name: str = "Pendulum-v1", episodes: int = 5):
    env = gym.make(env_name, render_mode="human")
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward, done = 0.0, False
        while not done:
            action = agent.select_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        print(f"Episode {ep}: reward = {total_reward:.2f}")
    env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    agent = train(
        env_name="Pendulum-v1",
        max_episodes=200,
        max_steps=200,
        warmup_steps=1000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        exploration_noise=0.1,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_delay=2,
    )
    agent.save("td3_pendulum.pt")
    play(agent)
