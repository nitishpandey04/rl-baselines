"""
Deep Deterministic Policy Gradient (DDPG)
==========================================
A clean, intuitive PyTorch implementation.

DDPG is an actor-critic, off-policy algorithm for continuous action spaces.
It combines ideas from DPG (deterministic policy gradient) and DQN:
  - Actor:  learns a deterministic policy  μ(s) → a
  - Critic: learns a Q-function            Q(s, a) → scalar
  - Uses replay buffer + target networks for stability.

Reference: Lillicrap et al., "Continuous control with deep reinforcement learning" (2015)
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
            nn.Tanh(),  # outputs in [-1, 1]
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
# Ornstein-Uhlenbeck Noise (temporally correlated exploration)
# ---------------------------------------------------------------------------

class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration.
    Produces smoother exploration than plain Gaussian noise, which helps
    in physical control tasks with inertia.
    """

    def __init__(self, action_dim: int, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def sample(self) -> np.ndarray:
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state += dx
        return self.state.copy()


# ---------------------------------------------------------------------------
# DDPG Agent
# ---------------------------------------------------------------------------

class DDPGAgent:
    """
    DDPG Agent.

    The learning loop works as follows:

    1. Actor picks action:        a = μ(s) + noise
    2. Environment returns:       s', r, done
    3. Store (s, a, r, s', done) in replay buffer
    4. Sample a mini-batch and update:

       Critic loss (MSE on Bellman equation):
           y = r + γ · Q_target(s', μ_target(s'))
           L_critic = (Q(s, a) - y)²

       Actor loss (maximize Q by ascending the policy gradient):
           L_actor = -mean( Q(s, μ(s)) )

    5. Soft-update target networks:
           θ_target ← τ·θ + (1-τ)·θ_target
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,       # discount factor
        tau: float = 0.005,         # target network soft-update rate
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        noise_type: str = "ou",     # "ou" or "gaussian"
        noise_sigma: float = 0.1,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_action = max_action

        # ---- Networks ----
        self.actor = Actor(state_dim, action_dim, max_action).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)

        # Target networks (frozen copies, updated via polyak averaging)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # ---- Optimizers ----
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ---- Replay buffer & exploration noise ----
        self.replay_buffer = ReplayBuffer(buffer_size)

        if noise_type == "ou":
            self.noise = OUNoise(action_dim, sigma=noise_sigma)
        else:
            self.noise = None
            self.noise_sigma = noise_sigma
            self._action_dim = action_dim

    # ----- Action selection -----

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        """Pick an action, optionally adding exploration noise."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()

        if explore:
            if self.noise is not None:
                action += self.noise.sample()
            else:
                action += self.noise_sigma * np.random.randn(self._action_dim)

        return np.clip(action, -self.max_action, self.max_action)

    # ----- Learning step -----

    def update(self) -> dict:
        """
        Sample a batch and do one gradient step on actor & critic.
        Returns a dict of losses for logging.
        """
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # ---- Update Critic ----
        with torch.no_grad():
            # Target actions from target actor
            next_actions = self.actor_target(next_states)
            # Bellman target: y = r + γ * Q_target(s', μ_target(s'))
            target_q = rewards + (1.0 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---- Update Actor ----
        # Maximize Q(s, μ(s)) ⟹ minimize -Q(s, μ(s))
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Soft-update target networks ----
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return {"critic_loss": critic_loss.item(), "actor_loss": actor_loss.item()}

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    # ----- Persistence -----

    def save(self, path: str):
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "Pendulum-v1",
    max_episodes: int = 200,
    max_steps: int = 200,
    warmup_steps: int = 1000,       # random actions before learning starts
    log_interval: int = 10,
    **agent_kwargs,
):
    """Standard DDPG training loop."""

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = DDPGAgent(state_dim, action_dim, max_action, **agent_kwargs)

    total_steps = 0
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        if hasattr(agent, "noise") and agent.noise is not None:
            agent.noise.reset()

        episode_reward = 0.0

        for step in range(max_steps):
            # Warmup: take random actions to fill the buffer
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state, explore=True)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(terminated))

            # Learn after warmup
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
    return agent, episode_rewards


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def play(agent, env_name="Pendulum-v1", episodes=5):
    env = gym.make(env_name, render_mode="human")
    
    for ep in range(1, episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False

        while not done:
            action = agent.select_action(state, explore=False)  # no noise
            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated

        print(f"Episode {ep}: reward = {total_reward:.2f}")

    env.close()


if __name__ == "__main__":
    agent, rewards = train(
        env_name="Pendulum-v1",
        max_episodes=200,
        max_steps=200,
        warmup_steps=1000,
        batch_size=256,
        gamma=0.99,
        tau=0.005,
        actor_lr=1e-4,
        critic_lr=1e-3,
        noise_type="ou",
        noise_sigma=0.2,
    )
    agent.save("ddpg_pendulum.pt")
    print("Training complete. Model saved.")

    # load and visualize trained agent
    # env = gym.make("Pendulum-v1")
    # agent = DDPGAgent(
    #     state_dim=env.observation_space.shape[0],
    #     action_dim=env.action_space.shape[0],
    #     max_action=float(env.action_space.high[0]),
    # )
    # agent.load("ddpg_pendulum.pt")
    # env.close()

    play(agent)
