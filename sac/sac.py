"""
Soft Actor-Critic (SAC)
=======================
A clean, intuitive PyTorch implementation.

SAC is an off-policy actor-critic algorithm for continuous action spaces built on
the maximum entropy framework — it maximizes both expected return AND policy entropy:

    J(π) = Σ E[ r(s,a) + α · H(π(·|s)) ]

This makes the agent explore broadly and avoid premature commitment to suboptimal
behaviors. Three key ideas separate SAC from DDPG/TD3:

  1. Stochastic actor — learns a Gaussian policy with reparameterization:
         u ~ N(μ(s), σ(s))
         a = tanh(u)                        (squash to [-max_action, max_action])
         log π(a|s) = log N(u) - Σ log(1 - tanh²(u))   (log-prob correction)

  2. Entropy-augmented Bellman target — next-state entropy is subtracted from the
     target Q value, incentivizing the policy to stay stochastic:
         ã', log_π' = actor(s')
         y = r + γ · ( min(Q1_t(s',ã'), Q2_t(s',ã')) - α · log_π' )

  3. Automatic temperature tuning — α is learned by a third optimizer to maintain
     a target entropy H_target = -|A| (one negative action dimension):
         L_α = mean( -α · (log_π(a|s) + H_target).detach() )

No target actor or exploration noise is needed — the policy is already stochastic.

Reference: Haarnoja et al., "Soft Actor-Critic: Off-Policy Maximum Entropy Deep
           Reinforcement Learning with a Stochastic Actor" (2018)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import random
import gymnasium as gym

LOG_STD_MIN = -5
LOG_STD_MAX = 2


# ---------------------------------------------------------------------------
# Neural Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """
    Stochastic Gaussian actor.
    Outputs μ and log_σ, then samples via reparameterization and squashes with tanh.
    """

    def __init__(self, state_dim: int, action_dim: int, max_action: float, hidden: int = 256):
        super().__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
        )
        self.mean_layer    = nn.Linear(hidden, action_dim)
        self.log_std_layer = nn.Linear(hidden, action_dim)

    def forward(self, state: torch.Tensor):
        """Returns (action, log_prob). Uses reparameterization trick."""
        h       = self.net(state)
        mean    = self.mean_layer(h)
        log_std = self.log_std_layer(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std     = log_std.exp()

        # Reparameterization: sample u ~ N(mean, std), then squash
        dist = torch.distributions.Normal(mean, std)
        u    = dist.rsample()         # differentiable sample
        a    = torch.tanh(u)

        # Log-prob with tanh correction: log π(a) = log N(u) - Σ log(1 - tanh²(u))
        log_prob = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return self.max_action * a, log_prob


class Critic(nn.Module):
    """Q-function: maps (state, action) → Q-value."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),                 nn.ReLU(),
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
# SAC Agent
# ---------------------------------------------------------------------------

class SACAgent:
    """
    SAC Agent.

    The learning loop:
      1. Actor samples action stochastically: a, log_π = actor(s)
      2. Environment returns: s', r, done
      3. Store (s, a, r, s', done) in replay buffer
      4. Sample a mini-batch and update:

         Critic (twin Q-networks, entropy-augmented target):
             ã', log_π' = actor(s')
             y = r + γ · (min(Q1_t(s',ã'), Q2_t(s',ã')) - α·log_π')
             L_critic = MSE(Q1(s,a), y) + MSE(Q2(s,a), y)

         Actor (maximize Q - α·log_π):
             ã, log_π = actor(s)
             L_actor = mean(α·log_π - min(Q1(s,ã), Q2(s,ã)))

         Temperature α (maintain target entropy):
             L_α = mean(-α · (log_π + H_target).detach())

      5. Soft-update critic target networks
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        max_action: float,
        gamma: float = 0.99,
        tau: float = 0.005,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        alpha_lr: float = 3e-4,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        target_entropy: float = None,   # defaults to -action_dim
    ):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.max_action = max_action

        # ---- Actor ----
        self.actor           = Actor(state_dim, action_dim, max_action).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)

        # ---- Twin critics + targets ----
        self.critic1         = Critic(state_dim, action_dim).to(self.device)
        self.critic2         = Critic(state_dim, action_dim).to(self.device)
        self.critic1_target  = copy.deepcopy(self.critic1)
        self.critic2_target  = copy.deepcopy(self.critic2)
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=critic_lr)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=critic_lr)

        # ---- Automatic temperature tuning ----
        self.target_entropy = target_entropy if target_entropy is not None else -float(action_dim)
        self.log_alpha      = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

        self.replay_buffer = ReplayBuffer(buffer_size)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    def select_action(self, state: np.ndarray, explore: bool = True) -> np.ndarray:
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor(state_t)
        return action.cpu().numpy().flatten()

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states      = states.to(self.device)
        actions     = actions.to(self.device)
        rewards     = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones       = dones.to(self.device)

        # ---- Update Twin Critics ----
        with torch.no_grad():
            next_actions, next_log_pi = self.actor(next_states)
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            # Entropy-augmented Bellman target
            target_q  = rewards + (1.0 - dones) * self.gamma * (
                torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            )

        critic1_loss = F.mse_loss(self.critic1(states, actions), target_q)
        critic2_loss = F.mse_loss(self.critic2(states, actions), target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # ---- Update Actor ----
        new_actions, log_pi = self.actor(states)
        q1 = self.critic1(states, new_actions)
        q2 = self.critic2(states, new_actions)
        actor_loss = (self.alpha.detach() * log_pi - torch.min(q1, q2)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---- Update Temperature α ----
        alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ---- Soft-update critic targets ----
        self._soft_update(self.critic1, self.critic1_target)
        self._soft_update(self.critic2, self.critic2_target)

        return {
            "critic1_loss": critic1_loss.item(),
            "critic2_loss": critic2_loss.item(),
            "actor_loss":   actor_loss.item(),
            "alpha":        self.alpha.item(),
        }

    def _soft_update(self, source: nn.Module, target: nn.Module):
        """Polyak averaging: θ_target ← τ·θ + (1-τ)·θ_target"""
        for param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def save(self, path: str = "sac_agent.pt"):
        torch.save({
            "actor":    self.actor.state_dict(),
            "critic1":  self.critic1.state_dict(),
            "critic2":  self.critic2.state_dict(),
            "log_alpha": self.log_alpha.detach(),
        }, path)

    def load(self, path: str = "sac_agent.pt"):
        ckpt = torch.load(path, weights_only=True)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic1.load_state_dict(ckpt["critic1"])
        self.critic2.load_state_dict(ckpt["critic2"])
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        with torch.no_grad():
            self.log_alpha.copy_(ckpt["log_alpha"])


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
) -> SACAgent:
    env = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = SACAgent(state_dim, action_dim, max_action, **agent_kwargs)

    total_steps    = 0
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            if total_steps < warmup_steps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(state)

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
            print(f"Episode {episode:4d} | Avg Reward (last {log_interval}): {avg:8.2f} | "
                  f"α: {agent.alpha.item():.4f} | Steps: {total_steps}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: SACAgent, env_name: str = "Pendulum-v1", episodes: int = 5):
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
        actor_lr=3e-4,
        critic_lr=3e-4,
        alpha_lr=3e-4,
    )
    agent.save("sac_pendulum.pt")
    play(agent)
