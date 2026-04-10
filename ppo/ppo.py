"""
Proximal Policy Optimization (PPO)
===================================
A clean, intuitive PyTorch implementation.

PPO builds on A2C with two key ideas:

  1. Clipped surrogate objective — controls how much the policy is allowed to
     change in a single update. It computes an importance-sampling ratio
     r = π_new(a|s) / π_old(a|s) and clips it to [1-ε, 1+ε], preventing
     destabilizing large steps:

         L_clip = mean( min(r·A, clip(r, 1-ε, 1+ε)·A) )

  2. Multiple epochs on the same rollout — unlike A2C which discards data after
     one gradient step, PPO reuses each collected batch for K epochs. The clip
     prevents overfitting to the same data.

  3. GAE (Generalized Advantage Estimation) — interpolates between high-variance
     Monte-Carlo returns (λ=1) and low-variance TD(0) (λ=0):

         δt  = rt + γ·V(st+1) - V(st)
         A_t = Σ_{k=0}^{T} (γλ)^k · δ_{t+k}

  Full loss:
      L = L_clip - c_value · MSE(V(s), G) + c_entropy · H(π)

Reference: Schulman et al., "Proximal Policy Optimization Algorithms" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli
import numpy as np
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
# PPO Agent
# ---------------------------------------------------------------------------

class PPOAgent:
    """
    PPO Agent.

    select_action  — called during rollout collection, returns (action, log_prob, value)
    update         — called once per rollout with the full batch; runs K epochs
                     over the data using the clipped surrogate objective
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,   # GAE λ: 1.0 → Monte-Carlo, 0.0 → TD(0)
        clip_eps: float = 0.2,       # PPO clip range
        epochs: int = 10,            # gradient epochs per rollout
        value_coef: float = 0.5,     # weight of critic loss
        entropy_coef: float = 0.01,  # weight of entropy bonus (encourages exploration)
        device: str = "cuda",
    ):
        self.gamma       = gamma
        self.gae_lambda  = gae_lambda
        self.clip_eps    = clip_eps
        self.epochs      = epochs
        self.value_coef  = value_coef
        self.entropy_coef = entropy_coef
        self.action_dim  = action_dim
        self.device      = device

        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.value  = ValueNetwork(state_dim).to(device)

        self.actor_optimizer  = torch.optim.AdamW(self.policy.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.AdamW(self.value.parameters(),  lr=critic_lr)

    def select_action(self, state):
        """Returns (action, log_prob, value_estimate). Called during rollout collection."""
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.policy(state_t)
            value  = self.value(state_t)

        dist   = Bernoulli(logits=logits) if self.action_dim == 2 else Categorical(logits=logits)
        action = dist.sample()

        return action.item(), dist.log_prob(action), value

    def compute_gae(self, rewards: list, values: list, dones: list, last_value: float) -> tuple:
        """
        Compute GAE advantages and discounted returns.

        last_value: V(s_T), the value of the state after the last step
                    (0 if the episode terminated, V(s_T) if truncated).
        """
        advantages = []
        gae = 0.0
        values_ext = values + [last_value]  # append bootstrap value

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values_ext[t + 1] * (1 - dones[t]) - values_ext[t]
            gae   = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns

    def update(
        self,
        states: list,
        actions: list,
        old_log_probs: list,
        advantages: list,
        returns: list,
    ) -> dict:
        """
        Run K epochs of PPO updates on the collected rollout.

        Inputs are lists of per-timestep data collected across multiple episodes.
        old_log_probs are fixed (from collection time) — used to compute the ratio.
        """
        states_t       = torch.as_tensor(np.array(states),     dtype=torch.float32, device=self.device)
        actions_t      = torch.as_tensor(np.array(actions),    dtype=torch.long,    device=self.device)
        old_log_probs_t = torch.stack(old_log_probs).detach()
        advantages_t   = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns_t      = torch.tensor(returns,    dtype=torch.float32, device=self.device)

        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        total_actor_loss, total_critic_loss = 0.0, 0.0

        for _ in range(self.epochs):
            # Recompute log_probs and values under the current policy
            logits   = self.policy(states_t)
            dist     = Bernoulli(logits=logits) if self.action_dim == 2 else Categorical(logits=logits)
            log_probs = dist.log_prob(actions_t)
            entropy   = dist.entropy().mean()
            values    = self.value(states_t)

            # Importance sampling ratio
            ratio = (log_probs - old_log_probs_t).exp()

            # Clipped surrogate objective
            surr1 = ratio * advantages_t
            surr2 = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages_t
            actor_loss  = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy
            critic_loss =  F.mse_loss(values, returns_t)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            total_actor_loss  += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return {
            "actor_loss":  total_actor_loss  / self.epochs,
            "critic_loss": total_critic_loss / self.epochs,
        }

    def save(self, path: str = "ppo_agent.pt"):
        torch.save({
            "policy": self.policy.state_dict(),
            "value":  self.value.state_dict(),
        }, path)

    def load(self, path: str = "ppo_agent.pt"):
        ckpt = torch.load(path, weights_only=True)
        self.policy.load_state_dict(ckpt["policy"])
        self.value.load_state_dict(ckpt["value"])


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "CartPole-v1",
    iterations: int = 200,
    steps_per_iter: int = 2048,   # timesteps collected before each update
    gamma: float = 0.99,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    device: str = "cuda",
    **agent_kwargs,
) -> PPOAgent:
    env = gym.make(env_name)
    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(state_dim, action_dim, actor_lr=actor_lr, critic_lr=critic_lr,
                     gamma=gamma, device=device, **agent_kwargs)

    for iteration in range(1, iterations + 1):
        # ---- Collect rollout ----
        states, actions, old_log_probs = [], [], []
        rewards, values, dones = [], [], []
        episode_rewards, current_ep_reward = [], 0.0

        obs, _ = env.reset()
        for _ in range(steps_per_iter):
            action, log_prob, value = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(obs)
            actions.append(action)
            old_log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value.item())
            dones.append(float(terminated))

            current_ep_reward += reward
            obs = next_obs

            if done:
                episode_rewards.append(current_ep_reward)
                current_ep_reward = 0.0
                obs, _ = env.reset()

        # Bootstrap value for the last state (0 if terminated, V(s) if truncated)
        _, _, last_value = agent.select_action(obs)
        last_value = 0.0 if dones[-1] else last_value.item()

        # ---- Compute GAE advantages and returns ----
        advantages, returns = agent.compute_gae(rewards, values, dones, last_value)

        # ---- Update ----
        losses = agent.update(states, actions, old_log_probs, advantages, returns)

        if episode_rewards:
            avg_reward = np.mean(episode_rewards)
            print(f"Iter {iteration:4d} | Avg Reward: {avg_reward:8.2f} | "
                  f"Actor Loss: {losses['actor_loss']:.4f} | Critic Loss: {losses['critic_loss']:.4f}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: PPOAgent, env_name: str = "CartPole-v1"):
    env = gym.make(env_name, render_mode="human")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _, _ = agent.select_action(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    env.close()


def record(agent: PPOAgent, env_name: str = "CartPole-v1", video_folder: str = "./agent_video"):
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
    agent = train(
        env_name="CartPole-v1",
        iterations=200,
        steps_per_iter=2048,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        epochs=10,
        value_coef=0.5,
        entropy_coef=0.01,
    )
    agent.save("ppo_cartpole.pt")
    play(agent, env_name="CartPole-v1")
