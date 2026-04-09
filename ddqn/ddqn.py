"""
Double DQN
==========
A clean, intuitive PyTorch implementation.

The only difference from DQN is in how the Bellman target is computed.
Standard DQN uses the target network for both action selection and evaluation:

    y = r + γ · max_a' Q_target(s', a')

This causes overestimation because the same network picks and scores the action.
Double DQN decouples the two steps:

    a* = argmax_a  Q_online(s', a)       ← online network selects
    y  = r + γ · Q_target(s', a*)        ← target network evaluates

Reference: van Hasselt et al., "Deep Reinforcement Learning with Double Q-learning" (2015)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque


# ---------------------------------------------------------------------------
# Neural Network
# ---------------------------------------------------------------------------

class QNetwork(nn.Module):
    """Q-value network: maps states → Q(s, a) for all actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
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


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------

class ReplayBuffer:
    """Simple FIFO experience replay buffer."""

    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Double DQN Agent
# ---------------------------------------------------------------------------

class DoubleDQNAgent:
    """
    Double DQN Agent.

    Identical to DQN except the Bellman target uses the online network to
    select the next action and the target network to evaluate it:

        a* = argmax_a Q_online(s', a)
        y  = r + γ · Q_target(s', a*)
        L  = MSE(Q_online(s, a), y)

    Target network is hard-updated every `target_update_freq` steps.
    Exploration follows an ε-greedy schedule.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-3,
        batch_size: int = 64,
        buffer_size: int = 100_000,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        target_update_freq: int = 100,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self._update_count = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, explore: bool = True) -> int:
        if explore and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def update(self) -> dict:
        if len(self.replay_buffer) < self.batch_size:
            return {}

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Current Q-values for taken actions
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: online selects, target evaluates
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)  # online selects
            next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)  # target evaluates
            targets = rewards + (1.0 - dones) * self.gamma * next_q

        loss = F.mse_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        self._update_count += 1
        if self._update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": loss.item(), "epsilon": self.epsilon}

    def save(self, path: str = "ddqn_agent.pt"):
        torch.save(self.q_network.state_dict(), path)

    def load(self, path: str = "ddqn_agent.pt"):
        self.q_network.load_state_dict(torch.load(path, weights_only=True))
        self.target_network.load_state_dict(self.q_network.state_dict())


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(
    env_name: str = "CartPole-v1",
    max_episodes: int = 500,
    max_steps: int = 500,
    log_interval: int = 10,
    **agent_kwargs,
) -> DoubleDQNAgent:
    import gymnasium as gym
    import numpy as np

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = DoubleDQNAgent(state_dim, action_dim, **agent_kwargs)
    episode_rewards = []

    for episode in range(1, max_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0.0

        for _ in range(max_steps):
            action = agent.select_action(state, explore=True)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.replay_buffer.push(state, action, reward, next_state, float(terminated))
            agent.update()

            state = next_state
            episode_reward += reward

            if done:
                break

        episode_rewards.append(episode_reward)

        if episode % log_interval == 0:
            avg = np.mean(episode_rewards[-log_interval:])
            print(f"Episode {episode:4d} | Avg Reward (last {log_interval}): {avg:8.2f} | ε: {agent.epsilon:.3f}")

    env.close()
    return agent


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def play(agent: DoubleDQNAgent, env_name: str = "CartPole-v1", episodes: int = 5):
    import gymnasium as gym

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
    agent = train(env_name="CartPole-v1", max_episodes=500)
    agent.save("ddqn_cartpole.pt")
    play(agent, env_name="CartPole-v1")
