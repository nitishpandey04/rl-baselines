# rl-baselines
Baseline implementations of popular RL algorithms in PyTorch. Each algorithm is self-contained in its own folder with a detailed README.

## Algorithms

| Algorithm | Type | Action Space | Default Env |
|-----------|------|--------------|-------------|
| [VPG](vpg/) (REINFORCE) | On-policy, policy gradient | Discrete | CartPole-v1 |
| [A2C](a2c/) | On-policy, actor-critic | Discrete | CartPole-v1 |
| [PPO](ppo/) | On-policy, actor-critic | Discrete | CartPole-v1 |
| [DQN](dqn/) | Off-policy, value-based | Discrete | CartPole-v1 |
| [Double DQN](ddqn/) | Off-policy, value-based | Discrete | CartPole-v1 |
| [DDPG](ddpg/) | Off-policy, actor-critic | Continuous | Pendulum-v1 |
| [TD3](td3/) | Off-policy, actor-critic | Continuous | Pendulum-v1 |
| [SAC](sac/) | Off-policy, actor-critic | Continuous | Pendulum-v1 |

## Which algorithm to use

**Discrete action spaces** — start with DQN. Use Double DQN as a drop-in improvement. If you prefer policy-based methods, VPG → A2C → PPO is the natural progression with PPO being the most reliable.

**Continuous action spaces** — SAC is the best default. TD3 is a solid alternative. DDPG is simpler but less stable; good for learning the actor-critic pattern before moving to TD3/SAC.

**Just learning RL** — follow the progression: VPG → A2C → PPO on the policy gradient side, DQN → DDQN → DDPG → TD3 → SAC on the value/actor-critic side. Each algo's README explains what it adds over the previous one.

## Installation

This repo uses `uv`. Clone the repo and run:

```bash
uv sync
```

## Usage

All algorithms follow the same interface:

```python
from <algo>.<algo> import train, play

agent = train(env_name="CartPole-v1")
play(agent, env_name="CartPole-v1")
```

## References
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [Policy Gradient Algorithms — Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
