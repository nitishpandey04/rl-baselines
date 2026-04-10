# rl-baselines
Baseline implementations of popular RL algorithms in PyTorch.

## Algorithms

| Algorithm | Type | Action Space | Default Env |
|-----------|------|--------------|-------------|
| [VPG](vpg/vpg.py) (REINFORCE) | On-policy, policy gradient | Discrete | CartPole-v1 |
| [A2C](a2c/a2c.py) | On-policy, actor-critic | Discrete | CartPole-v1 |
| [DQN](dqn/dqn.py) | Off-policy, value-based | Discrete | CartPole-v1 |
| [Double DQN](ddqn/ddqn.py) | Off-policy, value-based | Discrete | CartPole-v1 |
| [DDPG](ddpg/ddpg.py) | Off-policy, actor-critic | Continuous | Pendulum-v1 |
| [TD3](td3/td3.py) | Off-policy, actor-critic | Continuous | Pendulum-v1 |

## Installation and setup
This repo uses `uv`. Clone the repo and run:

```bash
uv sync
```

## Usage

**VPG (REINFORCE)**
```python
from vpg.vpg import ReinforceTrainer

trainer = ReinforceTrainer(env_id="CartPole-v1")
trainer.train()
trainer.play()
```

**DDPG**
```python
from ddpg.ddpg import train, visualize

agent, rewards = train(env_name="Pendulum-v1", max_episodes=200)
visualize(agent)
```

## References
- [Spinning Up in Deep RL](https://spinningup.openai.com/en/latest/index.html)
- [Policy Gradient Algorithms — Lilian Weng](https://lilianweng.github.io/posts/2018-04-08-policy-gradient/)
- Lillicrap et al., [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (DDPG, 2015)
- van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (Double DQN, 2015)
- Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (TD3, 2018)
- Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (A3C/A2C, 2016)