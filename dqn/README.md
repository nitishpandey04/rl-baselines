# DQN — Deep Q-Network

DQN was the first algorithm to successfully combine deep learning with Q-learning, learning to play Atari games from pixels. The core idea: approximate the Q-function Q(s,a) — the expected return of taking action a in state s — with a neural network.

## How it works

Q-learning learns by iteratively applying the Bellman equation:

```
Q(s, a) ← r + γ · max_a' Q(s', a')
```

The optimal policy is then simply: always pick the action with the highest Q-value.

With a neural network, naïve Q-learning is unstable. DQN fixes this with two tricks:

### 1. Experience replay

Transitions `(s, a, r, s', done)` are stored in a replay buffer. At each step, a random mini-batch is sampled for the update. This breaks temporal correlations between consecutive transitions that would otherwise destabilize training.

### 2. Target network

A separate target network (frozen copy of Q) is used to compute the Bellman target:

```
y = r + γ · max_a' Q_target(s', a')
L = MSE( Q(s, a),  y )
```

Without this, the target moves every step — like chasing a moving goalpost. The target network is hard-updated every `target_update_freq` steps.

### Exploration: ε-greedy

With probability ε, take a random action; otherwise take the greedy action. ε is annealed from 1.0 to a small value over training.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lr` | 1e-3 | Q-network learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` | 1.0 | Initial exploration rate |
| `epsilon_end` | 0.05 | Final exploration rate |
| `epsilon_decay` | 0.995 | Decay per step. Lower → faster decay |
| `target_update_freq` | 100 | Steps between target network syncs. Lower → more stable targets but slower learning |
| `batch_size` | 64 | Mini-batch size from replay buffer |

## Key limitation

DQN overestimates Q-values because `max` over noisy estimates is biased upward. **[Double DQN](../ddqn/)** fixes this.

## Paper
Mnih et al., [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) (2015)
