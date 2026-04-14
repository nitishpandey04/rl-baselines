# Double DQN

A minimal but impactful fix to DQN's overestimation problem. Everything is identical to DQN except one line in the Bellman target.

## The problem with DQN

DQN computes the target as:

```
y = r + γ · max_a' Q_target(s', a')
```

`max` over noisy Q-estimates is biased upward — the highest value is likely the most overestimated one. The same network both selects the best action and evaluates it, compounding the error.

## The fix

Decouple action selection from action evaluation:

```
a* = argmax_a  Q_online(s', a)      ← online network selects
y  = r + γ · Q_target(s', a*)       ← target network evaluates
```

The online network picks the action it currently thinks is best. The target network, being an older and independent copy, gives a less correlated estimate of its value. The bias largely cancels out.

## In code

The only change from DQN (`dqn/dqn.py`) is these two lines in `update()`:

```python
# DQN
next_q = self.target_network(next_states).max(dim=1).values

# Double DQN
next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)  # online selects
next_q = self.target_network(next_states).gather(1, next_actions).squeeze(1)  # target evaluates
```

Everything else — replay buffer, target network, ε-greedy, hyperparameters — is the same.

## What to read next

**[DDPG](../ddpg/)** — extends Q-learning to continuous action spaces using an actor-critic architecture.

## Paper
van Hasselt et al., [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) (2015)
