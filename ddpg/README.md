# DDPG — Deep Deterministic Policy Gradient

DDPG extends DQN to continuous action spaces by combining it with a deterministic policy gradient. It is an actor-critic, off-policy algorithm.

## How it works

With continuous actions, you can't take `max_a Q(s,a)` directly — there are infinitely many actions. DDPG sidesteps this by learning a deterministic policy μ(s) → a that always outputs the action that maximizes Q:

```
Actor:  μ(s) → a                        (deterministic policy)
Critic: Q(s, a) → scalar                (Q-function)
```

**Critic update** — same as DQN with a Bellman target:
```
y = r + γ · Q_target(s', μ_target(s'))
L_critic = MSE( Q(s,a), y )
```

**Actor update** — directly ascend the Q gradient:
```
L_actor = -mean( Q(s, μ(s)) )
```

The actor is updated by backpropagating through the critic — this is the policy gradient.

### Stability tricks (inherited from DQN)

- **Replay buffer** — breaks temporal correlations
- **Target networks** — separate frozen copies of both actor and critic, soft-updated via Polyak averaging: `θ_target ← τ·θ + (1-τ)·θ_target`

### Exploration

Since the policy is deterministic, exploration noise must be added externally. DDPG uses Ornstein-Uhlenbeck noise — a temporally correlated process that produces smoother exploration than pure Gaussian noise, which helps in physical control tasks with inertia.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `actor_lr` | 1e-4 | Policy learning rate |
| `critic_lr` | 1e-3 | Q-function learning rate |
| `tau` | 0.005 | Soft-update rate. Higher → target tracks faster but less stable |
| `gamma` | 0.99 | Discount factor |
| `noise_sigma` | 0.1 | Exploration noise scale |
| `warmup_steps` | 1000 | Random actions before learning starts |

## Key limitation

DDPG overestimates Q-values (same root cause as DQN) and is sensitive to hyperparameters. **[TD3](../td3/)** fixes both.

## Paper
Lillicrap et al., [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971) (2015)
