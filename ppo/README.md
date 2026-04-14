# PPO — Proximal Policy Optimization

PPO is the most widely used policy gradient algorithm in practice. It keeps the actor-critic structure of A2C but adds two ideas that make training dramatically more stable: a clipped update objective and reuse of collected data across multiple epochs.

## How it works

### 1. Clipped surrogate objective

A2C directly maximizes `log π(a|s) · A`. The problem: a large gradient step can move the policy too far, collapsing performance in a single update.

PPO replaces the log probability with an importance-sampling ratio `r = π_new(a|s) / π_old(a|s)` and clips it:

```
L_clip = mean( min( r·A,  clip(r, 1-ε, 1+ε)·A ) )
```

When A > 0 (good action): the update is capped at `(1+ε)·A` — can't increase probability too aggressively.  
When A < 0 (bad action): the update is capped at `(1-ε)·A` — can't decrease probability too aggressively.

This creates a conservative trust region without needing to solve a constrained optimization (unlike TRPO).

### 2. Multiple epochs on the same data

A2C collects a batch and does one gradient step. PPO runs `K` epochs over the same batch. The clip prevents overfitting to the same data since large ratio deviations get penalized.

### 3. GAE — Generalized Advantage Estimation

Instead of Monte-Carlo returns, PPO uses GAE to estimate advantages:

```
δt  = rt + γ·V(st+1) - V(st)        (TD error)
At  = δt + (γλ)·δt+1 + (γλ)²·δt+2 + ...
```

`λ` interpolates between:
- `λ=1` → Monte-Carlo: high variance, low bias
- `λ=0` → TD(0): low variance, high bias

`λ=0.95` is the standard choice — mostly Monte-Carlo with some smoothing.

### 4. Entropy bonus

A small entropy term `c_entropy · H(π)` is added to the loss to discourage premature convergence to a deterministic policy.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `clip_eps` | 0.2 | Clip range. Smaller → more conservative updates |
| `epochs` | 10 | Gradient steps per rollout. Too high → policy drifts from old_log_probs |
| `gae_lambda` | 0.95 | GAE λ. Higher → more Monte-Carlo |
| `steps_per_iter` | 2048 | Timesteps collected before each update |
| `entropy_coef` | 0.01 | Exploration bonus. Higher → more random policy |

## What to read next

For continuous action spaces, **[DDPG](../ddpg/)** or **[SAC](../sac/)**.

## Paper
Schulman et al., [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) (2017)
