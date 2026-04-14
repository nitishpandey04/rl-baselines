# A2C — Advantage Actor-Critic

A2C extends VPG by adding a learned value function V(s) as a baseline. The key insight: instead of asking "was this action good?", ask "was this action *better than expected*?"

## How it works

A2C has two networks:
- **Actor** — the policy π(a|s), same as VPG
- **Critic** — a value function V(s) that estimates expected return from state s

The policy is updated using the **advantage** instead of raw returns:

```
A(s, a) = G - V(s)
```

If A > 0, the action was better than the baseline expected → increase its probability. If A < 0, it was worse → decrease it. This keeps the gradient direction the same as VPG but with much lower variance.

**Two losses optimized jointly:**
```
L_actor  = -mean( log π(a|s) · A(s,a) )
L_critic =  MSE( V(s), G )
```

## Why the baseline doesn't introduce bias

Subtracting any function b(s) that doesn't depend on the action from the return leaves the policy gradient unbiased:

```
E[ ∇ log π(a|s) · b(s) ] = 0
```

So V(s) is a perfect baseline — it reduces variance without changing what the gradient is pointing toward.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `actor_lr` | 1e-3 | Policy learning rate |
| `critic_lr` | 1e-3 | Value function learning rate |
| `gamma` | 0.99 | Discount factor |
| `batch_size` | 32 | Episodes per update |

## What to read next

**[PPO](../ppo/)** — keeps the actor-critic structure but clips the policy update to prevent large, destabilizing steps.

## Paper
Mnih et al., [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) (2016)
