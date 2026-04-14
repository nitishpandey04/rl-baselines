# SAC — Soft Actor-Critic

SAC is the best default algorithm for continuous control. It is off-policy (sample efficient), handles continuous actions, and is significantly more stable than DDPG/TD3 across a wide range of tasks and hyperparameters. The key idea: maximize both reward *and* entropy.

## The maximum entropy framework

Standard RL maximizes expected return: `J(π) = E[ Σ r ]`

SAC maximizes expected return **plus** policy entropy at every step:

```
J(π) = E[ Σ r(s,a) + α · H(π(·|s)) ]
```

Why? A high-entropy policy:
- Explores broadly — doesn't commit to one solution early
- Is robust — doesn't get stuck in local optima
- Generalizes better — captures multiple modes of near-optimal behavior

`α` (temperature) controls the tradeoff between reward and entropy.

## How it works

### Stochastic actor with reparameterization

Unlike DDPG/TD3's deterministic policy, SAC learns a Gaussian:

```
u ~ N( μ(s), σ(s) )
a = tanh(u)                                    (squash to [-1, 1])
```

The reparameterization trick (`u = μ + σ·ε`, `ε ~ N(0,1)`) makes sampling differentiable so gradients flow back through the action to the policy parameters.

The tanh squashing requires a log-probability correction:

```
log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
```

Without this correction, the log-prob would be wrong and the entropy term meaningless.

### Entropy-augmented Bellman target

The critic target includes next-state entropy, incentivizing the policy to remain stochastic:

```
ã', log_π' = actor(s')
y = r + γ · ( min(Q1_t(s',ã'), Q2_t(s',ã')) - α · log_π' )
```

The `min` of twin critics (like TD3) prevents overestimation.

### Automatic temperature tuning

Instead of hand-tuning `α`, SAC learns it automatically. A third optimizer adjusts `log_alpha` to maintain a target entropy `H_target = -|action_dim|`:

```
L_α = mean( -α · (log_π(a|s) + H_target).detach() )
```

If the policy is too deterministic (entropy < H_target), `α` increases to push it toward more exploration. If too stochastic (entropy > H_target), `α` decreases.

This is why `log_alpha` is used instead of `alpha` directly — optimizing in log space keeps `α` positive and makes the gradient well-behaved.

### No target actor, no exploration noise

Since the policy is already stochastic, SAC doesn't need:
- A target actor (the current actor is used for next-action sampling)
- Explicit exploration noise (entropy maximization provides natural exploration)

## Compared to TD3

| | TD3 | SAC |
|---|---|---|
| Policy | Deterministic | Stochastic (Gaussian) |
| Exploration | Gaussian noise | Entropy maximization |
| Temperature | — | Learned automatically |
| Target actor | Yes | No |
| Stability | Good | Better |

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `actor_lr` | 3e-4 | Policy learning rate |
| `critic_lr` | 3e-4 | Q-function learning rate |
| `alpha_lr` | 3e-4 | Temperature learning rate |
| `tau` | 0.005 | Soft-update rate |
| `gamma` | 0.99 | Discount factor |
| `target_entropy` | `-action_dim` | Lower → more stochastic policy |

## Paper
Haarnoja et al., [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290) (2018)
