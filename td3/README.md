# TD3 — Twin Delayed DDPG

TD3 is a direct upgrade to DDPG that addresses its overestimation and instability problems with three targeted fixes. If you're using DDPG, you should probably be using TD3 instead.

## The three fixes

### 1. Twin critics (addresses overestimation)

DDPG uses a single critic. When the actor is updated to maximize Q(s, μ(s)), it exploits any overestimation in Q — the policy learns to trigger the critic's errors rather than actually improve.

TD3 trains two independent critics Q1 and Q2, and uses the **minimum** for the Bellman target:

```
y = r + γ · min( Q1_target(s', ã'), Q2_target(s', ã') )
```

Taking the min is pessimistic — it consistently underestimates rather than overestimates, which is much safer for policy optimization.

### 2. Delayed policy updates (reduces variance)

In DDPG, the actor and critics update every step. But the critic needs several updates to converge before the actor gradient is reliable — updating the actor on a poorly-fitted critic introduces noise.

TD3 updates the actor (and target networks) only once every `policy_delay` critic updates (default: 2). This lets the critic stabilize before the actor uses it.

### 3. Target policy smoothing (prevents exploitation)

When computing the critic target, DDPG uses `μ_target(s')` directly. If the critic has sharp peaks, the actor learns to seek them out rather than find genuinely good actions.

TD3 adds clipped Gaussian noise to the target action:

```
ã' = clip( μ_target(s') + clip(ε, -c, c),  -max_a, max_a )     ε ~ N(0, σ)
```

This smooths the value landscape around the target action, making the critic harder to exploit.

## Compared to DDPG

| | DDPG | TD3 |
|---|---|---|
| Critics | 1 | 2 (min of both) |
| Actor update | every step | every `policy_delay` steps |
| Target actions | clean | + clipped noise |
| Exploration | OUNoise | Gaussian |

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `policy_delay` | 2 | Actor update frequency. Higher → more stable but slower actor |
| `policy_noise` | 0.2 | Smoothing noise std. Higher → smoother value landscape |
| `noise_clip` | 0.5 | Clip range for smoothing noise |
| `exploration_noise` | 0.1 | Std of noise added during data collection |
| `tau` | 0.005 | Soft-update rate for target networks |

## What to read next

**[SAC](../sac/)** — also fixes overestimation but uses a stochastic policy and entropy maximization, making it more robust and often better performing than TD3.

## Paper
Fujimoto et al., [Addressing Function Approximation Error in Actor-Critic Methods](https://arxiv.org/abs/1802.09477) (2018)
