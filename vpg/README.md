# VPG — Vanilla Policy Gradient (REINFORCE)

The simplest policy gradient algorithm. Directly optimizes the policy by ascending the gradient of expected return.

## How it works

Instead of learning a value function, VPG parameterizes the policy directly as π(a|s) and updates it using the policy gradient theorem:

```
∇J(θ) = E[ ∇ log π(a|s) · G ]
```

where G is the discounted return from that timestep. Intuitively: if an action led to high return, increase its probability; if low, decrease it.

**Training loop:**
1. Collect a batch of full episodes using the current policy
2. Compute discounted returns G for every timestep
3. Normalize returns (reduces variance)
4. Update: `loss = -mean( log π(a|s) · G )`

## Key limitations

- **High variance** — returns G include all future rewards, making the gradient estimate noisy. A2C fixes this with a baseline.
- **Sample inefficient** — collected data is thrown away after each update (on-policy).
- **No credit assignment** — a good action early in an episode gets credited for all subsequent rewards.

## Hyperparameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `lr` | 1e-2 | Higher → faster but unstable |
| `gamma` | 0.99 | Discount factor. Lower → agent is more short-sighted |
| `batch_size` | 32 | Episodes per update. Higher → lower variance, slower |

## What to read next

**[A2C](../a2c/)** — adds a value baseline to reduce gradient variance without changing the core idea.

## Paper
Williams, [Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) (1992)
