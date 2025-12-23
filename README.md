# rl-baselines
Baseline implementations of popular RL algorithms in pytorch.

## Installation and setup
This repo uses `uv` package manager. Just clone the repo, and run `uv sync` command inside the folder to setup the environment.

## REINFORCE algorithm
`vpg.py` contains the implementation of REINFORCE algorithm.<br>

Run `python vpg.py` to train agent on gymnasium's `CartPole-v1` environment, using REINFORCE.<br>

`CartPole-v1` agent trained using REINFORCE: [click here](./agent_video/rl-video-episode-0.mp4)

References:
https://spinningup.openai.com/en/latest/algorithms/vpg.html
https://lilianweng.github.io/posts/2018-04-08-policy-gradient/