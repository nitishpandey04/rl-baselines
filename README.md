# rl-baselines
Baseline implementations of popular RL algorithms in pytorch.

## Installation and setup
This repo uses `uv` package manager. Just clone the repo, and run `uv sync` command inside the folder to setup the environment.

## REINFORCE algorithm
`vpg.py` contains the implementation of REINFORCE algorithm.<br><br>

Run `python vpg.py` to train agent on gymnasium's `CartPole-v1` environment, using REINFORCE.<br><br>

`CartPole-v1` agent trained using REINFORCE:
[Click to play video](./agent_video/rl-video-episode-0.mp4)

<video width="640" height="360" controls>
  <source src="./agent_video/rl-video-episode-0.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>