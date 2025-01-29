"""environment for the simulation."""

import gymnasium as gym

env = gym.make("InvertedPendulum-v4")

STATE_DIM = env.observation_space.shape[0]
ACTION_DIM = env.action_space.shape[0]
MAX_ACTION = env.action_space.high[0]
