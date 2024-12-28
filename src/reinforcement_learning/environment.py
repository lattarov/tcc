"""environment for the simulation."""

import gymnasium as gym

ENVIRONMENT = gym.make("InvertedPendulum-v4")

STATE_DIM = ENVIRONMENT.observation_space.shape[0]
ACTION_DIM = ENVIRONMENT.action_space.shape[0]
MAX_ACTION = ENVIRONMENT.action_space.high[0]
