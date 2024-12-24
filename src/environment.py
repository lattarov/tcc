"""environment for the simulation."""

import gymnasium.envs.classic_control.cartpole as cartpole

class CustomCartPoleEnv(cartpole.CartPoleEnv):
    def __init__(self, **kwargs):
        super(CustomCartPoleEnv, self).__init__(**kwargs)

    def step(self, action):
        # Get the next state, reward, done, and info using the parent class's step function
        observation, reward, terminated, truncated, info = super(CustomCartPoleEnv, self).step(action)

        # Add penalty if the agent distances from the 0 position.
        reward -= 0 - observation[0]

        return observation, reward, terminated, truncated, info

