"""Script to run simulations using different models."""

import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from controllers.actor_critic import ActorCriticAgent
from reinforcement_learning.actor import ActorNetwork  # TODO: move elsewhere


SEED = 42

SIMULATION_STEPS = 2000

max_action = 3


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    env = gym.make('InvertedPendulum-v4', render_mode="human", max_episode_steps=SIMULATION_STEPS)

    states, _ = env.reset(seed=SEED)

    # initialize variables
    truncated = terminated = False
    state_buff = list()
    controller = ActorCriticAgent(args.model)

    # main loop
    while True:

        action = controller.control(states)

        next_state, reward, terminated, truncated, _ = env.step(action)

        state_buff.append([*states, action[0]])

        states = next_state

        if truncated or terminated:
            states, _ = env.reset(seed=SEED)
            env.close()
            break

    # put data in a pd.DataFrame to simplify analysis
    time = np.linspace(0, SIMULATION_STEPS, endpoint=False)

    df = pd.DataFrame([time.transpose(), *state_buff], columns=["time", "x", "theta", "x_dot", "theta_dot", "action"])

    # plotting results
    df.plot()
    plt.show()
