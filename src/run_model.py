"""Script to run simulations using different models."""

import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from controllers.actor_critic import ActorCriticAgent
from controllers.pid import PIDController
from reinforcement_learning.actor import ActorNetwork  # TODO: move elsewhere


SEED = 42

SIMULATION_STEPS = 200

max_action = 3


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("controller")
    args = parser.parse_args()

    env = gym.make(
        "InvertedPendulum-v4",
        # render_mode="human",
        max_episode_steps=SIMULATION_STEPS
    )

    states, _ = env.reset(seed=SEED)

    # initialize variables
    truncated = terminated = False
    state_buff = list()

    controllers = {"PID": PIDController(),
                   "RL":  ActorCriticAgent("neural_networks/mojuco_actor.pth")}

    controller = controllers[args.controller]

    # main loop
    nb_steps = 0
    while True:
        action = controller.control(states)

        next_state, reward, terminated, truncated, _ = env.step(action)

        time = nb_steps*0.02

        state_buff.append([time, *states, action[0]])

        states = next_state

        nb_steps += 1

        if truncated or terminated:
            states, _ = env.reset(seed=SEED)
            env.close()
            break

    # put data in a pd.DataFrame to simplify analysis
    df = pd.DataFrame(
        state_buff,
        columns=["time", "x", "theta", "x_dot", "theta_dot", "action"],
    )

    # plotting results
    df.plot(x="time", grid=True)
    df.plot(x="time", y="x", title="position", kind='scatter', grid=True)
    df.plot(x="time", y="theta", title="angle", kind='scatter', grid=True)
    plt.show()

    # performance statistics
    def angle_mean_squared_error(df: pd.DataFrame) -> float:
        reference = np.zeros(shape=len(df))
        mean_squared_error = np.sum(np.square(reference - df["theta"])/len(df))
        return mean_squared_error

    print(f"Mean squared error for {args.controller}: \
        {angle_mean_squared_error(df)}")
