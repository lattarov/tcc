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

SIMULATION_STEPS = 2000


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("controller")
    parser.add_argument("-impulse", action="store_false")
    parser.add_argument("render", action="store_false")
    args = parser.parse_args()

    env = gym.make(
        "InvertedPendulum-v4",
        render_mode=None if not args.render else "human",
        max_episode_steps=SIMULATION_STEPS
    )

    states, _ = env.reset()

    # initialize variables
    truncated = terminated = False
    state_buff = list()

    controllers = {"PID": PIDController(),
                   "RL":  ActorCriticAgent("neural_networks/mojuco_actor.pth")}

    controller = controllers[args.controller]

    if args.impulse:
        action = [2]

        next_state, reward, terminated, truncated, _ = env.step(action)

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
    df.plot(x="time", y="x", kind='scatter', grid=True, xlabel="Tempo [s]", ylabel="Posição [m]")

    df.plot.scatter(x='x', y="theta", c='time', grid=True, colormap='viridis', s=100, xlabel="Posição [m]", ylabel=r"$\theta$ [rad]")

    df.plot(x="time", y="theta", kind='scatter', grid=True, xlabel="Tempo [s]", ylabel=r"$\theta$ [rad]")

    plt.show()

    # performance statistics
    def angle_mean_squared_error(df: pd.DataFrame) -> float:
        reference = np.zeros(shape=len(df))
        mean_squared_error_theta = np.sum(np.square(reference - df["theta"]))/len(df)
        mean_squared_error_x = np.sum(np.square(reference - df["x"]))/len(df)
        return [mean_squared_error_theta, mean_squared_error_x]

    print(f"Mean squared error for {args.controller}: \
        {angle_mean_squared_error(df)}")
