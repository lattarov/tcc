"""Script to run simulations using different models."""

import argparse
import gymnasium as gym
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from qlearning_mujoco import ActorNetwork

SEED = 42

max_action = 3

class ActorCriticAgent:
    def __init__(self, actor):
        self.actor = torch.load(args.model, weights_only=False)
        self.actor.eval()

    def control(self, states):
        state_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        action = np.clip(action, -max_action, max_action)
        return action

class PIDController:
    def __init__(self):
        self.kp = 1
        self.kd = 0.01
        self.ki = 0
        self.last_error = 0
        self.total_error = 0

    def control(self, ref, angle):
        error = angle - ref
        d_error = error - self.last_error
        self.last_error = error
        self.total_error = self.total_error + error
        action = self.kp * error + self.kd * d_error + self.ki*self.total_error
        action = np.clip(action, -max_action, max_action)
        return action


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    env = gym.make('InvertedPendulum-v4', render_mode="human", max_episode_steps=2000)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    states, _ = env.reset(seed=SEED)

    # env.data.qpos[:] = [0.2, 0]  # directly modify qpos values

    # initialize variables
    truncated = terminated = False
    state_buff = list()
    # pid = PIDController()
    controller = ActorCriticAgent(args.model)

    step = 0

    # main loop
    while True:

        # action = controller.control(states) + 0.1*np.sin(step/100)
        action = controller.control(states)

        next_state, reward, terminated, truncated, _ = env.step(action)

        state_buff.append([*states, action[0]])

        states = next_state

        step += 1

        if truncated or terminated:
            states, _ = env.reset(seed=SEED)
            env.close()
            break

    # put data in a pd.DataFrame to simplify analysis
    df = pd.DataFrame(state_buff, columns=["x", "theta", "x_dot", "theta_dot", "action"])

    # plotting results
    df.plot()
    plt.show()
