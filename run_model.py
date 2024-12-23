"""Script to run simulations using different models."""

import argparse
import gymnasium as gym
import numpy as np
import torch

from qlearning_mujoco import ActorNetwork

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    args = parser.parse_args()

    env = gym.make('InvertedPendulum-v4', render_mode="human")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    actor = torch.load(args.model, weights_only=False)
    actor.eval()

    state, _ = env.reset()
    truncated = terminated = False

    while not truncated or terminated:

        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = actor(state_tensor).detach().cpu().numpy()[0]
        # action += 0.1 * np.random.normal(size=action_dim)
        action = np.clip(action, -max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        state = next_state
