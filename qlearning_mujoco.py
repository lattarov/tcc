
import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logger
logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return np.array(states), np.array(actions), np.array(rewards), np.array(next_states), np.array(dones)

    def __len__(self):
        return len(self.buffer)


def train_agent():
    env = gym.make('InvertedDoublePendulum-v4', render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = env.action_space.high[0]

    # Initialize networks
    actor = ActorNetwork(state_dim, action_dim, max_action).to(device)
    actor_target = ActorNetwork(state_dim, action_dim, max_action).to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = CriticNetwork(state_dim, action_dim).to(device)
    critic_target = CriticNetwork(state_dim, action_dim).to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(100000)

    episodes = 1000
    batch_size = 64
    gamma = 0.99
    tau = 0.005
    noise_scale = 0.1

    rewards_list = []

    for episode in range(episodes):
        state, _ = env.reset()
        env.render()
        total_reward = 0

        while True:
            # Add noise for exploration
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            action += noise_scale * np.random.normal(size=action_dim)
            action = np.clip(action, -max_action, max_action)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward

            # Store transition
            replay_buffer.add((state, action, reward, next_state, done))
            state = next_state


            # Train if enough samples are available
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.FloatTensor(actions).to(device)
                rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).unsqueeze(1).to(device)

                # Critic loss
                with torch.no_grad():
                    next_actions = actor_target(next_states_tensor)
                    target_q = rewards_tensor + gamma * (1 - dones_tensor) * critic_target(next_states_tensor, next_actions)
                current_q = critic(states_tensor, actions_tensor)
                critic_loss = nn.MSELoss()(current_q, target_q)

                # Optimize critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor loss
                actor_loss = -critic(states_tensor, actor(states_tensor)).mean()

                # Optimize actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            if done:
                break

        rewards_list.append(total_reward)

        if total_reward == max(rewards_list):
            logger.info("models have been updated")
            torch.save(actor, "neural_networks/mojuco_actor.pth")
            torch.save(critic, "neural_networks/mojuco_critic.pth")


        logger.info(f"Episode {episode + 1}: Reward {total_reward}, max reward: {max(rewards_list)}")

    return rewards_list



# Plot results
def plot_results(rewards):
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.show()
    plt.savefig("mujoco.png")


def setup_logging(logger: logging.Logger):
    # Stream handler for stdout with INFO level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for file with DEBUG level
    file_handler = logging.FileHandler('log.log')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":

    logger = setup_logging(logger)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device} for calculation")

    rewards = train_agent()
    plot_results(rewards)
