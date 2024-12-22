import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
import logging
import sys
from collections import deque


logger = logging.getLogger("MAIN")
logger.setLevel(logging.INFO)



# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

# Train the agent
def train_agent():
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    q_network = QNetwork(state_dim, action_dim).to(device)
    target_network = QNetwork(state_dim, action_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())
    target_network.eval()

    optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(10000)

    episodes = 50
    batch_size = 64
    gamma = 0.99
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01
    target_update_freq = 10

    rewards_list = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0

        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = q_network(state_tensor)
                action = torch.argmax(q_values).item()

            observation, reward, terminated, truncated, info = env.step(
                action)

            logger.debug(f"observation: {observation}, reward: {reward}, \
                terminated: {terminated}, truncated: {truncated}, \
                info: {info}")

            total_reward += reward

            # Store transition in replay buffer
            replay_buffer.add((state, action, reward, observation, terminated))
            state = observation

            # Sample from replay buffer and train
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_tensor = torch.FloatTensor(states).to(device)
                actions_tensor = torch.LongTensor(actions).unsqueeze(1).to(device)
                rewards_tensor = torch.FloatTensor(rewards).to(device)
                next_states_tensor = torch.FloatTensor(next_states).to(device)
                dones_tensor = torch.FloatTensor(dones).to(device)

                q_values = q_network(states_tensor).gather(1, actions_tensor).squeeze(1)
                next_q_values = target_network(next_states_tensor).max(1)[0]
                target_q_values = rewards_tensor + gamma * next_q_values * (1 - dones_tensor)

                loss = nn.MSELoss()(q_values, target_q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            done = truncated or terminated

        # Update epsilon
        epsilon = max(epsilon * epsilon_decay, epsilon_min)
        rewards_list.append(total_reward)

        # Update target network
        if episode % target_update_freq == 0:
            target_network.load_state_dict(q_network.state_dict())

        if total_reward == max(rewards_list):
            file_name = "neural_networks/cartpole_q_network.pth"
            torch.save(q_network.state_dict(), file_name)

        logger.info(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, max reward: {max(rewards_list)}")


    # Save the trained network
    env.close()

    logger.info("Training done.'")
    return rewards_list

# Plot results
def plot_results(rewards):
    plt.plot(rewards, label="Episode Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Performance")
    plt.legend()
    plt.show()
    plt.savefig("qlearning.png")

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
    logger.info(f"Using device: {device} for calculation")

    rewards = train_agent()
    plot_results(rewards)
