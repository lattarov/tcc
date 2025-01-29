import environment as env
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
import sys


from actor import ActorNetwork
from critic import CriticNetwork
from replay_buffer import ReplayBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set up logger
logger = logging.getLogger("MAIN")
logger.setLevel(logging.DEBUG)

EPISODES = 1000
BATCH_SIZE = 64
GAMMA = 0.99
TAU = 0.005
NOISE_SCALE = 0.1


def train_agent(rewards:list, losses_critic:list, losses_actor:list):
    # Initialize networks
    actor = ActorNetwork().to(device)
    actor_target = ActorNetwork().to(device)
    actor_target.load_state_dict(actor.state_dict())

    critic = CriticNetwork().to(device)
    critic_target = CriticNetwork().to(device)
    critic_target.load_state_dict(critic.state_dict())

    actor_optimizer = optim.Adam(actor.parameters(), lr=1e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=1e-3)
    replay_buffer = ReplayBuffer(100000)

    for episode in range(EPISODES):
        state, _ = env.env.reset()
        total_reward = 0

        while True:
            # Add noise for exploration
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).detach().cpu().numpy()[0]
            action += NOISE_SCALE * np.random.normal(size=env.ACTION_DIM)
            action = np.clip(action, -env.MAX_ACTION, env.MAX_ACTION)

            next_state, reward, terminated, truncated, _ = env.env.step(action)

            reward = (
                reward
                - (state[0] - 0) ** 2
                - 0.1 * action[0] ** 2
                - (0 - state[2]) ** 2
            )
            total_reward += reward

            # Store transition
            replay_buffer.add(
                (state, action, reward, next_state, terminated or truncated)
            )
            state = next_state

            # Train if enough samples are available
            if len(replay_buffer) >= BATCH_SIZE:
                rp_states, rp_actions, rp_rewards, rp_next_states, rp_dones = replay_buffer.sample(
                    BATCH_SIZE
                )

                states_tensor = torch.FloatTensor(rp_states).to(device)
                actions_tensor = torch.FloatTensor(rp_actions).to(device)
                rewards_tensor = torch.FloatTensor(rp_rewards).unsqueeze(1).to(device)
                next_states_tensor = torch.FloatTensor(rp_next_states).to(device)
                dones_tensor = torch.FloatTensor(rp_dones).unsqueeze(1).to(device)

                # Critic loss
                with torch.no_grad():
                    next_actions = actor_target(next_states_tensor)
                    target_q = rewards_tensor + GAMMA * (
                        1 - dones_tensor
                    ) * critic_target(next_states_tensor, next_actions)
                current_q = critic(states_tensor, actions_tensor)
                critic_loss = nn.MSELoss()(current_q, target_q)
                losses_critic.append(critic_loss.cpu().detach().numpy())

                # Optimize critic
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                # Actor loss
                actor_loss = -critic(states_tensor, actor(states_tensor)).mean()
                losses_actor.append(actor_loss.cpu().detach().numpy())


                # Optimize actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # Soft update target networks
                for param, target_param in zip(
                    actor.parameters(), actor_target.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )
                for param, target_param in zip(
                    critic.parameters(), critic_target.parameters()
                ):
                    target_param.data.copy_(
                        TAU * param.data + (1 - TAU) * target_param.data
                    )

            if terminated or truncated:
                break

        rewards.append(total_reward)

        is_new_maximum_reward = total_reward == max(rewards)
        model_update_message = (
            " - Best models models have been updated" if is_new_maximum_reward else ""
        )

        if is_new_maximum_reward:
            torch.save(actor, "neural_networks/mojuco_actor.pth")
            torch.save(critic, "neural_networks/mojuco_critic.pth")

        logger.debug(
            f"Episode {episode + 1}: Reward {total_reward}, MaxReward: {max(rewards)}{model_update_message}"
        )


def plot_results(rewards, losses_critic, losses_actor):
    plt.figure(1)
    plt.scatter(rewards)
    plt.xlabel("Episódio [N°]")
    plt.ylabel("Recompensa []")
    plt.grid()
    plt.legend()
    plt.show()
    # plt.savefig("mujoco.png")

    plt.figure(2)
    plt.scatter(losses_critic)
    plt.xlabel("Episódio [N°]")
    plt.ylabel("Perda rede critico []")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure(3)
    plt.scatter(losses_actor)
    plt.xlabel("Episódio [N°]")
    plt.ylabel("Perda rede ator []")
    plt.grid()
    plt.legend()
    plt.show()


def setup_logging(logger: logging.Logger):
    # Stream handler for stdout with INFO level
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # File handler for file with DEBUG level
    file_handler = logging.FileHandler("log.log")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    logger = setup_logging(logger)

    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device {device} for calculation")

    rewards = []
    losses_critic = []
    losses_actor = []

    try:
        train_agent(rewards, losses_critic, losses_actor)
    except KeyboardInterrupt:
        logger.info("keyboard interrupt.")
        plot_results(rewards, losses_critic, losses_actor)
