"""Defines the actor network."""

import torch.nn as nn
import torch
import reinforcement_learning.environment as env


class ActorNetwork(nn.Module):
    """The actor neural network."""

    def __init__(self, state_dim, action_dim, max_action):
        """Define the topology of the NN."""
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(env.STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, env.ACTION_DIM)
        self.max_action = max_action

    def forward(self, state):
        """Forward propagation through the network."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))
