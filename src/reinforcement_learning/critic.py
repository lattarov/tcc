"""The critic network."""

import torch.nn as nn
import torch

import environment as env


class CriticNetwork(nn.Module):
    """The critic neural network."""

    def __init__(self):
        """Define the topology of the NN."""
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(env.STATE_DIM + env.ACTION_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, 1)

    def forward(self, state, action):
        """Forward propagation through the network."""
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        return self.fc6(x)
