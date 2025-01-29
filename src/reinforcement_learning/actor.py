"""Defines the actor network."""

import torch.nn as nn
import torch


class ActorNetwork(nn.Module):
    """The actor neural network."""

    def __init__(self):
        """Define the topology of the NN."""
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(env.STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, env.ACTION_DIM)
        self.max_action = env.MAX_ACTION

    def forward(self, state):
        """Forward propagation through the network."""
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.max_action * torch.tanh(self.fc3(x))
