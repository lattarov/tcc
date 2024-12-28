"""Defines the actor neural network-based controller."""

import torch
import numpy as np
from controllers.controller_abc import ControllerABC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCriticAgent(ControllerABC):
    """A neural network based controller.

    Uses the actor network previously trained in order to implement a
    controller.

    """

    def __init__(self, actor_model):
        """Initialize the actor model using the parameters file."""
        super().__init__()
        self.actor = torch.load(actor_model, weights_only=False)
        self.actor.eval()

    def control(self, states):
        """Calculate the control signal for a given state."""
        state_tensor = torch.FloatTensor(states).unsqueeze(0).to(DEVICE)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action
