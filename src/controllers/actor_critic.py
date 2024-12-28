import torch
import numpy as np
from controllers.controller_abc import ControllerABC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCriticAgent(ControllerABC):
    def __init__(self, actor_model):
        super().__init__()
        self.actor = torch.load(actor_model, weights_only=False)
        self.actor.eval()

    def control(self, states):
        state_tensor = torch.FloatTensor(states).unsqueeze(0).to(device)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        action = np.clip(action, -self.max_action, self.max_action)
        return action
