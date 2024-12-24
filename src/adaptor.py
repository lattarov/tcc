"""Creates adaptors for the different controllers."""


class Adapter():
    """
    Adapter class to allow using different controllers (RL, PID, MPC, etc.) in the simulation



    Attributes
    ----------
    controller : type
        TODO: Description of controller.

    Exemples
    ----------


    """

    def __init__(self, controller):
        self._controller = controller
        self._max_threshold_action = 3
        self._min_threshold_action = -3

    def get_control(self, state:np.array):
        """calculates the control effort for a given state."""

        if isinstance(_controller, torch.nn.Module):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = _controller(state_tensor).detach().cpu().numpy()[0]
            action = np.clip(action, _min_threshold_action, _max_threshold_action)
            return action
