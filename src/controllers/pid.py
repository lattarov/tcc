"""Implements a PID controller."""

import numpy as np
import controllers


class PIDController(controllers.controller_abc.ControllerABC):
    """A PID controller."""

    def __init__(self):
        """Initialize the error counter, feedback gains and last error."""
        super().__init__()
        self.kp_theta = 5
        self.kd_theta = 10
        self.ki_theta = 0
        self.last_error_theta = 0
        self.total_error_theta = 0

        self.kp_x = 1
        self.kd_x = 10
        self.ki_x = 0
        self.last_error_x = 0
        self.total_error_x = 0

    def control(self, states):
        """Calculate the control signal for a given state."""
        error_theta = states[1] - 0
        d_error = error_theta - self.last_error_theta
        self.last_error_theta = error_theta
        self.total_error_theta = self.total_error_theta + error_theta
        action_theta = self.kp_theta * error_theta + self.kd_theta * d_error + self.ki_theta * self.total_error_theta
        action_theta = np.clip(action_theta, -self.max_action, self.max_action)

        error_x = states[0] - 0
        d_error = error_x - self.last_error_x
        self.last_error_x = error_x
        self.total_error_x = self.total_error_x + error_x
        action_x = self.kp_x * error_x + self.kd_x * d_error + self.ki_x * self.total_error_x
        action_x = np.clip(action_x, -self.max_action, self.max_action)


        return np.array([action_theta + action_x])
