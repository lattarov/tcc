"""Implements a PID controller."""

import numpy as np
import controllers


class PIDController(controllers.controller_abc.ControllerABC):
    """A PID controller."""

    def __init__(self):
        """Initialize the error counter, feedback gains and last error."""
        super().__init__()
        self.kp = 1
        self.kd = 0.01
        self.ki = 0
        self.last_error = 0
        self.total_error = 0

    def control(self, ref, angle):
        """Calculate the control signal for a given state."""
        error = angle - ref
        d_error = error - self.last_error
        self.last_error = error
        self.total_error = self.total_error + error
        action = self.kp * error + self.kd * d_error + self.ki*self.total_error
        action = np.clip(action, -self.max_action, self.max_action)
        return action
