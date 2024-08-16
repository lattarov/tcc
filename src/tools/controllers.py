"""
Contains different controller implementations.
"""

import numpy as np

from control.matlab import ss, lsim, feedback, lqr, step
from tools.model import sys

# Implement PID controller manually
class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0
        self.prev_error = 0

    def control(self, measurement, dt):
        error = self.setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        self.prev_error = error
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative



# Model Predictive Control (MPC) using LQR
Q = np.diag([1, 1, 10, 100])  # State cost
R = np.array([[0.01]])  # Input cost

K, _, _ = lqr(sys.A, sys.B, Q, R)

# Closed-loop system with LQR (MPC approximation)
A_cl = sys.A - sys.B @ K
sys_mpc = ss(A_cl, sys.B, sys.C, sys.D)
