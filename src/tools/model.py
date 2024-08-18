"""
Describes the LTI model of the pole on a cart system.
"""
import logging
import numpy as np
from control.matlab import ss

logger = logging.getLogger(__name__)

# System parameters
m = 0.1  # Mass of the pendulum (kg)
M = 1.0  # Mass of the cart (kg)
L = 1.0  # Length of the pendulum (m)
g = 9.81  # Acceleration due to gravity (m/s^2)
d = 0.1  # Damping coefficient (N*m*s)

# State-space representation
A = np.array([[0, 1, 0, 0],
              [0, -d/M, m*g/M, 0],
              [0, 0, 0, 1],
              [0, -d/(M*L), (M+m)*g/(M*L), 0]])

B = np.array([[0],
              [1/M],
              [0],
              [1/(M*L)]])

C = np.eye(4)
D = np.zeros((4, 1))

# Create the state-space model
sys = ss(A, B, C, D)

logger.info(ss)
