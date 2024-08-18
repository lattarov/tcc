import logging
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import lsim

from tools.model import sys
from tools.controllers import PIDController
from tools.controllers import MPCController


logging.basicConfig(
    level=logging.INFO,
    handlers=[logging.StreamHandler(),
              logging.FileHandler('tcc.log')],
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y-%H:%M:%S'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Define initial conditions
    x0 = np.array([0, 0, np.pi/4, 0])  # Initial position and angle

    # Define time span for the simulation
    t = np.linspace(0, 10, 10000)
    dt = t[1] - t[0]
    print(dt)

    # PID gains
    gains = {
        "Kp": 100,
        "Ki": 1,
        "Kd": 20
    }

    # MPC cost matrices
    Q = np.diag([1, 1, 1, 10])  # State cost
    R = np.array([[0.01]])  # Input cost

    setpoint = 1  # Desired cart position after step input

    pid_controller = PIDController(gains)
    mpc_controller = MPCController(Q, R, sys, x0, len(t))

    # Simulate the closed-loop response with PID and step input
    x_pid = np.zeros((len(t), 4))
    x_pid[0] = x0
    u_pid_array = []

    for i in range(len(t) - 1):
        logger.info(f"Simulation step {i+1}/{len(t)}")
        mpc_controller.control_step(setpoint, i, dt)

        u_pid = pid_controller.control_step(setpoint, x_pid[i, 0], dt)
        u_pid_array.append(u_pid)
        _, x_next, _ = lsim(sys, [u_pid, u_pid], [0, dt], x_pid[i])
        x_pid[i] = x_next[-1]

    logger.info("Plotting results")

    # Plotting results
    plt.figure(figsize=(12, 10))

    plt.subplot(3, 1, 1)
    # plt.plot(t, mpc_controller.setpoints[:][0], label='setpoint')
    plt.plot(t, x_pid[:, 0], label='PID - Cart Position (m)')
    plt.plot(t, mpc_controller.x_mpc[:, 1], label='MPC - Cart Position (m)')
    plt.title('Cart Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Position (m)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 2)
    plt.plot(t, x_pid[:, 2], label='PID - Pendulum Angle (rad)')
    plt.plot(t, mpc_controller.x_mpc[:, 2], label='MPC - Pendulum Angle (rad)')
    plt.title('Pendulum Angle')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    plt.legend()
    plt.grid()

    plt.subplot(3, 1, 3)
    plt.plot(t, x_pid[:, 1], label='PID - Cart Velocity (m/s)')
    plt.plot(t, mpc_controller.x_mpc[:, 1], label='MPC - Cart Velocity (m/s)')
    plt.title('Cart Velocity')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity (m/s)')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()
