import logging
import numpy as np
import matplotlib.pyplot as plt
from control.matlab import lsim

from tools.model import sys
from tools.controllers import PIDController
from tools.controllers import MPCController


logging.basicConfig(
    level=logging.DEBUG,
    handlers=[logging.StreamHandler(),
              logging.FileHandler('tcc.log')],
    format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
    datefmt='%d/%m/%Y-%H:%M:%S'
)

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Define initial conditions
    init_pos = np.float32(0)
    init_speed = np.float32(0)
    init_angular_position = np.pi/4
    init_angular_velocity = np.float32(0)

    x0 = np.array([init_pos, init_speed, init_angular_position,
                   init_angular_velocity])

    logger.info(f"initial position [m] {init_pos}")
    logger.info(f"initial speed [m/s] {init_speed}")
    logger.info(f"initial angular position [rad] {init_angular_position}")
    logger.info(f"initial angular velocity [rad/s] {init_angular_velocity}")

    # Define time span for the simulation
    t = np.linspace(0, 1, 100)
    dt = t[1] - t[0]

    logger.debug(f"time step size for simulation [s]: {dt}")

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
