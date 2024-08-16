import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from control.matlab import ss, lsim, feedback, lqr, step

from tools.model import sys
from tools.controllers import PIDController
from tools.controllers import sys_mpc



# Define initial conditions
x0 = np.array([0, 0, np.pi/4, 0])  # Initial position and angle

# Define time span for the simulation
t = np.linspace(0, 300, 500)

# Simulate the closed-loop response with MPC and step input
_, y_mpc, x_mpc = lsim(sys_mpc, np.ones_like(t), t, x0)


################ PID #########################
# Define PID parameters and create a controller
gains = {
    "Kp" : 100
    "Ki" : 1
    "Kd" : 20

}
setpoint = 1  # Desired cart position after step input

pid_controller = PIDController(Kp, Ki, Kd, setpoint)

# Simulate the closed-loop response with PID and step input
x_pid = np.zeros((len(t), 4))
x_pid[0] = x0
dt = t[1] - t[0]
u_pid_array = []

for i in range(1, len(t)):
    u_pid = pid_controller.control(x_pid[i-1, 0], dt)
    u_pid_array.append(u_pid)
    _, x_next, _ = lsim(sys, [u_pid, u_pid], [0, dt], x_pid[i-1])
    x_pid[i] = x_next[-1]


# Plotting results
plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t, x_pid[:, 0], label='PID - Cart Position (m)')
plt.plot(t, x_mpc[:, 0], label='MPC - Cart Position (m)')
plt.title('Cart Position')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 2)
plt.plot(t, x_pid[:, 2], label='PID - Pendulum Angle (rad)')
plt.plot(t, x_mpc[:, 2], label='MPC - Pendulum Angle (rad)')
plt.title('Pendulum Angle')
plt.xlabel('Time (s)')
plt.ylabel('Angle (rad)')
plt.legend()
plt.grid()

plt.subplot(3, 1, 3)
plt.plot(t, x_pid[:, 1], label='PID - Cart Velocity (m/s)')
plt.plot(t, x_mpc[:, 1], label='MPC - Cart Velocity (m/s)')
plt.title('Cart Velocity')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
