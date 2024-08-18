"""
Contains different controller implementations.

"""
import logging
import control.matlab
import cvxpy as cp
import numpy as np

from tools.model import sys


logger = logging.getLogger(__name__)


class PIDController():
    """
    Implements a classical PID controller.



    Attributes
    ----------
    gains : dict
        contains gains for proportional, integral and derivative gains.

    _error_integral : np.float64
        integral of error from start time until step N-1.

    _prev_error : np.float64
        error value at time step N-1.

    duration : int
        Total number of timesteps in the simulation.

    Exemples
    ----------
    gains = {"Kp" : 5, "Kd" : 3, "Ki" : 0.00005}
    controller = PIDController(gains)

    """

    def __init__(self, gains: dict) -> None:
        self.gains = gains
        self._error_integral = np.float64(0)
        self._prev_error = np.float64(0)

    def control_step(self, setpoint, measurement, dt):
        error = np.float64(setpoint - measurement)
        self._error_integral += np.float64(error * dt)
        error_derivative = np.float64((error - self._prev_error) / dt)
        self._prev_error = error

        return self.gains["Kp"] * error + \
            self.gains["Ki"] * self._error_integral + \
            self.gains["Kd"] * error_derivative


class MPCController():
    """
    Implements a model predictive controller.



    Attributes
    ----------
    q_matrix : np.diag
        TODO: Description of q_matrix.

    r_matrix : np.diag
        TODO: Description of r_matrix.

    sys : control.matlab.ss
        State space model of the system.

    x0 : np.array
        Array containing the initial positions for each state of the state
        space model to be controlled.

    duration : int
        Total number of timesteps in the simulation.

    Exemples
    ----------
    controller = MPCController(np.diag([1, 1, 10, 100]), np.array([[0.01r]),
                            np.array([0, 0, np.pi/4, 0]), 10)

    """

    def __init__(self, q_matrix: np.diag,
                 r_matrix: np.array,
                 sys: control.matlab.ss,
                 x0: np.array,
                 duration: int) -> None:
        self.q_matrix = q_matrix
        self.r_matrix = r_matrix
        self.sys = sys
        self.x0 = x0
        self.duration = duration

        self.u_max = 100  # Maximum control input (force)
        self.u_min = -100  # Minimum control input (force)

        self.x_mpc = np.zeros((duration, len(x0)))
        self.x_mpc[0] = x0
        self.u_mpc_array = []
        self.setpoints = []

        self._get_closed_loop_gain_matrix()
        self._get_closed_loop_state_matrix()
        self._get_closed_loop_model()

    def _get_closed_loop_gain_matrix(self, ):
        self.k_matrix, _, _ = control.matlab.lqr(self.sys.A,
                                                 self.sys.B,
                                                 self.q_matrix, self.r_matrix)

    def _get_closed_loop_state_matrix(self, ):
        self.a_matrix_closed_loop = self.sys.A - self.sys.B @ self.k_matrix

    def _get_closed_loop_model(self, ):
        self.closed_loop_model = control.matlab.ss(
            self.a_matrix_closed_loop, sys.B, sys.C, sys.D)

    def control_step(self, setpoint, i, dt, N=100,):
        """
        Calculates the control value for the Nth time step.

        Parameters
        ----------
        setpoint : np.float64
            Setpoint for the control loop.
        i : int
            Simulation step number.
        dt : int
            Time step duration in seconds.
        N : int = 10
            Prediction horizon for the MPC.

        WARNING: control steps should go up to duration-1 !

        Exemples
        ----------

        """

        # Define optimization variables
        x = cp.Variable((4, N+1))
        u = cp.Variable((1, N))

        # Define cost function
        cost = 0
        r_matrix = np.array([[0.01]])  # Input cost, not used.
        constraints = [x[:, 0] == self.x_mpc[i]]
        self.setpoints.append(np.array([0, setpoint, 0, 0]))
        for k in range(N):
            cost += cp.quad_form(x[:, k] - self.setpoints[i][0], self.q_matrix) + \
                cp.quad_form(u[:, k], r_matrix)
            constraints += [x[:, k+1] == self.sys.A @ x[:, k] + self.sys.B @ u[:, k],
                            self.u_min <= u[:, k], u[:, k] <= self.u_max]

        # Solve the optimization problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve()

        if prob.status != cp.OPTIMAL:
            message = f"""Optimization problem not solved optimally \n
            time step {i}, status: {prob.status}, \n N={N},\n
            Q={self.q_matrix} \n R={self.r_matrix} \n setpoint={setpoint}"""
            logger.error(message)
            raise Exception()

        # Apply the first control input
        u_mpc = u.value[:, 0]  # Get the first control input
        self.u_mpc_array.append(u_mpc)

        # Simulate the system for the next time step
        t_sim = [0, dt]
        u_sim = np.array([u_mpc, u_mpc])  # Control input for simulation
        _, x_next, _ = control.matlab.lsim(sys, u_sim, t_sim, self.x_mpc[i])
        self.x_mpc[i+1] = x_next[-1]
        logger.info(f"States : {self.x_mpc[i]}")
        logger.info(f"Control effort : {u_mpc}")
