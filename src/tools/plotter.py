

class Plotter:
    def __init__(self, pid:PIDController, mpc: MPCController):
        self.pid = pid
        self.mpc = mpc

    def plot_pid(self, gains: dict):
