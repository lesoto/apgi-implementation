import numpy as np


class LiquidNetwork:
    def __init__(self, n_units=500):
        self.n = n_units
        self.W_res = np.random.randn(n_units, n_units) * 0.1
        self.W_in = np.random.randn(n_units) * 0.1
        self.x = np.zeros(n_units)

    def step(self, u, tau, activation=np.tanh):
        dx = -self.x / tau + activation(self.W_res @ self.x + self.W_in * u)
        self.x += dx
        return self.x

    def readout_signal(self):
        return np.dot(self.x, self.x)

    def apply_suprathreshold_gain(self, S, theta, A=1.0):
        if S > theta:
            self.x += A * self.x * max(0, S - theta)
