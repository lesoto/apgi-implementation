import numpy as np
from collections import deque


class RunningStats:
    window: deque[float]

    def __init__(self, window_size: int):
        self.window = deque(maxlen=window_size)

    def update(self, value: float):
        self.window.append(value)

    def mean(self):
        return np.mean(self.window) if self.window else 0.0

    def std(self):
        return np.std(self.window) if self.window else 1.0


def compute_prediction_error(x: float, x_hat: float) -> float:
    return x - x_hat


def z_score(epsilon: float, stats: RunningStats) -> float:
    mu = stats.mean()
    sigma = stats.std()
    return (epsilon - mu) / (sigma + 1e-8)
