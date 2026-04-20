import numpy as np


def integrate_euler_maruyama(x, mu, sigma, dt):
    noise = np.random.normal(0, 1)
    return x + mu * dt + sigma * np.sqrt(dt) * noise
