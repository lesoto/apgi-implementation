import numpy as np


def update_signal_ode(S, z_e, z_i, pi_e, pi_i, beta, tau_s, noise_std=0.01):
    noise = np.random.normal(0, noise_std)
    return -S / tau_s + pi_e * abs(z_e) + beta * pi_i * abs(z_i) + noise
