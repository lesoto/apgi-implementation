import numpy as np


def power_spectrum(theta_series, tau_levels, sigma_levels, freqs):
    S = np.zeros_like(freqs)
    for tau, sigma in zip(tau_levels, sigma_levels):
        S += (sigma**2 * tau**2) / (1 + (2 * np.pi * freqs * tau) ** 2)
    return S


def hurst_from_slope(beta_spec):
    return (beta_spec + 1) / 2
