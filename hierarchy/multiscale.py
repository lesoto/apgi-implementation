import numpy as np


def compute_level_count(tau_min, tau_max, overlap):
    return int(np.log10(tau_max / tau_min) / np.log10(overlap))


def phase_signal(omega, t, phi0=0):
    return omega * t + phi0


def modulate_threshold(theta_0, pi_above, phi_above, k_down):
    return theta_0 * (1 + k_down * pi_above * np.cos(phi_above))


def bottom_up_cascade(theta, S_lower, theta_lower, k_up):
    if S_lower > theta_lower:
        return theta * (1 - k_up)
    return theta


def apply_reset_rule(S, theta, rho=0.1, delta=2.0):
    return S * rho, theta + delta
