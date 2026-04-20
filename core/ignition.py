import numpy as np


def compute_ignition_probability(S, theta, alpha=3.0):
    return 1.0 / (1.0 + np.exp(-alpha * (S - theta)))


def detect_ignition_event(S, theta):
    return S > theta


def compute_margin(S, theta):
    return S - theta
