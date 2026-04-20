import numpy as np

K_B = 1.380649e-23


def metabolic_cost(kappa, bits):
    return kappa * bits


def landauer_limit(T):
    return K_B * T * np.log(2)
