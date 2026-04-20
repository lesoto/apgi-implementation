import numpy as np


def compute_precision(sigma: float) -> float:
    return 1.0 / (sigma**2 + 1e-8)


def compute_effective_interoceptive_precision(
    pi_baseline: float,
    beta: float,
    somatic_marker: float
) -> float:
    return pi_baseline * np.exp(beta * somatic_marker)


def update_precision_ode(
    pi,
    epsilon,
    pi_above,
    pi_below,
    tau_pi,
    alpha,
    c_down,
    c_up,
    psi_fn
):
    return (
        -pi / tau_pi
        + alpha * abs(epsilon)
        + c_down * (pi_above - pi)
        + c_up * psi_fn(epsilon)
    )
