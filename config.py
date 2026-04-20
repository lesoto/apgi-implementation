CONFIG = {
    # Initial states
    "S0": 0.0,
    "theta_0": 1.0,
    "theta_base": 1.0,
    "sigma2_e0": 1.0,
    "sigma2_i0": 1.0,
    # Numerical stability
    "eps": 1e-8,
    "pi_min": 1e-4,
    "pi_max": 1e4,
    # EMA variance update
    "alpha_e": 0.05,
    "alpha_i": 0.05,
    # Neuromodulation
    "g_ach": 1.0,
    "g_ne": 1.0,
    "beta": 0.0,
    # Prevent NE double-counting (recommended)
    "ne_on_precision": True,
    "ne_on_threshold": False,
    # NE modulation strength (use gamma_ne<=0.01 when ne_on_threshold=True)
    "gamma_ne": 0.1,
    # Threshold decay rate (use kappa>=0.15 when ne_on_threshold=True)
    "kappa": 0.15,
    # Signal accumulation
    "lam": 0.2,
    "signal_log_nonlinearity": True,
    # Threshold update + refractory dynamics
    "eta": 0.1,
    "delta": 0.5,
    # Cost-value model
    "use_realistic_cost": True,
    "c0": 0.0,
    "c1": 0.2,
    "c2": 0.5,
    "v1": 0.5,
    "v2": 0.5,
    # Ignition dynamics
    "ignite_tau": 0.5,
    "stochastic_ignition": False,
    # Multi-scale recommendation parameter
    "timescale_k": 1.6,
}
