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
    # Variance estimation method
    "variance_method": "ema",  # "ema" or "sliding_window"
    "T_win": 50,  # Sliding window size (only used if variance_method == "sliding_window")
    # Neuromodulation
    "g_ach": 1.0,
    "g_ne": 1.0,
    "beta": 0.0,  # Spec alias: beta_da
    "beta_da": 0.0,  # Spec-preferred name (alias for backward compatibility)
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
    "use_canonical_discrete_mode": False,  # Use discrete leaky accumulation instead of ODE
    # Threshold update + refractory dynamics
    "eta": 0.1,
    "delta": 0.5,
    # Post-ignition reset (§6)
    "reset_factor": 0.5,  # Signal reset factor ρ ∈ (0,1)
    # Cost-value model
    "use_realistic_cost": True,
    "c0": 0.0,
    "c1": 0.2,
    "c2": 0.5,
    "v1": 0.5,
    "v2": 0.5,
    # Ignition dynamics
    "ignite_tau": 0.5,  # Spec alias: tau_sigma
    "tau_sigma": 0.5,  # Spec-preferred name (alias for backward compatibility)
    "stochastic_ignition": False,
    # Continuous-time signal ODE / SDE parameters
    "tau_s": 5.0,  # Signal decay time constant (dt/tau_s ≈ lam)
    "dt": 0.5,  # Integration time step (Spec §7.4: dt ≤ min(τ_S, τ_θ, τ_Π) / 10)
    "noise_std": 0.01,  # SDE diffusion coefficient
    # Generative model dynamics (§1.4)
    "use_internal_predictions": True,
    "kappa_e": 0.01,
    "kappa_i": 0.01,
    # Multi-scale recommendation parameter
    "timescale_k": 1.6,
    # Thermodynamic constraints (§11)
    "use_thermodynamic_cost": False,  # Enable Landauer's principle
    "k_boltzmann": 1.38e-23,  # Boltzmann constant (J/K)
    "T_env": 310.0,  # Environmental temperature (K, body temp)
    "kappa_meta": 1.0,  # Metabolic efficiency factor
    "kappa_units": "dimensionless",  # "dimensionless" or "joules_per_bit"
    # BOLD fMRI calibration for energy conversion
    "use_bold_calibration": False,  # Enable BOLD-to-Joule conversion
    "bold_conversion_factor": 1.2e-18,  # Joules per 1% BOLD change per cm³ tissue
    "bold_tissue_volume": 1.0,  # Tissue volume in cm³
    "bold_ignition_spike_factor": 1.075,  # 7.5% energy spike during ignition (5-10% range)
    # Reservoir layer (§10)
    "use_reservoir": False,  # Enable reservoir computing layer
    "reservoir_size": 100,  # Number of reservoir units
    "reservoir_tau": 1.0,  # Base time constant (ms)
    "reservoir_spectral_radius": 0.9,  # Spectral radius (must be < 1)
    "reservoir_input_scale": 0.1,  # Input weight scaling
    "reservoir_readout_method": "linear",  # "linear" or "energy"
    "reservoir_amplification": 0.0,  # Suprathreshold amplification strength
    "reservoir_ridge_alpha": 1e-6,  # Ridge regression regularization
    # Kuramoto oscillators (§9)
    "use_kuramoto": False,  # Enable Kuramoto oscillators with phase noise
    "kuramoto_tau_xi": 1.0,  # OU noise correlation timescale (ms)
    "kuramoto_sigma_xi": 0.1,  # OU noise amplitude (rad/ms)
    "kuramoto_reset_amount": 3.14159,  # Phase reset on ignition (radians, default π)
    # Observable mapping (§14)
    "use_observable_mapping": False,  # Enable neural/behavioral observable extraction
    # Stability analysis (§7)
    "use_stability_analysis": False,  # Enable fixed-point stability analysis
}
