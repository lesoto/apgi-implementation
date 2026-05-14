CONFIG = {
    # Initial states
    "S0": 0.0,
    "theta_0": 1.0,
    "theta_base": 1.0,
    "sigma2_e0": 1.0,
    "sigma2_i0": 1.0,
    # Numerical stability
    "eps": 1e-8,
    "pi_min": 0.01,
    "pi_max": 100.0,
    # EMA variance update
    "alpha_e": 0.05,
    "alpha_i": 0.05,
    # Variance estimation method
    "variance_method": "ema",  # "ema" or "sliding_window"
    "T_win": 50,  # Sliding window size (only used if variance_method == "sliding_window")
    # Neuromodulation — four transmitter systems (§8)
    # ACh  → Π_e gain:         Π_e_eff = g_ACh · Π_e           (exteroceptive precision)
    # NE   → Π_i gain:         Π_i_eff = g_NE  · Π_i           (interoceptive precision)
    # DA   → error bias:       z_i_eff = z_i + β_DA             (reward prediction error)
    # 5-HT → threshold offset: θ_eff   = θ    + β_5HT           (patience / uncertainty tolerance)
    "g_ach": 1.0,
    "g_ne": 1.0,
    "beta_5ht": 0.0,  # 5-HT patience offset; 0 = no serotonergic modulation (§8.4)
    # Somatic Bias Parameters (Notation Appendix v1.1 compliance)
    "BETA_DISCRETE": (0.3, 0.8),  # For discrete behavioral modeling/IGT
    "BETA_CONTINUOUS": (0.5, 2.5),  # For continuous ODE accumulation
    "BETA_AWAKE_AVG": 1.15,  # Canonical baseline for alert wakefulness
    "beta": 1.15,  # Alias for BETA_AWAKE_AVG (backward compatibility)
    "beta_da": 1.15,  # Spec-preferred name (alias for backward compatibility)
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
    # 9) Post-ignition signal reset (§6)
    "RHO_RETAIN": 0.1,  # Matches Spec §6.1; strong reset (retains 10%, clears 90%)
    "DELTA_RESET": 0.5,  # Baseline threshold boost post-ignition
    "reset_factor": 0.1,  # Alias for RHO_RETAIN (backward compatibility)
    # 10) Threshold Adaptation Timescales (Audit HIGH-4)
    "TAU_THETA_ALLOSTATIC": 20.0,  # 10-30s range; governing θ₀ (metabolic)
    "TAU_THETA_RECOVERY": 0.45,  # 300-600ms range; perceptual timescale (Level 1)
    # 11) Hierarchical Cascade Tuning
    "KAPPA_UP": 0.1,  # Bottom-up cascade strength (suppression)
    "KAPPA_DOWN": 0.1,  # Top-down coupling strength (PAC)
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
    # Signed Nonlinear Prediction Error Transform φ(ε) (§6)
    # α⁺, α⁻ ∈ [0.5, 2.0] — valence-specific amplitude gain
    # γ⁺, γ⁻ ∈ [1.0, 5.0] — saturation steepness
    # Symmetric defaults (α⁺=α⁻, γ⁺=γ⁻) recover the unsigned |ε| approximation
    "alpha_plus": 1.0,  # reward/approach gain
    "alpha_minus": 1.0,  # threat/avoidance gain
    "gamma_plus": 2.0,  # reward saturation slope
    "gamma_minus": 2.0,  # avoidance saturation slope
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
    # Cross-Level Threshold Resonance — Russian Doll Architecture (§8 / §9)
    # True = per-level S_l accumulators + θ_l = θ_{0,l}·(1+κ·Π_{l+1}·cos(φ_{l+1})) (spec §8)
    # False = single aggregated S, parallel accumulator fallback (non-affective / diagnostic only)
    # Has no effect when use_hierarchical=False.
    "use_resonance": True,
    "resonance_kappa_down": 0.1,  # Top-down coupling κ_down (threshold + phase entrainment)
    "resonance_kappa_up": 0.0,  # Bottom-up suppression after level-0 ignition
    "resonance_phi_noise_std": 0.0,  # Per-step phase jitter (biological noise)
    # Active Inference Action Loop (§19)
    "use_active_inference": False,  # Enable perception-action loop
    "ai_on_ignition_only": True,  # Fire only on Bₜ=1 (True) or every step (False)
    "ai_n_actions": 3,  # Number of discrete candidate policies K
    "ai_policy_precision": 2.0,  # Boltzmann sharpness γ_policy
    "ai_w_epistemic": 1.0,  # Weight on epistemic term of F(a)
    "ai_w_pragmatic": 1.0,  # Weight on pragmatic term of F(a)
    "ai_w_metabolic": 0.5,  # Weight on metabolic cost term
    "ai_sensory_feedback_rate": 0.1,  # Scale of channel-1 Δx̂ₑ (sensory)
    "ai_metabolic_feedback_rate": 0.1,  # Scale of channel-2 ΔM (interoceptive)
    "ai_precision_update_rate": 0.05,  # Scale of channel-3 ΔΣ (epistemic)
    "ai_action_params": None,  # None → use default 3-action table
}
