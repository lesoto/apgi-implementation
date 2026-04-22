from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.dynamics import (
    compute_precision_coupled_noise_std,
    update_prediction,
    update_signal_ode,
)
from core.signal import integrate_signal_leaky
from core.allostatic import allostatic_threshold_ode
from core.ignition import (
    compute_ignition_probability,
    detect_ignition_event,
    sample_ignition_state,
)
from core.precision import (
    apply_ach_gain,
    apply_dopamine_bias_to_error,
    apply_ne_gain,
    compute_interoceptive_precision_exponential,
    compute_precision,
    update_mean_ema,
    update_variance_ema,
)
from hierarchy.coupling import HierarchicalPrecisionNetwork
from hierarchy.multiscale import (
    aggregate_multiscale_signal,
    build_timescales,
    multiscale_weights,
    update_multiscale_feature,
)
from core.preprocessing import compute_prediction_error, RunningStats
from core.signal import instantaneous_signal, stabilize_signal_log
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    compute_information_value,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
)
from core.thermodynamics import compute_landauer_cost
from core.validation import validate_config, ValidationError
from stats.hurst import estimate_hurst_robust
from stats.spectral_model import validate_pink_noise


@dataclass
class PrecisionState:
    sigma2_e: float
    sigma2_i: float
    pi_e: float = 1.0
    pi_i: float = 1.0
    mu_e: float = field(default=0.0)  # EMA mean for exteroceptive errors
    mu_i: float = field(default=0.0)  # EMA mean for interoceptive errors


@dataclass
class HierarchicalState:
    """State for hierarchical precision and threshold coupling."""

    n_levels: int = 1
    pis: list[float] = None  # type: ignore[assignment]
    thetas: list[float] = None  # type: ignore[assignment]
    phases: list[float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.pis is None:
            self.pis = [1.0] * self.n_levels
        if self.thetas is None:
            self.thetas = [1.0] * self.n_levels
        if self.phases is None:
            self.phases = [0.0] * self.n_levels


class APGIPipeline:
    """APGI one-step update implementing the full corrected mathematical pipeline."""

    M: float
    history: dict[str, list[float]]
    mu_e_levels: np.ndarray
    mu_i_levels: np.ndarray
    sigma2_e_levels: np.ndarray
    sigma2_i_levels: np.ndarray
    n_levels: int
    taus: np.ndarray

    def _apply_hierarchical_preset(self, config: dict) -> dict:
        """Apply hierarchical mode preset to configuration.

        Simplifies hierarchical system configuration by allowing users to set
        a single 'hierarchical_mode' parameter instead of three separate flags.

        Modes:
            - 'off': Disable all hierarchical features
            - 'basic': Enable hierarchical only (use_hierarchical=True)
            - 'advanced': Enable hierarchical + precision ODE (use_hierarchical=True,
                         use_hierarchical_precision_ode=True)
            - 'full': Enable all hierarchical features (use_hierarchical=True,
                     use_hierarchical_precision_ode=True, use_phase_modulation=True)

        Args:
            config: Configuration dictionary

        Returns:
            Updated configuration with hierarchical flags set

        Raises:
            ValueError: If hierarchical_mode has unknown value
        """
        mode = config.get("hierarchical_mode", None)

        # Only apply preset if hierarchical_mode is explicitly set
        if mode is None:
            return config

        if mode == "off":
            config.update(
                {
                    "use_hierarchical": False,
                    "use_hierarchical_precision_ode": False,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "basic":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_hierarchical_precision_ode": False,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "advanced":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_hierarchical_precision_ode": True,
                    "use_phase_modulation": False,
                }
            )
        elif mode == "full":
            config.update(
                {
                    "use_hierarchical": True,
                    "use_hierarchical_precision_ode": True,
                    "use_phase_modulation": True,
                }
            )
        else:
            raise ValueError(
                f"Unknown hierarchical_mode: {mode}. Must be one of: 'off', 'basic', 'advanced', 'full'"
            )

        return config

    def _compute_per_level_errors(
        self,
        epsilon_e: float,
        epsilon_i: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute per-level z-scores for hierarchical system.

        Spec §7: Each level ℓ processes errors at its own timescale τ_ℓ.
        This version is fully vectorized for performance (§4.1).

        Args:
            epsilon_e: Exteroceptive prediction error
            epsilon_i: Interoceptive prediction error

        Returns:
            (z_e_levels, z_i_levels) — z-scores at each level (np.ndarray)
        """
        if not self.use_hierarchical:
            # Single-scale case
            z_e = (epsilon_e - self.state.mu_e) / (
                self.state.sigma2_e**0.5 + self.config["eps"]
            )
            z_i = (epsilon_i - self.state.mu_i) / (
                self.state.sigma2_i**0.5 + self.config["eps"]
            )
            return np.array([z_e]), np.array([z_i])

        # Multi-scale: fully vectorized EMA updates
        alphas = 1.0 / self.taus

        # Vectorized mean updates
        self.mu_e_levels = (1.0 - alphas) * self.mu_e_levels + alphas * epsilon_e
        self.mu_i_levels = (1.0 - alphas) * self.mu_i_levels + alphas * epsilon_i

        # Vectorized variance updates (centered)
        self.sigma2_e_levels = (1.0 - alphas) * self.sigma2_e_levels + alphas * (
            epsilon_e - self.mu_e_levels
        ) ** 2
        self.sigma2_i_levels = (1.0 - alphas) * self.sigma2_i_levels + alphas * (
            epsilon_i - self.mu_i_levels
        ) ** 2

        # Vectorized z-scores
        z_e_levels = (epsilon_e - self.mu_e_levels) / (
            np.sqrt(self.sigma2_e_levels) + self.config["eps"]
        )
        z_i_levels = (epsilon_i - self.mu_i_levels) / (
            np.sqrt(self.sigma2_i_levels) + self.config["eps"]
        )

        return z_e_levels, z_i_levels

    def __init__(self, config: dict):
        # Copy to avoid mutating the caller's dict
        self.config = dict(config)

        # Apply hierarchical mode preset if specified
        self.config = self._apply_hierarchical_preset(self.config)

        # Support spec-preferred parameter names with backward compatibility
        if "beta_da" in self.config and "beta" not in self.config:
            self.config["beta"] = self.config["beta_da"]
        if "tau_sigma" in self.config and "ignite_tau" not in self.config:
            self.config["ignite_tau"] = self.config["tau_sigma"]

        # Validate configuration against spec constraints (§15)
        try:
            validate_config(self.config)
        except ValidationError as e:
            import warnings

            warnings.warn(
                f"Configuration validation failed: {e}. "
                "Some constraints may be violated. Spec §15.",
                RuntimeWarning,
                stacklevel=2,
            )

        # Validate NE configuration to prevent double-counting
        if self.config.get("ne_on_precision", False) and self.config.get(
            "ne_on_threshold", False
        ):
            raise ValidationError(
                "Both ne_on_precision and ne_on_threshold are True. "
                "This double-counts norepinephrine effects. "
                "Enable only one. See spec Section 2.3-2.4."
            )
        # Auto-adjust NE parameters for threshold mode to prevent runaway
        if (
            self.config.get("ne_on_threshold", False)
            and self.config.get("gamma_ne", 0.1) >= 0.1
        ):
            import warnings

            warnings.warn(
                "ne_on_threshold=True with gamma_ne>=0.1 causes threshold instability. "
                "Auto-adjusting: gamma_ne 0.1 → 0.01, kappa 0.02 → 0.15. "
                "To suppress this warning, set gamma_ne<=0.01 or kappa>=0.15.",
                RuntimeWarning,
                stacklevel=2,
            )
            self.config["gamma_ne"] = 0.01
            self.config["kappa"] = 0.15

        self.S = float(self.config["S0"])
        self.theta = float(self.config["theta_0"])
        self.theta_dot = 0.0  # For continuous ODE tracking
        # Internalized predictions per §1.4 if enabled
        self.x_hat_e = float(self.config.get("x_hat_e0", 0.0))
        self.x_hat_i = float(self.config.get("x_hat_i0", 0.0))
        self.state = PrecisionState(
            sigma2_e=float(self.config["sigma2_e0"]),
            sigma2_i=float(self.config["sigma2_i0"]),
        )
        self.B_prev = 0
        self.t = 0.0  # Continuous time (phase tracking + SDE)

        # Initialize sliding-window stats if variance_method is "sliding_window"
        self.stats_e = None
        self.stats_i = None
        if self.config.get("variance_method", "ema") == "sliding_window":
            T_win = self.config.get("T_win", 50)
            self.stats_e = RunningStats(window_size=T_win)
            self.stats_i = RunningStats(window_size=T_win)

        # Hierarchical state if enabled
        self.hierarchical = None
        self.hierarchical_network = None
        self.use_hierarchical = self.config.get("use_hierarchical", False)
        if self.use_hierarchical:
            n_levels = self.config.get("n_levels", 3)
            self.hierarchical = HierarchicalState(n_levels=n_levels)

            # Per-level statistics for true hierarchical error processing
            self.n_levels = n_levels
            self.taus = build_timescales(
                self.config.get("tau_0", 10.0), self.config.get("k", 1.6), n_levels
            )
            self.mu_e_levels = np.zeros(n_levels)
            self.mu_i_levels = np.zeros(n_levels)
            self.sigma2_e_levels = np.ones(n_levels)
            self.sigma2_i_levels = np.ones(n_levels)

            # Initialize hierarchical feature states for signal aggregation
            self.phi_e_levels = np.zeros(n_levels)
            self.phi_i_levels = np.zeros(n_levels)
            self.weights = multiscale_weights(n_levels, self.config.get("k", 1.6))

            # Initialize oscillatory phases for threshold modulation (basic mode)
            self.phase_levels = np.zeros(n_levels)
            self.omega_levels = (
                2 * np.pi / self.taus
            )  # Natural frequencies from timescales

            # Initialize HierarchicalPrecisionNetwork for proper per-level error computation
            # Enable for advanced mode, and also for basic mode
            if self.config.get(
                "use_hierarchical_precision_ode", False
            ) or self.config.get("use_hierarchical", False):
                self.hierarchical_network = HierarchicalPrecisionNetwork(
                    n_levels=n_levels,
                    tau_pi=self.config.get("tau_pi", 1000.0),
                    C_down=self.config.get("C_down", 0.1),
                    C_up=self.config.get("C_up", 0.05),
                )
        else:
            # Single-scale: initialize single z-score pair
            self.n_levels = 1
            self.taus = np.array([1.0])
            self.mu_e_levels = np.zeros(1)
            self.mu_i_levels = np.zeros(1)
            self.sigma2_e_levels = np.ones(1)
            self.sigma2_i_levels = np.ones(1)

        # Somatic marker state if enabled
        self.M = self.config.get("M_somatic", 0.0)  # Somatic marker ∈ [-2, +2]

        # Reservoir layer if enabled (§10)
        self.reservoir = None
        if self.config.get("use_reservoir", False):
            from reservoir.liquid_state_machine import LiquidStateMachine

            self.reservoir = LiquidStateMachine(
                N=self.config.get("reservoir_size", 100),
                M=2,  # [z_e, z_i]
                tau_res=self.config.get("reservoir_tau", 1.0),
                spectral_radius=self.config.get("reservoir_spectral_radius", 0.9),
                input_scale=self.config.get("reservoir_input_scale", 0.1),
            )

        # Kuramoto oscillators if enabled (§9)
        self.kuramoto = None
        if self.config.get("use_kuramoto", False):
            from oscillation.kuramoto import HierarchicalKuramotoSystem

            n_levels = self.config.get("n_levels", 3)
            self.kuramoto = HierarchicalKuramotoSystem(
                n_levels=n_levels,
                config=self.config,
            )

        # Observable mapping if enabled (§14)
        self.neural_observables = None
        self.behavioral_observables = None
        self.prediction_validator = None
        if self.config.get("use_observable_mapping", False):
            from validation.observable_mapping import (
                NeuralObservableExtractor,
                BehavioralObservableExtractor,
                KeyTestablePredictionValidator,
            )

            self.neural_observables = NeuralObservableExtractor()
            self.behavioral_observables = BehavioralObservableExtractor()
            self.prediction_validator = KeyTestablePredictionValidator(
                tau_sigma=self.config.get("ignite_tau", 0.5)
            )

        # Stability analyzer if enabled (§7)
        self.stability_analyzer = None
        if self.config.get("use_stability_analysis", False):
            from analysis.stability import StabilityAnalyzer

            self.stability_analyzer = StabilityAnalyzer(self.config)

        self.history = {
            "S": [],
            "theta": [],
            "B": [],
        }

        # Thermodynamic cost history if enabled (§11)
        if self.config.get("use_thermodynamic_cost", False):
            self.history["C_landauer"] = []
            self.history["bits_erased"] = []

    def step(
        self,
        x_e: float,
        x_i: float,
        x_hat_e: float | None = None,
        x_hat_i: float | None = None,
    ):
        # Use provided predictions or internalized ones per §1.4
        _x_hat_e = x_hat_e if x_hat_e is not None else self.x_hat_e
        _x_hat_i = x_hat_i if x_hat_i is not None else self.x_hat_i

        # 1) Raw prediction errors
        z_e = compute_prediction_error(x_e, _x_hat_e)
        z_i = compute_prediction_error(x_i, _x_hat_i)

        # 2) Online mean + variance update (EMA or sliding-window, centered)
        variance_method = self.config.get("variance_method", "ema")

        if variance_method == "sliding_window" and self.stats_e is not None:
            # Use sliding-window statistics
            assert self.stats_i is not None  # Type guard for mypy
            self.stats_e.update(z_e)
            self.stats_i.update(z_i)
            self.state.mu_e = self.stats_e.mean()
            self.state.mu_i = self.stats_i.mean()
            # Use Bessel correction for unbiased estimation with small windows
            self.state.sigma2_e = self.stats_e.variance(bessel_correction=True)
            self.state.sigma2_i = self.stats_i.variance(bessel_correction=True)
        else:
            # Use EMA (default)
            self.state.mu_e = update_mean_ema(
                self.state.mu_e, z_e, self.config["alpha_e"]
            )
            self.state.mu_i = update_mean_ema(
                self.state.mu_i, z_i, self.config["alpha_i"]
            )
            self.state.sigma2_e = update_variance_ema(
                self.state.sigma2_e, z_e, self.state.mu_e, self.config["alpha_e"]
            )
            self.state.sigma2_i = update_variance_ema(
                self.state.sigma2_i, z_i, self.state.mu_i, self.config["alpha_i"]
            )

        # 3) Proper z-score normalization: (z - μ) / (σ + ε)
        eps = self.config["eps"]
        z_e_n = (z_e - self.state.mu_e) / (self.state.sigma2_e**0.5 + eps)
        z_i_n = (z_i - self.state.mu_i) / (self.state.sigma2_i**0.5 + eps)

        # 4) Precision with clamping
        pi_e = compute_precision(
            self.state.sigma2_e,
            self.config["eps"],
            self.config["pi_min"],
            self.config["pi_max"],
        )
        pi_i = compute_precision(
            self.state.sigma2_i,
            self.config["eps"],
            self.config["pi_min"],
            self.config["pi_max"],
        )
        self.state.pi_e = pi_e
        self.state.pi_i = pi_i

        # 5) Neuromodulation (+ dopamine correction)
        pi_e_eff = apply_ach_gain(pi_e, self.config["g_ach"])

        # Interoceptive precision: exponential somatic form or linear NE gain
        if self.config.get("use_somatic_precision", False):
            pi_i_eff = compute_interoceptive_precision_exponential(
                pi_i,
                self.config.get("beta_somatic", 0.3),
                self.M,
                self.config["pi_min"],
                self.config["pi_max"],
            )
        elif self.config.get("ne_on_precision", False):
            pi_i_eff = apply_ne_gain(pi_i, self.config["g_ne"])
        else:
            pi_i_eff = pi_i

        z_i_eff = apply_dopamine_bias_to_error(z_i_n, self.config["beta"])

        # 5b) Hierarchical precision ODE if enabled
        if self.hierarchical_network is not None and self.config.get(
            "use_hierarchical_precision_ode", False
        ):
            dt = self.config.get("dt", 1.0)
            # Compute per-level z-scores for both channels via vectorized EMA (§7)
            z_e_levels, z_i_levels = self._compute_per_level_errors(z_e, z_i)

            # Full precision cascade: combine exteroceptive + interoceptive errors
            # per level so cross-level error propagation is complete (§7, §8.4)
            combined_epsilon = np.abs(z_e_levels) + np.abs(z_i_levels)

            # Step the hierarchical network using actual combined per-level errors
            pis_new, phi_new = self.hierarchical_network.step(
                epsilon_new=combined_epsilon,
                dt=dt,
                alpha_gain=self.config.get("alpha_gain", 0.1),
            )

            # Update hierarchical state with both precision and phase
            assert self.hierarchical is not None  # Type guard for mypy
            self.hierarchical.pis = pis_new.tolist()
            self.hierarchical.phases = phi_new.tolist()

        # 5c) Update hierarchical feature states for signal aggregation
        if self.use_hierarchical:
            dt = self.config.get("dt", 1.0)
            for level in range(self.n_levels):
                # Update exteroceptive and interoceptive features at each timescale
                self.phi_e_levels[level] = update_multiscale_feature(
                    self.phi_e_levels[level], z_e_n, self.taus[level]
                )
                self.phi_i_levels[level] = update_multiscale_feature(
                    self.phi_i_levels[level], z_i_eff, self.taus[level]
                )

        # 6) Instantaneous signal (diagnostic) + SDE integration via ODE drift
        # Use hierarchical aggregation if enabled, otherwise single-scale
        if self.use_hierarchical:
            # Aggregate multi-scale signal: S = Σ_i w_i Π_i |Φ_i|
            # Combine exteroceptive and interoceptive features
            phi_combined = np.abs(self.phi_e_levels) + np.abs(self.phi_i_levels)

            # Use hierarchical precision if available, otherwise compute from per-level variance
            if self.hierarchical_network is not None:
                pi_levels = self.hierarchical_network.pi
            else:
                # Basic hierarchical mode: compute per-level precision from variance estimates
                # This creates multi-scale precision structure even without precision ODE
                pi_levels = np.array(
                    [
                        compute_precision(
                            self.sigma2_e_levels[i] + self.sigma2_i_levels[i],
                            self.config["eps"],
                            self.config["pi_min"],
                            self.config["pi_max"],
                        )
                        for i in range(self.n_levels)
                    ]
                )

            S_inst = aggregate_multiscale_signal(phi_combined, pi_levels, self.weights)
        else:
            S_inst = instantaneous_signal(z_e_n, z_i_eff, pi_e_eff, pi_i_eff)

        # Wire dynamics.py (update_signal_ode) OR discrete leaky accumulation
        # Spec §7.3: σ_S = 1/sqrt(Π_e^eff + Π_i^eff)
        adaptive_noise_std = compute_precision_coupled_noise_std(pi_e_eff, pi_i_eff)
        dt = self.config.get("dt", 1.0)
        tau_s = self.config.get("tau_s", 5.0)

        # Choose between ODE (continuous) and discrete leaky accumulation
        if self.config.get("use_canonical_discrete_mode", False):
            # Discrete canonical mode: S(t+1) = (1-λ)S(t) + λS_inst(t)
            lam = self.config["lam"]
            self.S = integrate_signal_leaky(self.S, S_inst, lam)
        else:
            # ODE mode: dS/dt = -S/τ_S + Π_e|z_e| + β·Π_i|z_i| + σ·dW
            self.S = update_signal_ode(
                self.S,
                z_e_n,
                z_i_n,
                pi_e_eff,
                pi_i_eff,
                self.config["beta"],
                tau_s,
                dt,
                adaptive_noise_std,
            )

        self.S = stabilize_signal_log(
            self.S, enabled=self.config["signal_log_nonlinearity"]
        )

        # 7) Cost/value and threshold update (canonical spec semantics)
        # Mode A: Standard allostatic threshold (default)
        # Mode B: Reservoir-as-threshold (spec-explicit alternative per §10)
        use_reservoir_threshold = self.config.get("reservoir_as_threshold", False)

        if self.config["use_realistic_cost"]:
            C_t = compute_metabolic_cost_realistic(
                self.S, self.B_prev, self.config["c1"], self.config["c2"]
            )
        else:
            C_t = compute_metabolic_cost(self.S, self.config["c0"], self.config["c1"])

        # 7a) Thermodynamic cost validation (§11)
        C_landauer = 0.0
        bits_erased = 0.0
        if self.config.get("use_thermodynamic_cost", False):
            from core.thermodynamics import compute_information_bits

            C_landauer = compute_landauer_cost(
                self.S,
                self.config["eps"],
                k_b=self.config.get("k_boltzmann", 1.38e-23),
                T_env=self.config.get("T_env", 310.0),
                kappa_meta=self.config.get("kappa_meta", 1.0),
            )
            bits_erased = compute_information_bits(self.S, self.config["eps"])
            self.history["C_landauer"].append(C_landauer)
            self.history["bits_erased"].append(bits_erased)

        V_t = compute_information_value(
            z_e_n, z_i_eff, self.config["v1"], self.config["v2"]
        )

        # 7b/8) Threshold and Ignition: Standard or Reservoir-as-Threshold mode
        # Spec §10: Reservoir can serve as alternative execution path
        if use_reservoir_threshold:
            # Mode B: Reservoir-as-threshold (spec-explicit alternative)
            # Reservoir dynamics replace allostatic threshold computation
            if self.reservoir is None:
                raise ValueError("Reservoir mode enabled but reservoir not initialized")
            u_res = np.array([z_e_n, z_i_eff])
            self.reservoir.step(
                u_res,
                tau=self.config.get("reservoir_tau", 1.0),
                dt=dt,
                precision=pi_e_eff,
                S_target=self.S,
                theta=self.theta,
                A_amp=self.config.get("reservoir_amplification", 0.0),
            )
            # Reservoir readout serves as effective threshold
            S_reservoir = self.reservoir.readout(
                method=self.config.get("reservoir_readout_method", "linear")
            )
            # Map reservoir signal to effective threshold
            theta_next = self.config["theta_base"] * (
                1.0 + self.config.get("reservoir_theta_scale", 0.1) * S_reservoir
            )
            # Skip allostatic update - reservoir provides dynamics
            p_ignite = compute_ignition_probability(
                self.S, theta_next, self.config["ignite_tau"]
            )
            if self.config["stochastic_ignition"]:
                B_t = sample_ignition_state(p_ignite)
            else:
                B_t = int(detect_ignition_event(self.S, theta_next))
        else:
            # Mode A: Standard allostatic threshold (canonical spec semantics)
            # 7b) Threshold update: discrete or continuous ODE per APGI spec §4
            # Canonical semantics: θ(t+1) = θ(t) + η[C(t) - V(t)]
            # Refractory boost δ·B(t) applied AFTER ignition (step 9) per spec §6
            if self.config.get("use_continuous_threshold_ode", False):
                # Continuous ODE form. By default, refractory boost (δ·B(t)) is
                # handled post-ignition (step 9). When use_ode_refractory_drift=True
                # the impulse is included inside the ODE drift for closer alignment
                # with the analytic form in §7.4: dθ/dt = … + δ_reset·B(t)
                if self.config.get("use_ode_refractory_drift", False):
                    _ode_delta = self.config["delta"]
                    _ode_B = self.B_prev
                else:
                    _ode_delta = 0.0
                    _ode_B = 0

                theta_next = allostatic_threshold_ode(
                    theta=self.theta,
                    theta_0=self.config["theta_base"],
                    gamma=1.0 / self.config.get("tau_theta", 1000.0),
                    B_prev=_ode_B,
                    delta=_ode_delta,
                    C=C_t,
                    V=V_t,
                    eta=self.config["eta"],
                    dt=dt,
                    noise_std=self.config.get("noise_std", 0.01),
                )
            else:
                # Discrete form: θ_next = θ + η(C-V) [refractory in step 9]
                theta_next = update_threshold_discrete(
                    self.theta,
                    C_t,
                    V_t,
                    self.config["eta"],
                    delta=0.0,  # Refractory handled in step 9
                    B_prev=0,  # Refractory handled in step 9
                )

            # 7c) NE threshold modulation (per spec §4.4)
            if self.config.get("ne_on_threshold", False):
                theta_next = apply_ne_threshold_modulation(
                    theta_next, self.config["g_ne"], self.config["gamma_ne"]
                )

            # 7d) Hierarchical threshold modulation (PAC + Cascade) if enabled (§8.4)
            # Apply to ALL hierarchical modes for consistency
            if self.hierarchical is not None:
                # Prepare inputs for hierarchical threshold computation
                theta_0_levels = np.ones(self.n_levels) * theta_next
                # Level-specific signals for bottom-up cascade (if enabled)
                S_levels = np.zeros(self.n_levels)
                S_levels[0] = self.S
                if self.n_levels > 1:
                    S_levels[1:] = np.abs(self.phi_e_levels[1:]) + np.abs(
                        self.phi_i_levels[1:]
                    )

                # Use hierarchical precision if available, otherwise compute from per-level variance
                if self.hierarchical_network is not None:
                    _pi_levels = self.hierarchical_network.pi
                else:
                    _pi_levels = np.array(
                        [
                            compute_precision(
                                self.sigma2_e_levels[i] + self.sigma2_i_levels[i],
                                self.config["eps"],
                                self.config["pi_min"],
                                self.config["pi_max"],
                            )
                            for i in range(self.n_levels)
                        ]
                    )

                # Compute full hierarchical threshold set using modular component
                from oscillation.threshold_modulation import (
                    hierarchical_threshold_modulation,
                )
                from hierarchy.coupling import bottom_up_threshold_cascade

                thetas_mod = hierarchical_threshold_modulation(
                    thetas=theta_0_levels,
                    pis=_pi_levels,
                    phases=np.array(self.hierarchical.phases),
                    kappa_down=self.config.get("kappa_phase", 0.1),
                )

                # Apply bottom-up cascade if enabled
                if self.config.get("kappa_up", 0.0) > 0:
                    for level in range(1, self.n_levels):
                        thetas_mod[level] = bottom_up_threshold_cascade(
                            theta_ell=thetas_mod[level],
                            S_ell_minus_1=S_levels[level - 1],
                            theta_ell_minus_1=thetas_mod[level - 1],
                            kappa_up=self.config.get("kappa_up", 0.0),
                        )

                # Use the modulated threshold for the primary ignition level (level 0)
                theta_next = float(thetas_mod[0])
                self.hierarchical.thetas = thetas_mod.tolist()

            # Global threshold clamping for stability (§7.4)
            theta_next = float(np.clip(theta_next, 0.1, 20.0))

            # 8) Ignition (§5)
            p_ignite = compute_ignition_probability(
                self.S, theta_next, self.config["ignite_tau"]
            )
            if self.config["stochastic_ignition"]:
                B_t = sample_ignition_state(p_ignite)
            else:
                B_t = int(detect_ignition_event(self.S, theta_next))

        # 8b) Compute ignition margin using pre-reset values (§14)
        # Must compute BEFORE signal reset and refractory boost
        # Margin at ignition decision: Δ(t) = S(t) - θ(t) (pre-refractory)
        S_at_ignition = self.S  # Save pre-reset signal
        theta_at_ignition = theta_next  # Save pre-refractory threshold
        ignition_margin = S_at_ignition - theta_at_ignition

        # Post-ignition signal reset (§6)
        # Spec: S ← ρ·S on ignition
        if B_t == 1:
            reset_factor = self.config.get("reset_factor", 0.5)
            if not (0 < reset_factor < 1):
                raise ValueError(f"reset_factor must be in (0, 1), got {reset_factor}")
            self.S = self.S * reset_factor

        # 9) Post-ignition threshold dynamics (§6)
        # Canonical spec semantics:
        #   9a) Refractory boost: θ ← θ + δ·B(t)  [using CURRENT B_t]
        #   9b) Decay to baseline: θ ← θ_base + (θ - θ_base)·e^{-κ}
        theta_next = apply_refractory_boost(theta_next, B_t, self.config["delta"])
        theta_next = threshold_decay(
            theta_next, self.config["theta_base"], self.config["kappa"]
        )

        # 10) Internal Prediction Update per §1.4 Generative Model Dynamics
        # Support both flag names for backward compatibility
        use_generative_update = self.config.get(
            "use_generative_model_update", False
        ) or self.config.get("use_internal_predictions", False)
        if use_generative_update:
            self.x_hat_e = update_prediction(
                self.x_hat_e,
                z_e,
                pi_e,
                self.config.get("kappa_e", 0.01),
                self.config["pi_max"],
            )
            self.x_hat_i = update_prediction(
                self.x_hat_i,
                z_i,
                pi_i,
                self.config.get("kappa_i", 0.01),
                self.config["pi_max"],
            )

        # 11) Reservoir layer update (§10) - add-on mode only
        # When reservoir_as_threshold=True, reservoir is handled in step 7b/8
        S_reservoir = 0.0
        if self.reservoir is not None and not use_reservoir_threshold:
            u_res = np.array([z_e_n, z_i_eff])
            self.reservoir.step(
                u_res,
                tau=self.config.get("reservoir_tau", 1.0),
                dt=dt,
                precision=pi_e_eff,
                S_target=self.S,
                theta=self.theta,
                A_amp=self.config.get("reservoir_amplification", 0.0),
            )
            S_reservoir = self.reservoir.readout(
                method=self.config.get("reservoir_readout_method", "linear")
            )

        self.theta = theta_next
        self.B_prev = B_t
        self.t += self.config.get("dt", 1.0)

        # Update history for validation
        self.history["S"].append(self.S)
        self.history["theta"].append(self.theta)
        self.history["B"].append(float(B_t))

        result = {
            "z_e": z_e,
            "z_i": z_i,
            "z_e_norm": z_e_n,
            "z_i_norm": z_i_n,
            "mu_e": self.state.mu_e,
            "mu_i": self.state.mu_i,
            "pi_e": pi_e,
            "pi_i": pi_i,
            "pi_e_eff": pi_e_eff,
            "pi_i_eff": pi_i_eff,
            "z_i_eff": z_i_eff,
            "S_inst": S_inst,
            "S": self.S,
            "C": C_t,
            "V": V_t,
            "theta": self.theta,
            "ignition_margin": ignition_margin,
            "p_ignite": p_ignite,
            "B": B_t,
            "theta_dot": self.theta_dot,
            "x_hat_e": self.x_hat_e,
            "x_hat_i": self.x_hat_i,
            "M_somatic": self.M,
        }

        # Add thermodynamic info if enabled
        if self.config.get("use_thermodynamic_cost", False):
            result["C_landauer"] = C_landauer
            result["bits_erased"] = bits_erased

        # Add reservoir info if enabled
        if self.reservoir is not None:
            result["S_reservoir"] = S_reservoir
            result["reservoir_state_norm"] = float(np.linalg.norm(self.reservoir.x))

        # Add Kuramoto oscillator info if enabled (§9)
        if self.kuramoto is not None:
            kuramoto_result = self.kuramoto.step(dt=dt)
            result["kuramoto_phases"] = kuramoto_result["phases"].tolist()
            result["kuramoto_synchronization"] = kuramoto_result["synchronization"]

            # Apply phase reset on ignition
            if B_t == 1:
                # Broadcast phase reset across the full hierarchy when enabled.
                # kuramoto_broadcast_ignition=True triggers ignition-induced phase
                # resets at all levels with hierarchical distance decay (§9.2).
                broadcast_ignition = self.config.get(
                    "kuramoto_broadcast_ignition", False
                )
                if broadcast_ignition:
                    # Full-hierarchy broadcast: reset every level with decay
                    for _lvl in range(self.kuramoto.n_levels):
                        self.kuramoto.apply_ignition_reset(
                            level=_lvl,
                            broadcast=True,
                        )
                else:
                    # Default: reset only primary level 0
                    self.kuramoto.apply_ignition_reset(level=0)

        # Add observable mapping if enabled (§14)
        # Use pre-reset values for correct margin computation
        if self.neural_observables is not None:
            neural_obs = self.neural_observables.step(
                S_at_ignition, theta_at_ignition, B_t
            )
            result["neural_gamma_power"] = neural_obs["gamma_power"]
            result["neural_erp_amplitude"] = neural_obs["erp_amplitude"]
            result["neural_ignition_rate"] = neural_obs["ignition_rate"]

        if self.behavioral_observables is not None:
            behavioral_obs = self.behavioral_observables.step(
                S_at_ignition, theta_at_ignition, B_t
            )
            result["behavioral_rt_variability"] = behavioral_obs["rt_variability"]
            result["behavioral_response_criterion"] = behavioral_obs[
                "response_criterion"
            ]
            result["behavioral_decision_rate"] = behavioral_obs["decision_rate"]

        if self.prediction_validator is not None:
            pred_result = self.prediction_validator.step(
                S_at_ignition, theta_at_ignition, B_t
            )
            result["prediction_margin"] = pred_result["delta"]
            result["prediction_p_ign"] = pred_result["p_ign"]

        # Add stability analysis if enabled (§7)
        if self.stability_analyzer is not None:
            self.stability_analyzer.step(self.S, self.theta)

        if self.hierarchical is not None:
            result["hierarchical_pis"] = self.hierarchical.pis.copy()
            result["hierarchical_phases"] = self.hierarchical.phases.copy()
            result["hierarchical_thetas"] = self.hierarchical.thetas.copy()

        return result

    def validate(self) -> dict:
        """Perform statistical validation on the accumulated history."""

        if len(self.history["theta"]) < 64:
            return {"status": "insufficient_data"}

        # Validate threshold 1/f dynamics
        theta_arr = np.array(self.history["theta"])
        fs = 1.0 / self.config.get("dt", 1.0)

        # Estimate Hurst exponent
        hurst_val = estimate_hurst_robust(theta_arr, fs=fs)

        # Check for pink noise
        # Re-compute PSD for validate_pink_noise
        from stats.hurst import welch_periodogram

        freqs, psd = welch_periodogram(theta_arr, fs=fs)
        pink_stats = validate_pink_noise(freqs, psd)

        return {
            "status": "success",
            "hurst_exponent": hurst_val,
            "is_pink_noise": pink_stats["is_pink_noise"],
            "beta": pink_stats["beta"],
            "data_points": len(theta_arr),
        }
