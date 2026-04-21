from __future__ import annotations

from dataclasses import dataclass

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
    precision_coupling_ode_core,
    update_precision_euler,
    update_variance_ema,
)
from core.preprocessing import compute_prediction_error, normalize_error
from core.signal import (
    integrate_signal_leaky,
    instantaneous_signal,
    stabilize_signal_log,
)
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    compute_information_value,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
)
from core.dynamics import update_threshold_ode


@dataclass
class PrecisionState:
    sigma2_e: float
    sigma2_i: float
    pi_e: float = 1.0
    pi_i: float = 1.0


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

    def __init__(self, config: dict):
        self.config = config
        # Validate NE configuration to prevent double-counting
        if config.get("ne_on_precision", False) and config.get(
            "ne_on_threshold", False
        ):
            import warnings

            warnings.warn(
                "Both ne_on_precision and ne_on_threshold are True. "
                "This double-counts norepinephrine effects. "
                "Recommendation: enable only one. See spec Section 2.3-2.4.",
                RuntimeWarning,
                stacklevel=2,
            )
        # Auto-adjust NE parameters for threshold mode to prevent runaway
        if config.get("ne_on_threshold", False) and config.get("gamma_ne", 0.1) >= 0.1:
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
        self.S = float(config["S0"])
        self.theta = float(config["theta_0"])
        self.theta_dot = 0.0  # For continuous ODE tracking
        self.state = PrecisionState(
            sigma2_e=float(config["sigma2_e0"]),
            sigma2_i=float(config["sigma2_i0"]),
            pi_e=1.0,
            pi_i=1.0,
        )
        self.B_prev = 0
        self.t = 0.0  # Time tracking for phase

        # Hierarchical state if enabled
        self.hierarchical = None
        if config.get("use_hierarchical", False):
            n_levels = config.get("n_levels", 3)
            self.hierarchical = HierarchicalState(n_levels=n_levels)

        # Somatic marker state if enabled
        self.M = config.get("M_somatic", 0.0)  # Somatic marker ∈ [-2, +2]

    def step(self, x_e: float, x_hat_e: float, x_i: float, x_hat_i: float):
        # 1) Raw prediction errors
        z_e = compute_prediction_error(x_e, x_hat_e)
        z_i = compute_prediction_error(x_i, x_hat_i)

        # 2) Online variance update (EMA)
        self.state.sigma2_e = update_variance_ema(
            self.state.sigma2_e, z_e, self.config["alpha_e"]
        )
        self.state.sigma2_i = update_variance_ema(
            self.state.sigma2_i, z_i, self.config["alpha_i"]
        )

        # Optional normalization
        z_e_n = normalize_error(z_e, self.state.sigma2_e**0.5, self.config["eps"])
        z_i_n = normalize_error(z_i, self.state.sigma2_i**0.5, self.config["eps"])

        # 3) Precision with clamping
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

        # 4) Neuromodulation (+ dopamine correction)
        pi_e_eff = apply_ach_gain(pi_e, self.config["g_ach"])

        # Interoceptive precision: use exponential somatic modulation if enabled
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

        # 4b) Hierarchical precision ODE if enabled
        if self.hierarchical is not None and self.config.get(
            "use_hierarchical_precision_ode", False
        ):
            dt = self.config.get("dt", 1.0)
            for level in range(self.hierarchical.n_levels):
                pi_curr = self.hierarchical.pis[level]
                pi_plus = (
                    self.hierarchical.pis[level + 1]
                    if level < self.hierarchical.n_levels - 1
                    else None
                )
                pi_minus = self.hierarchical.pis[level - 1] if level > 0 else None
                # Use level 0 error for all levels (simplified)
                epsilon = abs(z_i_eff if level == 0 else 0.0)

                dpi_dt = precision_coupling_ode_core(
                    pi_ell=pi_curr,
                    tau_pi=self.config.get("tau_pi", 1000.0),
                    epsilon_ell=epsilon,
                    alpha_gain=self.config.get("alpha_gain", 0.1),
                    pi_ell_plus_1=pi_plus,
                    pi_ell_minus_1=pi_minus,
                    C_down=self.config.get("C_down", 0.1),
                    C_up=self.config.get("C_up", 0.05),
                )

                self.hierarchical.pis[level] = update_precision_euler(
                    pi_curr,
                    dpi_dt,
                    dt,
                    self.config["pi_min"],
                    self.config["pi_max"],
                )

        # 5) Instantaneous + leaky accumulated signal
        S_inst = instantaneous_signal(z_e_n, z_i_eff, pi_e_eff, pi_i_eff)
        self.S = integrate_signal_leaky(self.S, S_inst, self.config["lam"])
        self.S = stabilize_signal_log(
            self.S, enabled=self.config["signal_log_nonlinearity"]
        )

        # 6) Cost/value and threshold update
        if self.config["use_realistic_cost"]:
            C_t = compute_metabolic_cost_realistic(
                self.S, self.B_prev, self.config["c1"], self.config["c2"]
            )
        else:
            C_t = compute_metabolic_cost(self.S, self.config["c0"], self.config["c1"])

        V_t = compute_information_value(
            z_e_n, z_i_n, self.config["v1"], self.config["v2"]
        )

        # 6b) Threshold update: discrete or continuous ODE
        if self.config.get("use_continuous_threshold_ode", False):
            dt = self.config.get("dt", 1.0)
            self.theta_dot = update_threshold_ode(
                self.theta,
                self.S,
                C_t,
                V_t,
                self.config.get("tau_theta", 1000.0),
                self.config["eta"],
                self.config.get("gamma_theta", 0.1),
                self.config.get("theta_noise_std", 0.01),
            )
            theta_next = self.theta + dt * self.theta_dot
        else:
            theta_next = update_threshold_discrete(
                self.theta, C_t, V_t, self.config["eta"]
            )

        if self.config.get("ne_on_threshold", False):
            theta_next = apply_ne_threshold_modulation(
                theta_next, self.config["g_ne"], self.config["gamma_ne"]
            )

        # 6c) Phase-locked threshold modulation if enabled
        if self.hierarchical is not None and self.config.get(
            "use_phase_modulation", False
        ):
            from oscillation.threshold_modulation import modulate_threshold_by_phase

            # Update phases: ϕ(t) = ωt + ϕ_0
            for level in range(self.hierarchical.n_levels):
                omega = self.config.get("omega_phases", [0.1, 0.05, 0.01])[level]
                self.hierarchical.phases[level] = (
                    omega * self.t + self.hierarchical.phases[level]
                ) % (2 * 3.14159)

            # Apply phase modulation to threshold
            if self.hierarchical.n_levels > 1:
                theta_next = modulate_threshold_by_phase(
                    theta_next,
                    self.hierarchical.pis[1],
                    self.hierarchical.phases[1],
                    self.config.get("kappa_phase", 0.1),
                )

        # 7) Ignition
        p_ignite = compute_ignition_probability(
            self.S, theta_next, self.config["ignite_tau"]
        )
        if self.config["stochastic_ignition"]:
            B_t = sample_ignition_state(p_ignite)
        else:
            B_t = int(detect_ignition_event(self.S, theta_next))

        # 8) Refractory effects
        theta_next = apply_refractory_boost(theta_next, B_t, self.config["delta"])
        theta_next = threshold_decay(
            theta_next, self.config["theta_base"], self.config["kappa"]
        )

        self.theta = theta_next
        self.B_prev = B_t
        self.t += self.config.get("dt", 1.0)

        result = {
            "z_e": z_e,
            "z_i": z_i,
            "z_e_norm": z_e_n,
            "z_i_norm": z_i_n,
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
            "p_ignite": p_ignite,
            "B": B_t,
            "theta_dot": self.theta_dot,
            "M_somatic": self.M,
        }

        # Add hierarchical state if enabled
        if self.hierarchical is not None:
            result["hierarchical_pis"] = self.hierarchical.pis.copy()
            result["hierarchical_phases"] = self.hierarchical.phases.copy()
            result["hierarchical_thetas"] = self.hierarchical.thetas.copy()

        return result
