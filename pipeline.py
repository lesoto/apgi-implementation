from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.dynamics import signal_drift, update_prediction
from core.allostatic import allostatic_threshold_ode, update_threshold_euler
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
    update_mean_ema,
    update_precision_euler,
    update_variance_ema,
)
from core.preprocessing import compute_prediction_error
from core.sde import integrate_euler_maruyama
from core.signal import instantaneous_signal, stabilize_signal_log
from core.threshold import (
    apply_ne_threshold_modulation,
    compute_information_value,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
)
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

    def __init__(self, config: dict):
        # Copy to avoid mutating the caller's dict
        self.config = dict(config)

        # Validate NE configuration to prevent double-counting
        if self.config.get("ne_on_precision", False) and self.config.get(
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

        # Hierarchical state if enabled
        self.hierarchical = None
        if self.config.get("use_hierarchical", False):
            n_levels = self.config.get("n_levels", 3)
            self.hierarchical = HierarchicalState(n_levels=n_levels)

        # Somatic marker state if enabled
        self.M = self.config.get("M_somatic", 0.0)  # Somatic marker ∈ [-2, +2]

        self.history = {
            "S": [],
            "theta": [],
            "B": [],
        }

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

        # 2) Online mean + variance update (EMA, centered)
        self.state.mu_e = update_mean_ema(self.state.mu_e, z_e, self.config["alpha_e"])
        self.state.mu_i = update_mean_ema(self.state.mu_i, z_i, self.config["alpha_i"])
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
                # Bottom-up coupling from lower level error ψ(ε_{ℓ-1})
                epsilon_minus_1 = (
                    abs(z_i_eff) if level == 1 else (0.0 if level > 1 else None)
                )
                epsilon = abs(z_i_eff if level == 0 else 0.0)

                dpi_dt = precision_coupling_ode_core(
                    pi_ell=pi_curr,
                    tau_pi=self.config.get("tau_pi", 1000.0),
                    epsilon_ell=epsilon,
                    alpha_gain=self.config.get("alpha_gain", 0.1),
                    pi_ell_plus_1=pi_plus,
                    epsilon_ell_minus_1=epsilon_minus_1,
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

        # 6) Instantaneous signal (diagnostic) + SDE integration via ODE drift
        S_inst = instantaneous_signal(z_e_n, z_i_eff, pi_e_eff, pi_i_eff)

        # Wire dynamics.py (signal_drift) + sde.py (integrate_euler_maruyama):
        # dS/dt = -S/τ_S + Π_e|z_e| + β·Π_i|z_i| + σ·dW
        _z_e_n = z_e_n
        _z_i_n = z_i_n
        _pi_e_eff = pi_e_eff
        _pi_i_eff = pi_i_eff
        _beta = self.config["beta"]
        _tau_s = self.config.get("tau_s", 5.0)

        def drift_fn(s: float, _t: float) -> float:
            return signal_drift(s, _z_e_n, _z_i_n, _pi_e_eff, _pi_i_eff, _beta, _tau_s)

        self.S = integrate_euler_maruyama(
            self.S,
            drift_fn,
            self.config.get("noise_std", 0.01),
            self.t,
            self.config.get("dt", 1.0),
        )
        self.S = stabilize_signal_log(
            self.S, enabled=self.config["signal_log_nonlinearity"]
        )

        # 7) Cost/value and threshold update
        if self.config["use_realistic_cost"]:
            C_t = compute_metabolic_cost_realistic(
                self.S, self.B_prev, self.config["c1"], self.config["c2"]
            )
        else:
            C_t = compute_metabolic_cost(self.S, self.config["c0"], self.config["c1"])

        V_t = compute_information_value(
            z_e_n, z_i_eff, self.config["v1"], self.config["v2"]
        )

        # 7b) Threshold update: discrete or continuous ODE per APGI spec
        # Spec: θ(t+1) = θ(t) + η[C(t) - V(t)] + δ_reset·B(t) (with B from previous step)
        # Note: δ_reset·B is part of core allostatic update, applied BEFORE NE modulation
        if self.config.get("use_continuous_threshold_ode", False):
            dt = self.config.get("dt", 1.0)
            self.theta_dot = allostatic_threshold_ode(
                theta=self.theta,
                theta_0=self.config["theta_base"],
                gamma=1.0 / self.config.get("tau_theta", 1000.0),
                B_prev=self.B_prev,
                delta=self.config["delta"],
            )
            # Add allostatic mismatch contribution to theta_dot
            self.theta_dot += self.config["eta"] * (C_t - V_t)

            theta_next = update_threshold_euler(self.theta, self.theta_dot, dt)
        else:
            # Discrete form: θ += η(C-V) + δ_reset·B_prev
            # Refractory boost is part of the allostatic update per spec Section 4
            theta_next = update_threshold_discrete(
                self.theta,
                C_t,
                V_t,
                self.config["eta"],
                self.config["delta"],
                self.B_prev,
            )

        if self.config.get("ne_on_threshold", False):
            theta_next = apply_ne_threshold_modulation(
                theta_next, self.config["g_ne"], self.config["gamma_ne"]
            )

        # 7c) Phase-locked threshold modulation if enabled
        if self.hierarchical is not None and self.config.get(
            "use_phase_modulation", False
        ):
            from oscillation.threshold_modulation import modulate_threshold_by_phase

            for level in range(self.hierarchical.n_levels):
                omega = self.config.get("omega_phases", [0.1, 0.05, 0.01])[level]
                self.hierarchical.phases[level] = (
                    omega * self.t + self.hierarchical.phases[level]
                ) % (2 * 3.14159)

            if self.hierarchical.n_levels > 1:
                theta_next = modulate_threshold_by_phase(
                    theta_next,
                    self.hierarchical.pis[1],
                    self.hierarchical.phases[1],
                    self.config.get("kappa_phase", 0.1),
                )

        # 8) Ignition
        p_ignite = compute_ignition_probability(
            self.S, theta_next, self.config["ignite_tau"]
        )
        if self.config["stochastic_ignition"]:
            B_t = sample_ignition_state(p_ignite)
        else:
            B_t = int(detect_ignition_event(self.S, theta_next))

        # 9) Post-ignition threshold decay (exponential relaxation to baseline)
        # Note: Refractory boost δ_reset·B was already applied in step 7b as part of
        # the allostatic update per spec Section 13. Only decay is applied here.
        theta_next = threshold_decay(
            theta_next, self.config["theta_base"], self.config["kappa"]
        )

        # 10) Internal Prediction Update per §1.4 Generative Model Dynamics
        if self.config.get("use_internal_predictions", False):
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
            "p_ignite": p_ignite,
            "B": B_t,
            "theta_dot": self.theta_dot,
            "x_hat_e": self.x_hat_e,
            "x_hat_i": self.x_hat_i,
            "M_somatic": self.M,
        }

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
