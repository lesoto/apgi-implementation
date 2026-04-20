from __future__ import annotations

from dataclasses import dataclass

from core.ignition import compute_ignition_probability, detect_ignition_event, sample_ignition_state
from core.precision import (
    apply_ach_gain,
    apply_dopamine_bias_to_error,
    apply_ne_gain,
    compute_precision,
    update_variance_ema,
)
from core.preprocessing import compute_prediction_error, normalize_error
from core.signal import integrate_signal_leaky, instantaneous_signal, stabilize_signal_log
from core.threshold import (
    apply_ne_threshold_modulation,
    apply_refractory_boost,
    compute_information_value,
    compute_metabolic_cost,
    compute_metabolic_cost_realistic,
    threshold_decay,
    update_threshold_discrete,
)


@dataclass
class PrecisionState:
    sigma2_e: float
    sigma2_i: float


class APGIPipeline:
    """APGI one-step update implementing the full corrected mathematical pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.S = float(config["S0"])
        self.theta = float(config["theta_0"])
        self.state = PrecisionState(
            sigma2_e=float(config["sigma2_e0"]),
            sigma2_i=float(config["sigma2_i0"]),
        )
        self.B_prev = 0

    def step(self, x_e: float, x_hat_e: float, x_i: float, x_hat_i: float):
        # 1) Raw prediction errors
        z_e = compute_prediction_error(x_e, x_hat_e)
        z_i = compute_prediction_error(x_i, x_hat_i)

        # 2) Online variance update (EMA)
        self.state.sigma2_e = update_variance_ema(self.state.sigma2_e, z_e, self.config["alpha_e"])
        self.state.sigma2_i = update_variance_ema(self.state.sigma2_i, z_i, self.config["alpha_i"])

        # Optional normalization
        z_e_n = normalize_error(z_e, self.state.sigma2_e**0.5, self.config["eps"])
        z_i_n = normalize_error(z_i, self.state.sigma2_i**0.5, self.config["eps"])

        # 3) Precision with clamping
        pi_e = compute_precision(self.state.sigma2_e, self.config["eps"], self.config["pi_min"], self.config["pi_max"])
        pi_i = compute_precision(self.state.sigma2_i, self.config["eps"], self.config["pi_min"], self.config["pi_max"])

        # 4) Neuromodulation (+ dopamine correction)
        pi_e_eff = apply_ach_gain(pi_e, self.config["g_ach"])
        pi_i_eff = apply_ne_gain(pi_i, self.config["g_ne"]) if self.config["ne_on_precision"] else pi_i
        z_i_eff = apply_dopamine_bias_to_error(z_i_n, self.config["beta"])

        # 5) Instantaneous + leaky accumulated signal
        S_inst = instantaneous_signal(z_e_n, z_i_eff, pi_e_eff, pi_i_eff)
        self.S = integrate_signal_leaky(self.S, S_inst, self.config["lam"])
        self.S = stabilize_signal_log(self.S, enabled=self.config["signal_log_nonlinearity"])

        # 6) Cost/value and threshold update
        if self.config["use_realistic_cost"]:
            C_t = compute_metabolic_cost_realistic(
                self.S, self.B_prev, self.config["c1"], self.config["c2"]
            )
        else:
            C_t = compute_metabolic_cost(self.S, self.config["c0"], self.config["c1"])

        V_t = compute_information_value(z_e_n, z_i_n, self.config["v1"], self.config["v2"])

        theta_next = update_threshold_discrete(self.theta, C_t, V_t, self.config["eta"])
        if self.config["ne_on_threshold"]:
            theta_next = apply_ne_threshold_modulation(theta_next, self.config["g_ne"], self.config["gamma_ne"])

        # 7) Ignition
        p_ignite = compute_ignition_probability(self.S, theta_next, self.config["ignite_tau"])
        if self.config["stochastic_ignition"]:
            B_t = sample_ignition_state(p_ignite)
        else:
            B_t = int(detect_ignition_event(self.S, theta_next))

        # 8) Refractory effects
        theta_next = apply_refractory_boost(theta_next, B_t, self.config["delta"])
        theta_next = threshold_decay(theta_next, self.config["theta_base"], self.config["kappa"])

        self.theta = theta_next
        self.B_prev = B_t

        return {
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
        }
