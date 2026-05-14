"""Active Inference Action Loop — §19 APGI Full Specs.

Implements:
  aₜ = argmin_a E[F(a)]       (policy selection)

where Expected Free Energy decomposes as:
  F(a) = w_epi · expected_surprise(a)
       - w_prag · expected_salience_gain(a)
       + w_met  · metabolic_cost(a)

Actions feed back into ignition dynamics through three channels (§19):
  1. Sensory consequence   → Δx̂ₑ  (changes future εₜ)
  2. Interoceptive         → ΔM    (metabolic state → θₜ via allostatic ODE)
  3. Uncertainty reduction → ΔΣ   (reduces Σₜ, raises Πₜ)

Policy posterior uses a Boltzmann softmax:
  p(a) ∝ exp(−γ_policy · F(a))

The MAP action is returned as the selected policy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Default K=3 action parameter table: shape (n_actions, 3)
# Columns: [sensory_shift, epistemic_gain, metabolic_cost]
#   sensory_shift   — magnitude of Δx̂ₑ shift toward preferred observation
#   epistemic_gain  — reduction in Σₜ (ΔΣ ≤ 0 means uncertainty drops)
#   metabolic_cost  — fractional ATP cost relative to resting metabolism
_DEFAULT_ACTION_PARAMS = np.array(
    [
        [0.10, 0.80, 0.60],  # "explore": high epistemic gain, high cost
        [0.80, 0.10, 0.20],  # "exploit": high pragmatic gain, low cost
        [0.00, 0.00, 0.05],  # "rest":    no shift, minimal cost
    ],
    dtype=float,
)

ACTION_LABELS = ["explore", "exploit", "rest"]


def default_action_params() -> np.ndarray:
    """Return a copy of the default (n_actions=3, 3) action parameter table."""
    return _DEFAULT_ACTION_PARAMS.copy()


@dataclass
class PolicyResult:
    """Outcome of one policy-selection step."""

    action_idx: int  # selected action index (argmin F)
    action_label: str  # human-readable label if available
    F_values: list[float]  # E[F(a)] for each candidate action
    p_policies: list[float]  # Boltzmann posterior p(a) over actions
    expected_free_energy: float  # F of the selected action


@dataclass
class ActionFeedback:
    """The three spec-mandated feedback effects of the selected action (§19)."""

    delta_x_hat_e: float  # Sensory consequence: shift in exteroceptive prediction
    delta_x_hat_i: float  # Sensory consequence: shift in interoceptive prediction
    delta_M: float  # Interoceptive consequence: metabolic state change
    delta_sigma2_e: float  # Epistemic consequence: change in exteroceptive variance
    delta_sigma2_i: float  # Epistemic consequence: change in interoceptive variance


def compute_expected_free_energy(
    sigma2_e: float,
    sigma2_i: float,
    S: float,
    theta: float,
    action_sensory_shift: float,
    action_epistemic_gain: float,
    action_metabolic_cost: float,
    w_epistemic: float = 1.0,
    w_pragmatic: float = 1.0,
    w_metabolic: float = 0.5,
) -> float:
    """Scalar Expected Free Energy F(a) for one candidate action.

    Components
    ----------
    Epistemic:  expected residual variance after the action reduces uncertainty.
                Lower is better (less surprise in future observations).
    Pragmatic:  expected gain in precision-weighted salience relative to threshold.
                The action shifts the predicted observation toward preferred states,
                raising the expected S(t+1) − θ(t).  Higher is better, so it enters
                with a negative sign.
    Metabolic:  direct ATP cost of executing the action (§14 cost model).

    Returns
    -------
    F(a) — scalar. Smaller means the action is preferred.
    """
    total_sigma2 = sigma2_e + sigma2_i

    # Epistemic term: residual uncertainty after epistemic gain
    sigma2_after = max(total_sigma2 - action_epistemic_gain, 1e-6)

    # Pragmatic term: expected salience relative to threshold
    # The sensory shift reduces expected |ε| → lowers future S by action_sensory_shift,
    # but only if current signal already has informational value (S > 0).
    ignition_margin = S - theta
    pragmatic_gain = action_sensory_shift * max(ignition_margin, 0.0)

    F = (
        w_epistemic * sigma2_after
        - w_pragmatic * pragmatic_gain
        + w_metabolic * action_metabolic_cost
    )
    return float(F)


def _softmax(x: np.ndarray, gamma: float) -> np.ndarray:
    """Numerically stable softmax: p(a) ∝ exp(−γ · x)."""
    scaled = -gamma * x
    scaled -= np.max(scaled)
    exp_x = np.exp(scaled)
    return exp_x / np.sum(exp_x)  # type: ignore[no-any-return]


class ActiveInferenceAgent:
    """Discrete-policy active inference loop closing the perception-action cycle.

    Implements §19: aₜ = argmin_a E[F(a)] with feedback through three channels.

    Parameters
    ----------
    n_actions : int
        Number of candidate policies K.
    action_params : np.ndarray, shape (K, 3)
        Parameter table. Columns: [sensory_shift, epistemic_gain, metabolic_cost].
        All values in [0, 1] by convention (scaled by feedback rates at application).
    policy_precision : float
        γ_policy — sharpness of the Boltzmann softmax. High γ → near-deterministic
        argmin; γ → 0 → uniform random policy selection.
    w_epistemic, w_pragmatic, w_metabolic : float
        Weights on the three F(a) components.
    sensory_feedback_rate : float
        Scale factor for Δx̂ₑ (channel 1 effect magnitude).
    metabolic_feedback_rate : float
        Scale factor for ΔM (channel 2 effect magnitude).
    precision_update_rate : float
        Scale factor for ΔΣ (channel 3 effect magnitude).
    """

    def __init__(
        self,
        n_actions: int = 3,
        action_params: np.ndarray | None = None,
        policy_precision: float = 2.0,
        w_epistemic: float = 1.0,
        w_pragmatic: float = 1.0,
        w_metabolic: float = 0.5,
        sensory_feedback_rate: float = 0.1,
        metabolic_feedback_rate: float = 0.1,
        precision_update_rate: float = 0.05,
    ) -> None:
        if action_params is None:
            if n_actions == 3:
                self.action_params = _DEFAULT_ACTION_PARAMS.copy()
            else:
                # Uniform spread along explore ↔ exploit axis
                t = np.linspace(0, 1, n_actions)
                sensory = t
                epistemic = 1.0 - t
                metabolic = 0.05 + 0.55 * (1.0 - t)
                self.action_params = np.column_stack([sensory, epistemic, metabolic])
        else:
            self.action_params = np.asarray(action_params, dtype=float)
            if self.action_params.shape != (n_actions, 3):
                raise ValueError(
                    f"action_params must have shape ({n_actions}, 3), "
                    f"got {self.action_params.shape}"
                )

        self.n_actions = n_actions
        self.policy_precision = float(policy_precision)
        self.w_epistemic = float(w_epistemic)
        self.w_pragmatic = float(w_pragmatic)
        self.w_metabolic = float(w_metabolic)
        self.sensory_feedback_rate = float(sensory_feedback_rate)
        self.metabolic_feedback_rate = float(metabolic_feedback_rate)
        self.precision_update_rate = float(precision_update_rate)

        # Running history for diagnostics
        self.action_history: list[int] = []
        self.F_history: list[list[float]] = []

    def select_policy(
        self,
        sigma2_e: float,
        sigma2_i: float,
        S: float,
        theta: float,
    ) -> PolicyResult:
        """Compute E[F(a)] for all actions and return the MAP policy.

        Parameters
        ----------
        sigma2_e, sigma2_i : float
            Current extero- and interoceptive running variances (proxy for Σₜ).
        S : float
            Current accumulated ignition signal.
        theta : float
            Current ignition threshold.

        Returns
        -------
        PolicyResult with selected action, F values, and policy posterior.
        """
        F_values = np.array(
            [
                compute_expected_free_energy(
                    sigma2_e=sigma2_e,
                    sigma2_i=sigma2_i,
                    S=S,
                    theta=theta,
                    action_sensory_shift=self.action_params[k, 0],
                    action_epistemic_gain=self.action_params[k, 1],
                    action_metabolic_cost=self.action_params[k, 2],
                    w_epistemic=self.w_epistemic,
                    w_pragmatic=self.w_pragmatic,
                    w_metabolic=self.w_metabolic,
                )
                for k in range(self.n_actions)
            ],
            dtype=float,
        )

        p_policies = _softmax(F_values, self.policy_precision)
        action_idx = int(np.argmin(F_values))

        label = ACTION_LABELS[action_idx] if action_idx < len(ACTION_LABELS) else str(action_idx)

        result = PolicyResult(
            action_idx=action_idx,
            action_label=label,
            F_values=F_values.tolist(),
            p_policies=p_policies.tolist(),
            expected_free_energy=float(F_values[action_idx]),
        )

        self.action_history.append(action_idx)
        self.F_history.append(F_values.tolist())

        return result

    def apply_action_feedback(
        self,
        action_idx: int,
        z_e: float,
        z_i: float,
        sigma2_e: float,
        sigma2_i: float,
        M: float,
    ) -> ActionFeedback:
        """Compute the three feedback-channel deltas for the selected action.

        Channel 1 — Sensory consequence (§19 pt 1):
            Δx̂ₑ = sensory_feedback_rate · action_sensory_shift · sign(z_e)
            The action shifts the model's prediction toward reducing |ε| by
            moving x̂ₑ in the direction of the current error.

        Channel 2 — Interoceptive / metabolic consequence (§19 pt 2):
            ΔM = −metabolic_feedback_rate · action_metabolic_cost
            Costly actions deplete metabolic state; epistemic gain partially
            offsets the cost via reduced future surprise.

        Channel 3 — Uncertainty reduction (§19 pt 3):
            ΔΣₑ = −precision_update_rate · action_epistemic_gain · sigma2_e
            ΔΣᵢ = −precision_update_rate · action_epistemic_gain · sigma2_i
            Epistemic actions (exploration) selectively reduce variance,
            raising Πₜ on the next step.
        """
        params = self.action_params[action_idx]
        sensory_shift = params[0]
        epistemic_gain = params[1]
        metabolic_cost = params[2]

        # Channel 1: shift extero- and interoceptive predictions toward error
        delta_x_hat_e = self.sensory_feedback_rate * sensory_shift * float(np.sign(z_e))
        delta_x_hat_i = self.sensory_feedback_rate * sensory_shift * float(np.sign(z_i))

        # Channel 2: metabolic state change (negative = depletion)
        delta_M = -self.metabolic_feedback_rate * metabolic_cost

        # Channel 3: variance reduction (bounded — cannot go below zero)
        reduction = self.precision_update_rate * epistemic_gain
        delta_sigma2_e = -reduction * sigma2_e
        delta_sigma2_i = -reduction * sigma2_i

        return ActionFeedback(
            delta_x_hat_e=delta_x_hat_e,
            delta_x_hat_i=delta_x_hat_i,
            delta_M=delta_M,
            delta_sigma2_e=delta_sigma2_e,
            delta_sigma2_i=delta_sigma2_i,
        )

    def reset(self) -> None:
        """Clear history (call between trials or paradigm runs)."""
        self.action_history.clear()
        self.F_history.clear()
