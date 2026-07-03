"""Protocol 2 (Somatic-AgentSim): Active Inference Agent Simulations.
Protocols & Prediction Registry, "Protocol 2 (Somatic-AgentSim): Active
Inference Agent Simulations — Adaptive Advantages of Somatic Markers".
Tests whether somatic marker integration confers measurable adaptive
advantages over agents in which the somatic modulation pathway is
selectively lesioned, via:
    M̂(c,a) = γ_V · V(c,a) + γ_A^SM · A(c,a)
    Π_i_eff(a) = Π_i_baseline(a) · exp(β_SM · M̂(c,a))
on a stochastic three-armed bandit with hidden-state reversals, under three
volatility conditions σ_env ∈ {0.1, 0.3, 0.6}.
Five agent types (Protocol 2 spec): full APGI (M̂ active), β_SM-lesion
(somatic channel disabled, β_SM=0), Π_i-lesion (interoceptive precision
held constant), α-lesion (metabolic weighting disabled), and a random
baseline.
This is a self-contained, reasonably faithful simulation of the protocol's
structure and parameters — not a claim of publication-grade fidelity to a
specific empirical Iowa Gambling Task dataset. Reuses
core.precision.compute_interoceptive_precision_exponential so the somatic
modulation formula is identical to the one used in the main APGI pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from core.precision import clamp, compute_interoceptive_precision_exponential

AgentType = Literal["full", "beta_sm_lesion", "pi_lesion", "alpha_lesion", "random"]

# Notation Appendix / Protocol 2: 2:1 value-to-arousal weighting for M̂(c,a)
GAMMA_V_DEFAULT = 0.6
GAMMA_A_SM_DEFAULT = 0.3
BETA_SM_DEFAULT = 0.7  # within canonical [0.1, 1.2], typical waking [0.3, 0.8]


def compute_somatic_marker_estimate(
    V: float,
    A: float,
    gamma_V: float = GAMMA_V_DEFAULT,
    gamma_A_sm: float = GAMMA_A_SM_DEFAULT,
) -> float:
    """M̂(c,a) = γ_V · V(c,a) + γ_A^SM · A(c,a) (Protocol 2)."""
    return float(gamma_V * V + gamma_A_sm * A)


@dataclass
class ThreeArmedBanditEnv:
    """Stochastic three-armed bandit with volatile hidden-state reversals.
    Reward probabilities per arm perform a bounded random walk; volatility
    sigma_env controls the per-trial jump size (Protocol 2: sigma_env in
    {0.1, 0.3, 0.6}).
    """

    sigma_env: float
    n_arms: int = 3
    rng: np.random.Generator = field(default_factory=lambda: np.random.default_rng())
    p_min: float = 0.1
    p_max: float = 0.9

    def __post_init__(self) -> None:
        self.probs = self.rng.uniform(self.p_min, self.p_max, size=self.n_arms)

    def step_env(self) -> None:
        """Advance reward probabilities by one volatility-scaled random step."""
        jump = self.rng.normal(0.0, self.sigma_env, size=self.n_arms)
        self.probs = np.clip(self.probs + jump, self.p_min, self.p_max)

    def pull(self, arm: int) -> float:
        """Return a Bernoulli(p_arm) reward in {0, 1} and advance the environment."""
        reward = float(self.rng.random() < self.probs[arm])
        self.step_env()
        return reward

    @property
    def best_arm(self) -> int:
        return int(np.argmax(self.probs))


@dataclass
class TrialLog:
    """Per-trial diagnostics for one agent run."""

    action: list[int] = field(default_factory=list)
    reward: list[float] = field(default_factory=list)
    optimal: list[bool] = field(default_factory=list)
    m_hat: list[float] = field(default_factory=list)
    ignition: list[int] = field(default_factory=list)
    pi_i_eff: list[float] = field(default_factory=list)
    pi_e: list[float] = field(default_factory=list)
    z_i: list[float] = field(default_factory=list)
    z_e: list[float] = field(default_factory=list)


class APGISomaticAgent:
    """Precision-weighted bandit agent implementing the Protocol 2 somatic
    marker model and its four lesion variants.
    Per-arm state:
        Q(a)      — value estimate (EMA of observed reward)
        A(a)      — anticipated arousal (EMA of |reward prediction error|)
        sigma2(a) — running variance of reward prediction errors -> Pi_i_baseline(a)
    Decision rule: softmax over Q(a) * Pi_i_eff(a) (precision-weighted value),
    matching the APGI principle that precision modulates the influence a
    signal (here, the value estimate) has on behavior.
    """

    def __init__(
        self,
        agent_type: AgentType,
        n_arms: int = 3,
        learning_rate: float = 0.2,
        arousal_rate: float = 0.3,
        variance_rate: float = 0.3,
        beta_sm: float = BETA_SM_DEFAULT,
        gamma_V: float = GAMMA_V_DEFAULT,
        gamma_A_sm: float = GAMMA_A_SM_DEFAULT,
        choice_temperature: float = 0.5,
        pi_min: float = 0.1,
        pi_max: float = 10.0,
        theta_ignition: float = 4.0,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.agent_type = agent_type
        self.n_arms = n_arms
        self.lr = learning_rate
        self.arousal_rate = arousal_rate
        self.variance_rate = variance_rate
        self.gamma_V = gamma_V
        self.gamma_A_sm = gamma_A_sm
        self.choice_temperature = choice_temperature
        self.pi_min = pi_min
        self.pi_max = pi_max
        self.theta_ignition = theta_ignition
        self.rng = rng or np.random.default_rng()
        # Lesion semantics (Protocol 2):
        #   full           — beta_sm active, Pi_i_baseline adaptive
        #   beta_sm_lesion — beta_sm = 0 (M_hat has no effect on precision)
        #   pi_lesion      — Pi_i_baseline frozen at 1.0 (not estimated),
        #                    beta_sm modulation still active on top of it
        #   alpha_lesion   — metabolic/exploration-cost weighting disabled
        #   random         — uniform random choice, no learning
        self.beta_sm = 0.0 if agent_type == "beta_sm_lesion" else beta_sm
        self.freeze_pi_baseline = agent_type == "pi_lesion"
        self.disable_metabolic_weighting = agent_type == "alpha_lesion"
        self.Q = np.zeros(n_arms)
        self.A = np.zeros(n_arms)
        self.sigma2 = np.ones(n_arms)
        self.log = TrialLog()

    def _pi_i_baseline(self, arm: int) -> float:
        if self.freeze_pi_baseline:
            return 1.0
        return clamp(1.0 / (self.sigma2[arm] + 1e-6), self.pi_min, self.pi_max)

    def select_action(self, trial_idx: int, n_trials: int) -> tuple[int, float, float]:
        """Return (chosen_arm, M_hat_of_chosen_arm, Pi_i_eff_of_chosen_arm)."""
        if self.agent_type == "random":
            arm = int(self.rng.integers(self.n_arms))
            m_hat = compute_somatic_marker_estimate(
                self.Q[arm], self.A[arm], self.gamma_V, self.gamma_A_sm
            )
            pi_i_eff = self._pi_i_baseline(arm)
            return arm, m_hat, pi_i_eff
        m_hats = np.array(
            [
                compute_somatic_marker_estimate(self.Q[a], self.A[a], self.gamma_V, self.gamma_A_sm)
                for a in range(self.n_arms)
            ]
        )
        pi_i_effs = np.array(
            [
                compute_interoceptive_precision_exponential(
                    self._pi_i_baseline(a), self.beta_sm, m_hats[a], self.pi_min, self.pi_max
                )
                for a in range(self.n_arms)
            ]
        )
        weighted_value = self.Q * pi_i_effs
        # alpha_lesion: metabolic weighting disabled -> exploration temperature
        # stays constant; other agents cool exploration as trials progress
        # (a simple proxy for "costly, effortful exploration under a
        # metabolic budget" per §13's C(t) coupling).
        if self.disable_metabolic_weighting:
            temp = self.choice_temperature
        else:
            progress = trial_idx / max(n_trials, 1)
            temp = self.choice_temperature * (1.0 - 0.5 * progress)
        probs = _softmax(weighted_value, temp)
        arm = int(self.rng.choice(self.n_arms, p=probs))
        return arm, float(m_hats[arm]), float(pi_i_effs[arm])

    def update(self, arm: int, reward: float, m_hat: float, pi_i_eff: float) -> None:
        """Update value/arousal/variance estimates and log ignition diagnostics."""
        delta = reward - self.Q[arm]
        self.Q[arm] += self.lr * delta
        self.A[arm] = (1 - self.arousal_rate) * self.A[arm] + self.arousal_rate * abs(delta)
        if not self.freeze_pi_baseline:
            self.sigma2[arm] = (1 - self.variance_rate) * self.sigma2[arm] + self.variance_rate * (
                delta**2
            )
        # APGI-style ignition diagnostic: exteroceptive channel = value
        # spread across arms (environmental structure signal); interoceptive
        # channel = reward prediction error magnitude (embodied consequence
        # of the choice just made). Pred 2.B/2.C need both channels and a
        # binary ignition event to test interoceptive dominance and
        # M_hat-leads-ignition timing.
        pi_e = clamp(1.0 / (np.var(self.Q) + 1e-6), self.pi_min, self.pi_max)
        z_e = float(np.std(self.Q))
        z_i = abs(delta)
        S = pi_e * abs(z_e) + pi_i_eff * abs(z_i)
        ignition = int(S > self.theta_ignition)
        self.log.action.append(arm)
        self.log.reward.append(reward)
        self.log.m_hat.append(m_hat)
        self.log.ignition.append(ignition)
        self.log.pi_i_eff.append(pi_i_eff)
        self.log.pi_e.append(pi_e)
        self.log.z_i.append(z_i)
        self.log.z_e.append(z_e)


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    temperature = max(temperature, 1e-6)
    scaled = x / temperature
    scaled = scaled - np.max(scaled)
    exp_x = np.exp(scaled)
    return exp_x / np.sum(exp_x)  # type: ignore[no-any-return]


def run_agent_on_volatility_schedule(
    agent_type: AgentType,
    volatility_schedule: list[float] | None = None,
    trials_per_block: int = 100,
    seed: int | None = None,
    **agent_kwargs: float,
) -> TrialLog:
    """Run one agent across the Protocol 2 volatility schedule
    ([0.1, 0.3, 0.6, 0.3, 0.1] across 5 blocks of 100 trials, 500 total).
    """
    if volatility_schedule is None:
        volatility_schedule = [0.1, 0.3, 0.6, 0.3, 0.1]
    rng = np.random.default_rng(seed)
    agent = APGISomaticAgent(agent_type=agent_type, rng=rng, **agent_kwargs)
    n_trials = trials_per_block * len(volatility_schedule)
    trial_idx = 0
    for sigma_env in volatility_schedule:
        env = ThreeArmedBanditEnv(sigma_env=sigma_env, rng=rng)
        for _ in range(trials_per_block):
            arm, m_hat, pi_i_eff = agent.select_action(trial_idx, n_trials)
            best_before = env.best_arm
            reward = env.pull(arm)
            agent.update(arm, reward, m_hat, pi_i_eff)
            agent.log.optimal.append(arm == best_before)
            trial_idx += 1
    return agent.log


def estimate_convergence_trial(
    optimal: list[bool], window: int = 20, threshold: float = 0.6
) -> int | None:
    """First trial index at which the trailing-window optimal-choice rate
    first exceeds `threshold`, or None if it never does (Pred 2.A:
    convergence within 50-80 trials for the full agent).
    """
    opt = np.asarray(optimal, dtype=float)
    if len(opt) < window:
        return None
    rolling = np.convolve(opt, np.ones(window) / window, mode="valid")
    hits = np.where(rolling >= threshold)[0]
    return int(hits[0] + window) if len(hits) > 0 else None


def compute_interoceptive_dominance(log: TrialLog) -> float:
    """Pred 2.B: proportion of trials where Pi_i·|z_i| > Pi_e·|z_e|
    (interoceptive precision-weighted term exceeds exteroceptive)."""
    pi_i = np.asarray(log.pi_i_eff)
    z_i = np.asarray(log.z_i)
    pi_e = np.asarray(log.pi_e)
    z_e = np.asarray(log.z_e)
    if len(pi_i) == 0:
        return float("nan")
    dominant = (pi_i * np.abs(z_i)) > (pi_e * np.abs(z_e))
    return float(np.mean(dominant))


def compute_m_hat_lead_lag(log: TrialLog, max_lag: int = 5) -> dict:
    """Pred 2.C: cross-correlate M_hat against the ignition series B_t and
    report the lag of peak correlation. A negative lag means M_hat leads
    ignition (M_hat(t) predicts B(t+|lag|)), matching the spec's prediction
    that somatic-marker retrieval precedes threshold crossing.
    """
    m_hat = np.asarray(log.m_hat, dtype=float)
    b = np.asarray(log.ignition, dtype=float)
    n = len(m_hat)
    if n < 2 * max_lag + 2:
        return {"peak_lag": None, "peak_corr": float("nan"), "leads": False}
    lags = range(-max_lag, max_lag + 1)
    corrs = []
    informative = []
    for lag in lags:
        if lag < 0:
            a, c = m_hat[: n + lag], b[-lag:]
        elif lag > 0:
            a, c = m_hat[lag:], b[: n - lag]
        else:
            a, c = m_hat, b
        if np.std(a) < 1e-9 or np.std(c) < 1e-9:
            # Undefined correlation (constant series over this lag window) —
            # not a genuine zero, so it must not compete in the argmax below.
            corrs.append(float("nan"))
            informative.append(False)
        else:
            corrs.append(float(np.corrcoef(a, c)[0, 1]))
            informative.append(True)
    if not any(informative):
        return {
            "peak_lag": None,
            "peak_corr": float("nan"),
            "leads": False,
            "all_lags": list(lags),
            "all_corrs": corrs,
            "indeterminate": True,
        }
    ranked = [c if inf else -np.inf for c, inf in zip(corrs, informative)]
    peak_idx = int(np.argmax(ranked))
    peak_lag = list(lags)[peak_idx]
    return {
        "peak_lag": peak_lag,
        "peak_corr": corrs[peak_idx],
        "leads": peak_lag < 0,
        "all_lags": list(lags),
        "all_corrs": corrs,
        "indeterminate": False,
    }


def run_protocol_2(
    seed: int | None = None,
    trials_per_block: int = 100,
    agent_types: tuple[AgentType, ...] = (
        "full",
        "beta_sm_lesion",
        "pi_lesion",
        "alpha_lesion",
        "random",
    ),
) -> dict:
    """Run all five Protocol 2 agent types on the same volatility schedule
    and compute the confirmatory statistics (Pred 2.A, 2.B, 2.C).
    Returns a dict keyed by agent_type with reward totals, convergence
    trial, interoceptive-dominance proportion, and M_hat-lead-lag results.
    """
    results: dict[str, dict] = {}
    for i, agent_type in enumerate(agent_types):
        agent_seed = None if seed is None else seed + i
        log = run_agent_on_volatility_schedule(
            agent_type, trials_per_block=trials_per_block, seed=agent_seed
        )
        results[agent_type] = {
            "total_reward": float(np.sum(log.reward)),
            "mean_reward": float(np.mean(log.reward)),
            "convergence_trial": estimate_convergence_trial(log.optimal),
            "interoceptive_dominance": compute_interoceptive_dominance(log),
            "m_hat_lead_lag": compute_m_hat_lead_lag(log),
            "ignition_rate": float(np.mean(log.ignition)) if log.ignition else float("nan"),
            "log": log,
        }
    if "full" in results and "beta_sm_lesion" in results:
        full_reward = results["full"]["total_reward"]
        lesion_reward = results["beta_sm_lesion"]["total_reward"]
        results["reward_ratio_full_vs_beta_sm_lesion"] = (
            full_reward / lesion_reward if lesion_reward != 0 else float("inf")
        )
    return results
