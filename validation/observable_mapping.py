"""Observable mapping for neural and behavioral validation.
Implements extraction of neural and behavioral observables from APGI
dynamics, with key testable predictions per spec §14.
Observable Mapping (MathSpec §14, full table):
- S(t)     → LFP/EEG gamma-band power                  | —
- θ(t)     → P300/N200 ERP amplitude threshold          | RT variability; response criterion
- B(t)     → Global ignition (gamma synchrony)          | Overt decision / button press
- P_ign(t) → Pre-stimulus alpha-band suppression        | Hit rate in near-threshold detection
- Π_e(t)   → Gamma/beta power ratio, sensory cortex     | Perceptual sensitivity (d')
- Π_i(t)   → HRV-linked neural variability              | Interoceptive accuracy task score
- β_DA(t)  → Striatal BOLD signal                       | Reward expectation bias
- g_NE(t)  → Pupil diameter (LC-NE proxy)                | RT distribution shape (inverted U)
- g_ACh(t) → EEG alpha power (inverse)                   | Cued attentional modulation
- Δ(t)     → Pre-decision distance to neural boundary    | Decision confidence rating
See OBSERVABLE_MAPPING_TABLE below for this table as structured data.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from scipy import signal  # type: ignore

# Suppress LAPACK warnings
warnings.filterwarnings("ignore", message=".*On entry to DLASCL.*")


OBSERVABLE_MAPPING_TABLE: tuple[dict[str, str], ...] = (
    {
        "apgi_variable": "S(t)",
        "neural_observable": "LFP/EEG gamma-band power",
        "behavioral_observable": "—",
        "method": "Time-frequency (TF) analysis",
        "extractor": "NeuralObservableExtractor.extract_gamma_power",
    },
    {
        "apgi_variable": "theta(t)",
        "neural_observable": "P300/N200 ERP amplitude threshold",
        "behavioral_observable": "RT variability; response criterion",
        "method": "ERP; signal detection theory (d')",
        "extractor": "NeuralObservableExtractor.extract_erp_amplitude / "
        "BehavioralObservableExtractor.extract_rt_variability, extract_response_criterion",
    },
    {
        "apgi_variable": "B(t)",
        "neural_observable": "Global ignition (widespread gamma synchrony)",
        "behavioral_observable": "Overt decision / button press",
        "method": "Single-trial EEG ignition markers",
        "extractor": "NeuralObservableExtractor.extract_ignition_rate / "
        "BehavioralObservableExtractor.extract_decision_rate",
    },
    {
        "apgi_variable": "P_ign(t)",
        "neural_observable": "Pre-stimulus alpha-band suppression",
        "behavioral_observable": "Hit rate in near-threshold detection",
        "method": "Single-trial EEG TF decomposition",
        "extractor": "NeuralObservableExtractor.extract_prestimulus_alpha_suppression",
    },
    {
        "apgi_variable": "Pi_e(t)",
        "neural_observable": "Gamma/beta power ratio, sensory cortex",
        "behavioral_observable": "Perceptual sensitivity (d')",
        "method": "TF decomposition",
        "extractor": "NeuralObservableExtractor.extract_gamma_beta_ratio / "
        "BehavioralObservableExtractor.extract_perceptual_sensitivity",
    },
    {
        "apgi_variable": "Pi_i(t)",
        "neural_observable": "HRV-linked neural variability",
        "behavioral_observable": "Interoceptive accuracy task score",
        "method": "HRV + cardiac timing",
        "extractor": "NeuralObservableExtractor.extract_hrv_proxy / "
        "BehavioralObservableExtractor.extract_interoceptive_accuracy",
    },
    {
        "apgi_variable": "beta_DA(t)",
        "neural_observable": "Striatal BOLD signal",
        "behavioral_observable": "Reward expectation bias",
        "method": "fMRI; pupillometry",
        "extractor": "NeuralObservableExtractor.extract_striatal_bold_proxy / "
        "BehavioralObservableExtractor.extract_reward_expectation_bias",
    },
    {
        "apgi_variable": "g_NE(t)",
        "neural_observable": "Pupil diameter (LC-NE proxy)",
        "behavioral_observable": "RT distribution shape (inverted U)",
        "method": "Pupillometry",
        "extractor": "NeuralObservableExtractor.extract_pupil_diameter_proxy / "
        "BehavioralObservableExtractor.extract_rt_distribution_shape",
    },
    {
        "apgi_variable": "g_ACh(t)",
        "neural_observable": "EEG alpha power (inverse)",
        "behavioral_observable": "Cued attentional modulation",
        "method": "EEG alpha power",
        "extractor": "NeuralObservableExtractor.extract_alpha_power_proxy / "
        "BehavioralObservableExtractor.extract_cued_attention_modulation",
    },
    {
        "apgi_variable": "Delta(t)",
        "neural_observable": "Pre-decision distance to neural boundary",
        "behavioral_observable": "Decision confidence rating",
        "method": "EEG + confidence",
        "extractor": "KeyTestablePredictionValidator (delta history) / "
        "BehavioralObservableExtractor.extract_decision_confidence",
    },
)


def get_observable_mapping_table() -> tuple[dict[str, str], ...]:
    """Return the full APGI Variable -> Neural/Behavioral Observable mapping
    table (MathSpec §14) as structured data."""
    return OBSERVABLE_MAPPING_TABLE


class NeuralObservableExtractor:
    """Extract neural observables from APGI state variables.
    Maps internal variables to measurable neural signals:
    - S(t) → gamma-band power (30-100 Hz)
    - θ(t) → P300/N200 amplitude
    - B(t) → global ignition (gamma synchrony)
    """

    def __init__(self, fs: float = 100.0):
        """Initialize neural observable extractor.
        Args:
            fs: Sampling frequency (Hz)
        """
        self.fs = fs
        self.history: dict[str, list[float]] = {
            "S": [],
            "theta": [],
            "B": [],
            "gamma_power": [],
            "erp_amplitude": [],
            "ignition_rate": [],
        }

    def extract_gamma_power(
        self,
        S_history: np.ndarray,
        freq_range: tuple[float, float] = (30, 100),
    ) -> float:
        """Extract gamma-band power from signal history.
        Spec §14: S(t) → LFP/EEG gamma-band power
        Computes power spectral density and integrates over gamma band.
        Args:
            S_history: Signal history array
            freq_range: Frequency range for gamma band (Hz)
        Returns:
            Gamma-band power (normalized)
        """
        if len(S_history) < 64:
            return 0.0
        # Compute power spectral density via Welch method
        freqs, psd = signal.welch(S_history, fs=self.fs, nperseg=64)
        # Extract gamma band
        gamma_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        if np.any(gamma_mask):
            gamma_power = np.mean(psd[gamma_mask])
        else:
            gamma_power = 0.0
        return float(gamma_power)

    def extract_erp_amplitude(
        self,
        theta_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract ERP-like amplitude from threshold dynamics.
        Spec §14: θ(t) → P300/N200 ERP amplitude
        Uses threshold as proxy for ERP amplitude. Computes
        peak-to-baseline difference in recent window.
        Args:
            theta_history: Threshold history array
            window_size: Window size for amplitude computation
        Returns:
            ERP amplitude (peak deviation from baseline)
        """
        if len(theta_history) < window_size:
            if len(theta_history) > 0:
                return float(np.mean(theta_history))
            return 0.0
        recent = theta_history[-window_size:]
        baseline_array = theta_history[:-window_size]
        baseline = np.mean(baseline_array) if len(baseline_array) > 0 else 0.0
        peak = np.max(recent)
        erp_amplitude = peak - baseline
        return float(erp_amplitude)

    def extract_ignition_rate(
        self,
        B_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract ignition rate (global synchrony proxy).
        Spec §14: B(t) → Global ignition (gamma synchrony)
        Computes proportion of ignition events in recent window.
        Args:
            B_history: Binary ignition history
            window_size: Window size for rate computation
        Returns:
            Ignition rate (0-1)
        """
        if len(B_history) < window_size:
            if len(B_history) > 0:
                return float(np.mean(B_history))
            return 0.0
        recent = B_history[-window_size:]
        ignition_rate = np.mean(recent)
        return float(ignition_rate)

    def extract_prestimulus_alpha_suppression(
        self,
        p_ign_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract pre-stimulus alpha-band suppression proxy.
        Spec §14: P_ign(t) → Pre-stimulus alpha-band suppression
        Higher ignition probability is associated with GREATER pre-stimulus
        alpha suppression (lower raw alpha power) — this proxy returns the
        windowed mean P_ign(t) directly, since alpha suppression and P_ign
        are monotonically related by construction of the mapping (higher
        suppression = higher P_ign).
        """
        if len(p_ign_history) == 0:
            return 0.0
        recent = p_ign_history[-window_size:]
        return float(np.mean(recent))

    def extract_gamma_beta_ratio(
        self,
        pi_e_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract gamma/beta power ratio proxy from exteroceptive precision.
        Spec §14: Π_e(t) → Gamma/beta power ratio, sensory cortex
        """
        if len(pi_e_history) == 0:
            return 0.0
        recent = pi_e_history[-window_size:]
        return float(np.mean(recent))

    def extract_hrv_proxy(
        self,
        pi_i_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract HRV-linked neural variability proxy from interoceptive precision.
        Spec §14: Π_i(t) → HRV-linked neural variability
        """
        if len(pi_i_history) == 0:
            return 0.0
        recent = pi_i_history[-window_size:]
        return float(np.mean(recent))

    def extract_striatal_bold_proxy(
        self,
        beta_da_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract striatal BOLD signal proxy from dopaminergic bias.
        Spec §14: β_DA(t) → Striatal BOLD signal
        """
        if len(beta_da_history) == 0:
            return 0.0
        recent = beta_da_history[-window_size:]
        return float(np.mean(recent))

    def extract_pupil_diameter_proxy(
        self,
        g_ne_history: np.ndarray,
        window_size: int = 50,
    ) -> float:
        """Extract pupil diameter (LC-NE) proxy from noradrenergic gain.
        Spec §14: g_NE(t) → Pupil diameter (LC-NE proxy)
        """
        if len(g_ne_history) == 0:
            return 0.0
        recent = g_ne_history[-window_size:]
        return float(np.mean(recent))

    def extract_alpha_power_proxy(
        self,
        g_ach_history: np.ndarray,
        window_size: int = 50,
        eps: float = 1e-6,
    ) -> float:
        """Extract EEG alpha power proxy from cholinergic gain (inverse
        relationship: ACh suppresses alpha power).
        Spec §14: g_ACh(t) → EEG alpha power (inverse)
        """
        if len(g_ach_history) == 0:
            return 0.0
        recent = np.asarray(g_ach_history[-window_size:], dtype=float)
        return float(np.mean(1.0 / (recent + eps)))

    def step(
        self,
        S: float,
        theta: float,
        B: int,
    ) -> dict[str, float]:
        """Update observable extraction with current state.
        Args:
            S: Current signal value
            theta: Current threshold
            B: Current ignition (0 or 1)
        Returns:
            Dictionary with extracted observables
        """
        self.history["S"].append(S)
        self.history["theta"].append(theta)
        self.history["B"].append(B)
        # Extract observables
        gamma_power = self.extract_gamma_power(np.array(self.history["S"]))
        erp_amplitude = self.extract_erp_amplitude(np.array(self.history["theta"]))
        ignition_rate = self.extract_ignition_rate(np.array(self.history["B"]))
        self.history["gamma_power"].append(gamma_power)
        self.history["erp_amplitude"].append(erp_amplitude)
        self.history["ignition_rate"].append(ignition_rate)
        return {
            "gamma_power": gamma_power,
            "erp_amplitude": erp_amplitude,
            "ignition_rate": ignition_rate,
        }

    def get_history(self) -> dict[str, list]:
        """Get full observable history."""
        return self.history.copy()


class BehavioralObservableExtractor:
    """Extract behavioral observables from APGI dynamics.
    Maps internal variables to behavioral measures:
    - S(t) → Perceptual sensitivity (d')
    - θ(t) → RT variability, response criterion
    - B(t) → Overt decision/button press
    """

    def __init__(self) -> None:
        """Initialize behavioral observable extractor."""
        self.history: dict[str, list[float]] = {
            "S": [],
            "theta": [],
            "B": [],
            "rt_variability": [],
            "response_criterion": [],
            "decision_rate": [],
        }

    def extract_rt_variability(
        self,
        theta_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract RT variability from threshold dynamics.
        Spec §14: θ(t) → RT variability
        Computes standard deviation of threshold changes,
        proxy for reaction time variability.
        Args:
            theta_history: Threshold history
            window_size: Window for variability computation
        Returns:
            RT variability (std of threshold changes)
        """
        if len(theta_history) < window_size + 1:
            return 0.0
        recent = theta_history[-window_size:]
        theta_diff = np.diff(recent)
        rt_variability = np.std(theta_diff)
        return float(rt_variability)

    def extract_response_criterion(
        self,
        theta_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract response criterion from threshold baseline.
        Spec §14: θ(t) → Response criterion
        Uses mean threshold as proxy for response criterion
        (higher threshold = more conservative).
        Args:
            theta_history: Threshold history
            window_size: Window for criterion computation
        Returns:
            Response criterion (mean threshold)
        """
        if len(theta_history) < window_size:
            if len(theta_history) > 0:
                return float(np.mean(theta_history))
            return 0.0
        recent = theta_history[-window_size:]
        response_criterion = np.mean(recent)
        return float(response_criterion)

    def extract_decision_rate(
        self,
        B_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract decision rate from ignition events.
        Spec §14: B(t) → Overt decision/button press
        Computes proportion of ignition events (decisions).
        Args:
            B_history: Binary ignition history
            window_size: Window for rate computation
        Returns:
            Decision rate (0-1)
        """
        if len(B_history) < window_size:
            if len(B_history) > 0:
                return float(np.mean(B_history))
            return 0.0
        recent = B_history[-window_size:]
        decision_rate = np.mean(recent)
        return float(decision_rate)

    def extract_perceptual_sensitivity(
        self,
        pi_e_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract perceptual sensitivity (d') proxy from exteroceptive precision.
        Spec §14: Π_e(t) → Perceptual sensitivity (d')
        """
        if len(pi_e_history) == 0:
            return 0.0
        recent = pi_e_history[-window_size:]
        return float(np.mean(recent))

    def extract_interoceptive_accuracy(
        self,
        pi_i_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract interoceptive accuracy task score proxy from interoceptive precision.
        Spec §14: Π_i(t) → Interoceptive accuracy task score
        """
        if len(pi_i_history) == 0:
            return 0.0
        recent = pi_i_history[-window_size:]
        return float(np.mean(recent))

    def extract_reward_expectation_bias(
        self,
        beta_da_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract reward expectation bias proxy from dopaminergic bias.
        Spec §14: β_DA(t) → Reward expectation bias
        """
        if len(beta_da_history) == 0:
            return 0.0
        recent = beta_da_history[-window_size:]
        return float(np.mean(recent))

    def extract_rt_distribution_shape(
        self,
        g_ne_history: np.ndarray,
        window_size: int = 100,
        optimal_g_ne: float = 1.0,
    ) -> float:
        """Extract RT-distribution-shape proxy from noradrenergic gain.
        Spec §14: g_NE(t) → RT distribution shape (inverted U)
        The Yerkes-Dodson-style inverted-U relationship between arousal
        (g_NE) and performance is modeled as a quadratic penalty around
        optimal_g_ne: performance (inverse of RT-shape distortion) peaks at
        optimal_g_ne and degrades symmetrically on either side. Returns a
        [0, 1] "distortion" score where 0 = optimal, larger = more distorted.
        """
        if len(g_ne_history) == 0:
            return 0.0
        recent = np.asarray(g_ne_history[-window_size:], dtype=float)
        distortion = float(np.mean((recent - optimal_g_ne) ** 2))
        return distortion

    def extract_cued_attention_modulation(
        self,
        g_ach_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract cued attentional modulation proxy from cholinergic gain.
        Spec §14: g_ACh(t) → Cued attentional modulation
        """
        if len(g_ach_history) == 0:
            return 0.0
        recent = g_ach_history[-window_size:]
        return float(np.mean(recent))

    def extract_decision_confidence(
        self,
        delta_history: np.ndarray,
        window_size: int = 100,
    ) -> float:
        """Extract decision confidence rating proxy from the ignition margin Δ(t).
        Spec §14: Δ(t) → Decision confidence rating
        Confidence increases with |Δ(t) = S(t) - θ(t)|, the pre-decision
        distance to the ignition boundary; bounded to [0, 1) via tanh.
        """
        if len(delta_history) == 0:
            return 0.0
        recent = np.asarray(delta_history[-window_size:], dtype=float)
        return float(np.mean(np.tanh(np.abs(recent))))

    def step(
        self,
        S: float,
        theta: float,
        B: int,
    ) -> dict[str, float]:
        """Update behavioral observable extraction.
        Args:
            S: Current signal value
            theta: Current threshold
            B: Current ignition (0 or 1)
        Returns:
            Dictionary with extracted observables
        """
        self.history["S"].append(S)
        self.history["theta"].append(theta)
        self.history["B"].append(B)
        # Extract observables
        rt_variability = self.extract_rt_variability(np.array(self.history["theta"]))
        response_criterion = self.extract_response_criterion(np.array(self.history["theta"]))
        decision_rate = self.extract_decision_rate(np.array(self.history["B"]))
        self.history["rt_variability"].append(rt_variability)
        self.history["response_criterion"].append(response_criterion)
        self.history["decision_rate"].append(decision_rate)
        return {
            "rt_variability": rt_variability,
            "response_criterion": response_criterion,
            "decision_rate": decision_rate,
        }

    def get_history(self) -> dict[str, list]:
        """Get full observable history."""
        return self.history.copy()


class KeyTestablePredictionValidator:
    """Validate key testable prediction from spec §14.
    Prediction: Hit rate ∝ P_ign(t) = σ(Δ(t) / τ_σ)
    where Δ(t) = S(t) - θ(t) is the ignition margin.
    The margin should outperform signal alone as predictor of hits.
    """

    def __init__(self, tau_sigma: float = 0.5):
        """Initialize prediction validator.
        Args:
            tau_sigma: Sigmoid temperature for soft ignition
        """
        self.tau_sigma = tau_sigma
        self.history: dict[str, list[float]] = {
            "S": [],
            "theta": [],
            "B": [],
            "delta": [],
            "p_ign": [],
        }

    def step(
        self,
        S: float,
        theta: float,
        B: int,
    ) -> dict[str, float]:
        """Update prediction validator.
        Args:
            S: Current signal
            theta: Current threshold
            B: Current ignition (0 or 1)
        Returns:
            Dictionary with margin and soft ignition probability
        """
        self.history["S"].append(S)
        self.history["theta"].append(theta)
        self.history["B"].append(B)
        # Compute margin
        delta = S - theta
        self.history["delta"].append(delta)
        # Soft ignition probability via sigmoid
        p_ign = 1.0 / (1.0 + np.exp(-delta / self.tau_sigma))
        self.history["p_ign"].append(p_ign)
        return {
            "delta": delta,
            "p_ign": p_ign,
        }

    def validate(self) -> dict[str, Any]:
        """Validate key prediction against data.
        Computes correlation of margin vs signal with ignition events.
        Returns:
            Dictionary with validation results
        """
        if len(self.history["B"]) < 100:
            return {
                "valid": False,
                "reason": "Insufficient data (need >= 100 samples)",
            }
        B = np.array(self.history["B"])
        S = np.array(self.history["S"])
        delta = np.array(self.history["delta"])
        p_ign = np.array(self.history["p_ign"])
        # Compute correlations
        corr_margin = np.corrcoef(delta, B)[0, 1]
        corr_signal = np.corrcoef(S, B)[0, 1]
        corr_p_ign = np.corrcoef(p_ign, B)[0, 1]
        # Check if margin outperforms signal
        margin_better = corr_margin > corr_signal
        improvement = corr_margin - corr_signal
        # Compute effect size (Cohen's d)
        B_ignition = B[B == 1]
        B_no_ignition = B[B == 0]
        if len(B_ignition) > 0 and len(B_no_ignition) > 0:
            delta_ignition = delta[B == 1]
            delta_no_ignition = delta[B == 0]
            mean_diff = np.mean(delta_ignition) - np.mean(delta_no_ignition)
            pooled_std = np.sqrt((np.std(delta_ignition) ** 2 + np.std(delta_no_ignition) ** 2) / 2)
            cohens_d = mean_diff / (pooled_std + 1e-8)
        else:
            cohens_d = 0.0
        return {
            "valid": True,
            "correlation_margin": float(corr_margin),
            "correlation_signal": float(corr_signal),
            "correlation_p_ign": float(corr_p_ign),
            "margin_better": bool(margin_better),
            "improvement": float(improvement),
            "cohens_d": float(cohens_d),
            "n_samples": len(B),
            "ignition_rate": float(np.mean(B)),
        }

    def get_history(self) -> dict[str, list]:
        """Get full history."""
        return self.history.copy()


class ParameterIdentifiabilityAnalyzer:
    """Analyze parameter identifiability constraints.
    Spec §14.4: Three constraints to break degeneracy
    Implements identifiability checks to ensure parameters
    can be uniquely recovered from observable data.
    """

    @staticmethod
    def compute_fisher_information(
        S_history: np.ndarray,
        theta_history: np.ndarray,
        B_history: np.ndarray,
        params: dict[str, float],
    ) -> dict[str, Any]:
        """Compute Fisher information matrix for parameter estimation.
        Args:
            S_history: Signal history
            theta_history: Threshold history
            B_history: Ignition history
            params: Parameter dictionary
        Returns:
            Dictionary with Fisher information and identifiability metrics
        """
        # Compute log-likelihood gradients w.r.t. key parameters
        # This is a simplified version; full implementation would use
        # automatic differentiation
        n = len(B_history)
        # Parameters of interest
        tau_sigma = params.get("ignite_tau", 0.5)
        # Compute numerical gradients
        eps = 1e-6
        # Gradient w.r.t. lam
        grad_lam = np.zeros(n)
        for t in range(1, n):
            if S_history[t] > 0:
                grad_lam[t] = S_history[t - 1] / (S_history[t] + eps)
        # Gradient w.r.t. eta
        grad_eta = np.zeros(n)
        for t in range(1, n):
            grad_eta[t] = theta_history[t - 1]
        # Gradient w.r.t. tau_sigma
        grad_tau = np.zeros(n)
        for t in range(n):
            delta = S_history[t] - theta_history[t]
            p_ign = 1.0 / (1.0 + np.exp(-delta / tau_sigma))
            grad_tau[t] = p_ign * (1 - p_ign) * delta / (tau_sigma**2 + eps)
        # Fisher information matrix (simplified)
        FIM = np.array(
            [
                [
                    np.sum(grad_lam**2),
                    np.sum(grad_lam * grad_eta),
                    np.sum(grad_lam * grad_tau),
                ],
                [
                    np.sum(grad_eta * grad_lam),
                    np.sum(grad_eta**2),
                    np.sum(grad_eta * grad_tau),
                ],
                [
                    np.sum(grad_tau * grad_lam),
                    np.sum(grad_tau * grad_eta),
                    np.sum(grad_tau**2),
                ],
            ]
        )
        # Compute condition number (identifiability measure)
        try:
            with np.errstate(all="ignore"):
                eigs = np.linalg.eigvals(FIM)
                eig_abs = np.abs(eigs)
                eig_abs = np.maximum(eig_abs, 1e-12)  # Prevent division by near-zero
                condition_number = np.max(eig_abs) / np.min(eig_abs)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            condition_number = np.inf
        # Compute Cramér-Rao lower bound
        try:
            crlb = np.linalg.inv(FIM)
            crlb_diag = np.diag(crlb)
        except (np.linalg.LinAlgError, ValueError):
            crlb_diag = np.array([np.inf, np.inf, np.inf])
        return {
            "fisher_information": FIM,
            "condition_number": float(condition_number),
            "crlb_diag": crlb_diag.tolist(),
            "identifiable": condition_number < 1e6,
        }

    @staticmethod
    def check_identifiability_constraints(
        config: dict,
    ) -> dict[str, Any]:
        """Check three identifiability constraints per spec §14.4.
        Constraint 1: lam and tau_s must be distinct
        Constraint 2: eta and delta must be distinct
        Constraint 3: tau_sigma must be > 0
        Args:
            config: Configuration dictionary
        Returns:
            Dictionary with constraint check results
        """
        lam = config.get("lam", 0.2)
        tau_s = config.get("tau_s", 5.0)
        eta = config.get("eta", 0.1)
        delta = config.get("delta", 0.5)
        tau_sigma = config.get("ignite_tau", 0.5)
        # Constraint 1: lam and tau_s distinct
        # lam ≈ dt/tau_s, so they should be different scales
        constraint1 = abs(lam - 1.0 / tau_s) > 0.01
        # Constraint 2: eta and delta distinct
        constraint2 = abs(eta - delta) > 0.01
        # Constraint 3: tau_sigma > 0
        constraint3 = tau_sigma > 0
        return {
            "constraint1_lam_tau_s_distinct": bool(constraint1),
            "constraint2_eta_delta_distinct": bool(constraint2),
            "constraint3_tau_sigma_positive": bool(constraint3),
            "all_satisfied": bool(constraint1 and constraint2 and constraint3),
        }
