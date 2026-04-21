"""Observable mapping for neural and behavioral validation.

Implements extraction of neural and behavioral observables from APGI
dynamics, with key testable predictions per spec §14.

Observable Mapping (§14):
- S(t) → LFP/EEG gamma-band power
- θ(t) → P300/N200 ERP amplitude
- B(t) → Global ignition (gamma synchrony)
- RT variability, response criterion, decision rate
"""

from __future__ import annotations

import numpy as np
from scipy import signal  # type: ignore
from typing import Any


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

    def __init__(self):
        """Initialize behavioral observable extractor."""
        self.history = {
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
        response_criterion = self.extract_response_criterion(
            np.array(self.history["theta"])
        )
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
            pooled_std = np.sqrt(
                (np.std(delta_ignition) ** 2 + np.std(delta_no_ignition) ** 2) / 2
            )
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
            eigs = np.linalg.eigvals(FIM)
            condition_number = np.max(np.abs(eigs)) / (np.min(np.abs(eigs)) + 1e-8)
        except (np.linalg.LinAlgError, ValueError):
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
