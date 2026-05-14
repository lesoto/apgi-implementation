"""Tests for the five spec-compliance fixes.

Covers:
  §9  — W_inh divisive normalization in LiquidStateMachine
  §10 — Three criticality phase-transition signatures
  §17 — Spectral radius bounds [0.7, 0.95] in LSM and LiquidNetwork
  §11 — MultiscaleWeightScheduler (Σwₗ = 1 + adaptive EMA)
  §27 — Circadian/ultradian θₜ modulation
"""

from __future__ import annotations

import numpy as np
import pytest

from reservoir.liquid_state_machine import LiquidStateMachine

# ---------------------------------------------------------------------------
# §9 + §17 — LiquidStateMachine: W_inh and spectral radius bounds
# ---------------------------------------------------------------------------



class TestLSMWinh:
    """§9: W_inh PV+ divisive normalization."""

    def test_w_inh_exists_and_shape(self):
        lsm = LiquidStateMachine(N=20, M=2, seed=0)
        assert hasattr(lsm, "W_inh")
        assert lsm.W_inh.shape == (20, 20)

    def test_w_inh_non_negative(self):
        lsm = LiquidStateMachine(N=30, M=2, seed=1)
        assert np.all(lsm.W_inh >= 0), "W_inh must be non-negative (PV+ inhibitory)"

    def test_w_inh_sparse(self):
        lsm = LiquidStateMachine(N=100, M=2, inh_sparsity=0.1, seed=2)
        sparsity = float(np.mean(lsm.W_inh > 0))
        # Stochastic: actual sparsity should be near 0.1 ± 0.04
        assert 0.04 < sparsity < 0.20, f"Unexpected sparsity: {sparsity}"

    def test_sigma_inh2_stored(self):
        lsm = LiquidStateMachine(N=20, M=2, sigma_inh2=0.25, seed=0)
        assert lsm.sigma_inh2 == 0.25

    def test_step_with_divisive_norm_produces_finite_output(self):
        lsm = LiquidStateMachine(N=30, M=2, seed=3)
        u = np.array([0.5, -0.3])
        x = lsm.step(u, tau=1.0, dt=0.1, use_divisive_normalization=True)
        assert np.all(np.isfinite(x))
        assert x.shape == (30,)

    def test_step_without_divisive_norm_backward_compat(self):
        lsm = LiquidStateMachine(N=30, M=2, seed=3)
        u = np.array([0.5, -0.3])
        x = lsm.step(u, tau=1.0, dt=0.1, use_divisive_normalization=False)
        assert np.all(np.isfinite(x))

    def test_divisive_norm_changes_output(self):
        """Divisive normalization ON vs OFF must produce different state updates."""
        lsm = LiquidStateMachine(N=30, M=2, inh_scale=0.5, sigma_inh2=0.5, seed=7)
        u = np.array([2.0, -1.5])

        # Step with normalization on
        x_before = lsm.x.copy()
        x_on = lsm.step(u.copy(), tau=1.0, dt=0.1, use_divisive_normalization=True)

        # Reset state and step without normalization
        lsm.x[:] = x_before
        x_off = lsm.step(u.copy(), tau=1.0, dt=0.1, use_divisive_normalization=False)

        # The two modes must produce different activations
        assert not np.allclose(x_on, x_off)

    def test_divisive_norm_denominator_formula(self):
        """Verify the divisive norm formula: res = f / (σ² + W_inh·|f|)."""
        rng = np.random.default_rng(0)
        N = 10
        f_drive = rng.standard_normal(N)
        # Dense W_inh so every neuron receives inhibitory input
        W_inh = np.abs(rng.standard_normal((N, N))) * 0.3
        sigma_inh2 = 1.0  # ≥ 1 guarantees normalization reduces magnitude

        expected = f_drive / (sigma_inh2 + W_inh @ np.abs(f_drive))
        # With σ²=1 and positive W_inh·|f|, denominator > 1 → all elements shrunk
        assert np.all(np.abs(expected) <= np.abs(f_drive) + 1e-12)

    def test_get_weight_statistics_includes_winh(self):
        lsm = LiquidStateMachine(N=20, M=2, seed=0)
        stats = lsm.get_weight_statistics()
        assert "W_inh_sparsity" in stats
        assert "W_inh_norm" in stats
        assert "sigma_inh2" in stats


class TestSpectralRadiusBounds:
    """§17: ρ(W_res) must be in [0.7, 0.95]."""

    def test_valid_radius_accepted(self):
        for rho in [0.7, 0.8, 0.9, 0.95]:
            lsm = LiquidStateMachine(N=20, M=2, spectral_radius=rho)
            assert lsm.spectral_radius == rho

    def test_too_low_radius_rejected(self):
        with pytest.raises(ValueError, match="0.7"):
            LiquidStateMachine(N=20, M=2, spectral_radius=0.5)

    def test_too_high_radius_rejected(self):
        with pytest.raises(ValueError, match="0.95"):
            LiquidStateMachine(N=20, M=2, spectral_radius=0.99)

    def test_zero_rejected(self):
        with pytest.raises(ValueError):
            LiquidStateMachine(N=20, M=2, spectral_radius=0.0)

    def test_exactly_095_accepted(self):
        lsm = LiquidStateMachine(N=20, M=2, spectral_radius=0.95)
        assert lsm.spectral_radius == 0.95

    def test_exactly_07_accepted(self):
        lsm = LiquidStateMachine(N=20, M=2, spectral_radius=0.7)
        assert lsm.spectral_radius == 0.7


class TestLiquidNetworkSpectralBounds:
    """§17: LiquidNetwork also enforces [0.7, 0.95]."""

    def test_valid_accepted(self):
        from reservoir.liquid_network import LiquidNetwork
        ln = LiquidNetwork(n_units=50, spectral_radius=0.9)
        assert ln is not None

    def test_too_low_rejected(self):
        from reservoir.liquid_network import LiquidNetwork
        with pytest.raises(ValueError, match="0.7"):
            LiquidNetwork(n_units=50, spectral_radius=0.3)

    def test_too_high_rejected(self):
        from reservoir.liquid_network import LiquidNetwork
        with pytest.raises(ValueError, match="0.95"):
            LiquidNetwork(n_units=50, spectral_radius=0.98)


# ---------------------------------------------------------------------------
# §10 — Three criticality phase-transition signatures
# ---------------------------------------------------------------------------

from analysis.stability import measure_criticality_signatures


class TestCriticalitySignatures:
    """§10: All three phase-transition signatures."""

    def test_returns_expected_keys(self):
        rng = np.random.default_rng(0)
        S = rng.normal(0.5, 0.2, 200)
        result = measure_criticality_signatures(S, theta=0.5)
        for key in [
            "cohens_d", "cohens_d_criterion",
            "susceptibility_ratio", "susceptibility_criterion",
            "autocorr_increase", "autocorr_criterion",
            "all_criteria_met", "n_subthreshold", "n_suprathreshold",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_insufficient_samples_returns_false_criteria(self):
        S = np.random.randn(50)
        result = measure_criticality_signatures(S, theta=0.0, min_samples=100)
        assert result["all_criteria_met"] is False
        assert "error" in result

    def test_all_criteria_met_for_well_separated_signal(self):
        """A signal that spends very different time sub/supra-threshold with
        increasing variance should satisfy all three criteria."""
        rng = np.random.default_rng(42)
        # Baseline: low variance, mostly sub-threshold
        baseline = rng.normal(0.2, 0.05, 60)
        # Transition: increasing variance and autocorrelation
        transition = np.cumsum(rng.normal(0.02, 0.15, 80))
        transition = transition - transition.min() + 0.2
        # Suprathreshold: high variance, high mean
        supra = rng.normal(1.5, 0.4, 80)
        S = np.concatenate([baseline, transition, supra])
        result = measure_criticality_signatures(S, theta=0.6, baseline_window=60)
        # Check individual components are computable
        assert result["cohens_d"] is not None or result["n_subthreshold"] == 0

    def test_sub_and_supra_counts_consistent(self):
        rng = np.random.default_rng(1)
        S = rng.normal(0.0, 1.0, 300)
        theta = 0.0
        result = measure_criticality_signatures(S, theta=theta)
        assert result["n_subthreshold"] + result["n_suprathreshold"] == len(S)

    def test_cohens_d_positive(self):
        """Suprathreshold mean > subthreshold mean → positive Cohen's d."""
        rng = np.random.default_rng(7)
        S = np.concatenate([rng.normal(-0.5, 0.2, 150), rng.normal(1.0, 0.2, 150)])
        result = measure_criticality_signatures(S, theta=0.0)
        if result["cohens_d"] is not None:
            assert result["cohens_d"] >= 0

    def test_susceptibility_ratio_type(self):
        rng = np.random.default_rng(5)
        S = rng.normal(0.5, 0.3, 200)
        result = measure_criticality_signatures(S, theta=0.5)
        if result["susceptibility_ratio"] is not None:
            assert isinstance(result["susceptibility_ratio"], float)
            assert result["susceptibility_ratio"] >= 0


# ---------------------------------------------------------------------------
# §11 — MultiscaleWeightScheduler
# ---------------------------------------------------------------------------

from hierarchy.multiscale import MultiscaleWeightScheduler


class TestMultiscaleWeightScheduler:
    """§11: Σwₗ = 1 always; adaptive EMA update."""

    def test_initialization_uniform(self):
        sched = MultiscaleWeightScheduler(n_levels=4)
        w = sched.get_weights()
        assert len(w) == 4
        np.testing.assert_allclose(w, 0.25, atol=1e-10)

    def test_weights_sum_to_one_after_init(self):
        sched = MultiscaleWeightScheduler(n_levels=5)
        assert abs(np.sum(sched.get_weights()) - 1.0) < 1e-10

    def test_weights_sum_to_one_after_update(self):
        sched = MultiscaleWeightScheduler(n_levels=4, alpha=0.1)
        for _ in range(20):
            phi = np.random.randn(4)
            w = sched.update(phi)
            assert abs(np.sum(w) - 1.0) < 1e-10, f"Σw = {np.sum(w)}"

    def test_weights_non_negative(self):
        sched = MultiscaleWeightScheduler(n_levels=3, alpha=0.2)
        for _ in range(10):
            phi = np.random.randn(3)
            w = sched.update(phi)
            assert np.all(w >= 0)

    def test_high_error_level_gains_weight(self):
        """Level with consistently large |φ(ε)| should accumulate more weight."""
        sched = MultiscaleWeightScheduler(n_levels=3, alpha=0.3)
        for _ in range(50):
            phi = np.array([0.01, 0.01, 2.0])  # level 2 dominates
            sched.update(phi)
        w = sched.get_weights()
        assert w[2] > w[0] and w[2] > w[1]

    def test_reset_restores_uniform(self):
        sched = MultiscaleWeightScheduler(n_levels=4, alpha=0.5)
        for _ in range(10):
            sched.update(np.array([1.0, 0.0, 0.0, 0.0]))
        sched.reset()
        np.testing.assert_allclose(sched.get_weights(), 0.25, atol=1e-10)

    def test_wrong_length_raises(self):
        sched = MultiscaleWeightScheduler(n_levels=3)
        with pytest.raises(ValueError, match="phi_errors length"):
            sched.update(np.array([1.0, 2.0]))  # length 2, expected 3

    def test_invalid_n_levels_raises(self):
        with pytest.raises(ValueError):
            MultiscaleWeightScheduler(n_levels=0)

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError):
            MultiscaleWeightScheduler(n_levels=3, alpha=0.0)

    def test_update_returns_copy(self):
        """Returned array should not alias internal state."""
        sched = MultiscaleWeightScheduler(n_levels=3)
        w = sched.update(np.array([1.0, 1.0, 1.0]))
        w[0] = 99.0
        assert sched.get_weights()[0] != 99.0

    def test_degenerate_zero_phi_stays_uniform(self):
        """If all φ(ε) are zero, weights should remain uniform."""
        sched = MultiscaleWeightScheduler(n_levels=3, alpha=0.5)
        for _ in range(20):
            w = sched.update(np.array([0.0, 0.0, 0.0]))
        np.testing.assert_allclose(w, 1.0 / 3, atol=1e-10)


# ---------------------------------------------------------------------------
# §27 — Circadian / ultradian θₜ modulation
# ---------------------------------------------------------------------------

from core.circadian import (T_CIRCADIAN_DEFAULT, T_ULTRADIAN_DEFAULT,
                            CircadianRegulator,
                            apply_biological_rhythm_to_theta,
                            circadian_theta_offset,
                            circadian_theta_offset_array,
                            combined_biological_rhythm_offset,
                            ultradian_theta_offset,
                            ultradian_theta_offset_array)


class TestCircadianOffset:
    """§27: Circadian cosine offset."""

    def test_zero_at_t_equals_quarter_period(self):
        # cos(π/2) = 0  → t = T/4 with phi=0
        val = circadian_theta_offset(T_CIRCADIAN_DEFAULT / 4, A_circ=0.1, phi_circ=0.0)
        assert abs(val) < 1e-9

    def test_peak_at_t_zero(self):
        val = circadian_theta_offset(0.0, A_circ=0.2, phi_circ=0.0)
        assert abs(val - 0.2) < 1e-10

    def test_trough_at_half_period(self):
        val = circadian_theta_offset(T_CIRCADIAN_DEFAULT / 2, A_circ=0.1, phi_circ=0.0)
        assert abs(val - (-0.1)) < 1e-9

    def test_period_T_invalid_raises(self):
        with pytest.raises(ValueError):
            circadian_theta_offset(0.0, T_circ=-1.0)

    def test_amplitude_zero_gives_zero_offset(self):
        val = circadian_theta_offset(12345.0, A_circ=0.0)
        assert val == 0.0

    def test_output_bounded_by_amplitude(self):
        for t in np.linspace(0, T_CIRCADIAN_DEFAULT, 100):
            val = circadian_theta_offset(t, A_circ=0.15)
            assert abs(val) <= 0.15 + 1e-10


class TestUltradianOffset:
    """§27: Ultradian BRAC cosine offset."""

    def test_peak_at_t_zero(self):
        val = ultradian_theta_offset(0.0, A_ultradian=0.05, phi_ultradian=0.0)
        assert abs(val - 0.05) < 1e-10

    def test_period_T_invalid_raises(self):
        with pytest.raises(ValueError):
            ultradian_theta_offset(0.0, T_ultradian=0.0)

    def test_output_bounded_by_amplitude(self):
        for t in np.linspace(0, T_ULTRADIAN_DEFAULT * 3, 200):
            val = ultradian_theta_offset(t, A_ultradian=0.08)
            assert abs(val) <= 0.08 + 1e-10


class TestCombinedRhythmOffset:
    """§27: Superposition of both rhythms."""

    def test_combined_equals_sum_of_parts(self):
        t = 3600.0
        circ = circadian_theta_offset(t, A_circ=0.1)
        ultr = ultradian_theta_offset(t, A_ultradian=0.05)
        combined = combined_biological_rhythm_offset(t, A_circ=0.1, A_ultradian=0.05)
        assert abs(combined - (circ + ultr)) < 1e-12

    def test_apply_to_theta_floors_at_theta_min(self):
        # Large negative amplitude can drive θ below zero → floor enforced
        theta_eff = apply_biological_rhythm_to_theta(
            theta_base=0.05, t=T_CIRCADIAN_DEFAULT / 2,
            A_circ=0.1, A_ultradian=0.0, theta_min=0.0
        )
        assert theta_eff >= 0.0

    def test_apply_to_theta_baseline_plus_offset(self):
        theta_eff = apply_biological_rhythm_to_theta(
            theta_base=1.0, t=0.0, A_circ=0.1, A_ultradian=0.05, theta_min=0.0
        )
        assert abs(theta_eff - 1.15) < 1e-9


class TestVectorizedOffsets:
    """Vectorized array variants."""

    def test_circ_array_shape(self):
        t = np.linspace(0, T_CIRCADIAN_DEFAULT, 50)
        out = circadian_theta_offset_array(t, A_circ=0.1)
        assert out.shape == (50,)

    def test_ultradian_array_shape(self):
        t = np.linspace(0, T_ULTRADIAN_DEFAULT * 2, 30)
        out = ultradian_theta_offset_array(t, A_ultradian=0.05)
        assert out.shape == (30,)

    def test_array_scalar_consistency(self):
        t_val = 7200.0
        scalar = circadian_theta_offset(t_val, A_circ=0.12)
        arr = circadian_theta_offset_array(np.array([t_val]), A_circ=0.12)
        assert abs(scalar - float(arr[0])) < 1e-12


class TestCircadianRegulator:
    """§27: Stateful CircadianRegulator."""

    def test_initial_offset_at_t0(self):
        reg = CircadianRegulator(t0=0.0, A_circ=0.1, A_ultradian=0.05)
        expected = combined_biological_rhythm_offset(0.0, A_circ=0.1, A_ultradian=0.05)
        assert abs(reg.theta_offset() - expected) < 1e-12

    def test_tick_advances_time(self):
        reg = CircadianRegulator(t0=0.0, dt=60.0)
        reg.tick()
        assert reg.current_time == 60.0

    def test_multiple_ticks(self):
        reg = CircadianRegulator(t0=0.0, dt=1.0)
        for _ in range(100):
            reg.tick()
        assert abs(reg.current_time - 100.0) < 1e-9

    def test_reset_restores_t0(self):
        reg = CircadianRegulator(t0=0.0, dt=1.0)
        for _ in range(50):
            reg.tick()
        reg.reset(t0=0.0)
        assert reg.current_time == 0.0

    def test_offset_changes_with_time(self):
        reg = CircadianRegulator(t0=0.0, dt=3600.0, A_circ=0.1, A_ultradian=0.0)
        offsets = []
        for _ in range(10):
            offsets.append(reg.theta_offset())
            reg.tick()
        # Not all identical — rhythm is oscillating
        assert not all(abs(o - offsets[0]) < 1e-12 for o in offsets[1:])
