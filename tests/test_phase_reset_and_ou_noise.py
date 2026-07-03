"""Tests for phase-reset convention fix and OU-noise-as-default fix
(oscillation.kuramoto, oscillation.threshold_modulation, hierarchy.resonance)."""

import numpy as np

from hierarchy.resonance import NestedResonanceSystem, build_resonance_system
from oscillation.kuramoto import KuramotoOscillators
from oscillation.threshold_modulation import HierarchicalPhaseModulator


class TestKuramotoCanonicalReset:
    def test_default_resets_to_zero_phase(self):
        osc = KuramotoOscillators(n_levels=3, frequencies=np.array([0.1, 0.2, 0.3]))
        osc.phases = np.array([2.5, 1.0, 4.0])
        osc.reset_phase_on_ignition(level=1)
        assert osc.phases[1] == 0.0

    def test_default_resets_to_custom_theta_coupling(self):
        osc = KuramotoOscillators(n_levels=3, frequencies=np.array([0.1, 0.2, 0.3]))
        osc.phases = np.array([2.5, 1.0, 4.0])
        osc.reset_phase_on_ignition(level=0, theta_coupling=1.5)
        assert abs(osc.phases[0] - 1.5) < 1e-9

    def test_explicit_reset_amount_preserves_legacy_additive_behavior(self):
        osc = KuramotoOscillators(n_levels=3, frequencies=np.array([0.1, 0.2, 0.3]))
        osc.phases = np.array([2.5, 1.0, 4.0])
        osc.reset_phase_on_ignition(level=0, reset_amount=np.pi)
        expected = (2.5 + np.pi) % (2 * np.pi)
        assert abs(osc.phases[0] - expected) < 1e-9

    def test_broadcast_uses_correction_delta(self):
        osc = KuramotoOscillators(n_levels=3, frequencies=np.array([0.1, 0.2, 0.3]))
        osc.phases = np.array([2.0, 1.0, 4.0])
        osc.reset_phase_on_ignition(level=0, broadcast=True, broadcast_decay=0.5)
        assert osc.phases[0] == 0.0
        # Neighbor got a decayed fraction of the same correction delta
        assert osc.phases[1] != 1.0


class TestThresholdModulatorCanonicalReset:
    def test_default_resets_to_zero(self):
        mod = HierarchicalPhaseModulator(n_levels=3)
        mod.phases = np.array([2.5, 1.0, 4.0])
        mod.reset_phase_on_ignition(level=0)
        assert mod.phases[0] == 0.0

    def test_explicit_amount_preserves_legacy_behavior(self):
        mod = HierarchicalPhaseModulator(n_levels=3)
        mod.phases = np.array([2.5, 1.0, 4.0])
        mod.reset_phase_on_ignition(level=0, reset_amount=np.pi / 2)
        expected = (2.5 + np.pi / 2) % (2 * np.pi)
        assert abs(mod.phases[0] - expected) < 1e-9


class TestResonanceSystemPhaseResetWiring:
    def test_apply_level_ignition_resets_phase_by_default(self):
        sys = build_resonance_system(
            n_levels=3, taus=np.array([1.0, 10.0, 100.0]), theta_base=1.0, dt=1.0
        )
        sys.phi = np.array([2.5, 1.0, 4.0])
        sys.apply_level_ignition(level=0)
        assert sys.phi[0] == 0.0

    def test_reset_phase_can_be_disabled(self):
        sys = build_resonance_system(
            n_levels=3, taus=np.array([1.0, 10.0, 100.0]), theta_base=1.0, dt=1.0
        )
        sys.phi = np.array([2.5, 1.0, 4.0])
        sys.apply_level_ignition(level=0, reset_phase=False)
        assert sys.phi[0] == 2.5

    def test_broadcast_reaches_neighboring_levels(self):
        sys = build_resonance_system(
            n_levels=3,
            taus=np.array([1.0, 10.0, 100.0]),
            theta_base=1.0,
            dt=1.0,
            kappa_up=0.3,
        )
        sys.phi = np.array([2.0, 1.0, 4.0])
        original_phi1 = sys.phi[1]
        sys.apply_level_ignition(level=0)
        assert sys.phi[0] == 0.0
        assert sys.phi[1] != original_phi1


class TestOUPhaseNoiseDefault:
    def test_ou_noise_created_when_enabled_by_default(self):
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([0.1, 0.05]),
            lambda_rates=np.array([0.1, 0.01]),
            phi_noise_std=0.1,
        )
        assert sys._ou_noise is not None
        assert len(sys._ou_noise) == 2

    def test_white_noise_when_ou_disabled(self):
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([0.1, 0.05]),
            lambda_rates=np.array([0.1, 0.01]),
            phi_noise_std=0.1,
            use_ou_phase_noise=False,
        )
        assert sys._ou_noise is None

    def test_no_noise_object_when_std_zero(self):
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([0.1, 0.05]),
            lambda_rates=np.array([0.1, 0.01]),
            phi_noise_std=0.0,
        )
        assert sys._ou_noise is None

    def test_step_runs_with_ou_noise(self):
        sys = NestedResonanceSystem(
            n_levels=2,
            theta_0=np.array([1.0, 1.0]),
            omega=np.array([0.1, 0.05]),
            lambda_rates=np.array([0.1, 0.01]),
            phi_noise_std=0.1,
        )
        sys.step(S_inst_levels=np.array([0.5, 0.3]), pi_levels=np.array([1.0, 1.0]), dt=1.0)
        assert np.all(np.isfinite(sys.phi))
