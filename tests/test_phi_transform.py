"""Unit tests for core/phi_transform.py — §6 Signed Nonlinear Transform.

Covers:
- Scalar phi_transform: positive/negative branch correctness
- Vectorized phi_transform_array
- Parameter bounds validation
- Symmetric case reduces to standard tanh (not abs, per spec §6 Compatibility Note)
- Asymmetric gain ratio α⁺/α⁻ ≠ 1 distinguishes approach from avoidance
- Saturation: |φ(ε)| < α for all finite ε
- Sign preservation: φ(ε) ≥ 0 iff ε ≥ 0
"""

from __future__ import annotations

import numpy as np
import pytest

from core.phi_transform import (
    ALPHA_MAX,
    ALPHA_MIN,
    GAMMA_MAX,
    GAMMA_MIN,
    phi_transform,
    phi_transform_array,
    validate_phi_params,
)


class TestPhiTransformScalar:
    """Scalar phi_transform correctness."""

    def test_zero_error_returns_zero(self):
        assert phi_transform(0.0) == 0.0

    def test_positive_error_uses_pos_branch(self):
        result = phi_transform(1.0, alpha_pos=1.5, alpha_neg=0.5, gamma_pos=2.0, gamma_neg=1.0)
        expected = 1.5 * np.tanh(2.0 * 1.0)
        assert np.isclose(result, expected)

    def test_negative_error_uses_neg_branch(self):
        result = phi_transform(-1.0, alpha_pos=1.5, alpha_neg=0.5, gamma_pos=2.0, gamma_neg=1.0)
        expected = 0.5 * np.tanh(1.0 * -1.0)
        assert np.isclose(result, expected)

    def test_sign_preservation_positive(self):
        assert phi_transform(0.5) > 0.0

    def test_sign_preservation_negative(self):
        assert phi_transform(-0.5) < 0.0

    def test_saturation_positive(self):
        """φ(ε) must be bounded by α⁺ (tanh asymptote); large ε approaches but never exceeds."""
        result = phi_transform(10.0, alpha_pos=1.5, alpha_neg=1.0, gamma_pos=3.0, gamma_neg=2.0)
        assert result <= 1.5

    def test_saturation_negative(self):
        result = phi_transform(-10.0, alpha_pos=1.0, alpha_neg=0.8, gamma_pos=2.0, gamma_neg=2.0)
        assert result >= -0.8

    def test_asymmetric_gain_ratio(self):
        """α⁺/α⁻ ≠ 1 produces different magnitudes for ±ε."""
        pos = phi_transform(1.0, alpha_pos=2.0, alpha_neg=0.5, gamma_pos=2.0, gamma_neg=2.0)
        neg = phi_transform(-1.0, alpha_pos=2.0, alpha_neg=0.5, gamma_pos=2.0, gamma_neg=2.0)
        assert abs(pos) > abs(neg)

    def test_symmetric_case_is_odd_function(self):
        """With α⁺=α⁻ and γ⁺=γ⁻ the transform is antisymmetric: φ(-ε) = -φ(ε)."""
        val = phi_transform(1.5, alpha_pos=1.0, alpha_neg=1.0, gamma_pos=2.0, gamma_neg=2.0)
        neg_val = phi_transform(-1.5, alpha_pos=1.0, alpha_neg=1.0, gamma_pos=2.0, gamma_neg=2.0)
        assert np.isclose(val, -neg_val)

    def test_default_params_give_finite_output(self):
        for x in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            assert np.isfinite(phi_transform(x))


class TestPhiTransformArray:
    """Vectorized phi_transform_array."""

    def test_matches_scalar_per_element(self):
        arr = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])
        result = phi_transform_array(
            arr, alpha_pos=1.2, alpha_neg=0.8, gamma_pos=3.0, gamma_neg=1.5
        )
        for i, x in enumerate(arr):
            expected = phi_transform(x, alpha_pos=1.2, alpha_neg=0.8, gamma_pos=3.0, gamma_neg=1.5)
            assert np.isclose(result[i], expected), f"Mismatch at index {i} for x={x}"

    def test_sign_matches_input_sign(self):
        arr = np.linspace(-3.0, 3.0, 61)
        result = phi_transform_array(arr)
        np.testing.assert_array_equal(np.sign(result), np.sign(arr))

    def test_output_shape_preserved(self):
        arr = np.zeros((4, 3))
        result = phi_transform_array(arr)
        assert result.shape == (4, 3)

    def test_all_zero_input(self):
        arr = np.zeros(5)
        result = phi_transform_array(arr)
        np.testing.assert_array_equal(result, 0.0)


class TestValidatePhiParams:
    """Parameter bounds enforcement (§6)."""

    def test_valid_params_no_exception(self):
        validate_phi_params(1.0, 1.0, 2.0, 2.0)

    def test_alpha_pos_below_min_raises(self):
        with pytest.raises(ValueError, match="alpha_plus"):
            validate_phi_params(ALPHA_MIN - 0.1, 1.0, 2.0, 2.0)

    def test_alpha_pos_above_max_raises(self):
        with pytest.raises(ValueError, match="alpha_plus"):
            validate_phi_params(ALPHA_MAX + 0.1, 1.0, 2.0, 2.0)

    def test_alpha_neg_below_min_raises(self):
        with pytest.raises(ValueError, match="alpha_minus"):
            validate_phi_params(1.0, ALPHA_MIN - 0.1, 2.0, 2.0)

    def test_gamma_pos_below_min_raises(self):
        with pytest.raises(ValueError, match="gamma_plus"):
            validate_phi_params(1.0, 1.0, GAMMA_MIN - 0.1, 2.0)

    def test_gamma_neg_above_max_raises(self):
        with pytest.raises(ValueError, match="gamma_minus"):
            validate_phi_params(1.0, 1.0, 2.0, GAMMA_MAX + 0.1)

    def test_boundary_values_accepted(self):
        validate_phi_params(ALPHA_MIN, ALPHA_MAX, GAMMA_MIN, GAMMA_MAX)


class TestPhiTransformBiologicalProperties:
    """§6 Preserved Biological Properties."""

    def test_approach_exceeds_avoidance_with_asymmetric_gain(self):
        """Positive gain asymmetry: approach signal > avoidance magnitude at same |ε|."""
        epsilon = 1.0
        approach = phi_transform(
            epsilon, alpha_pos=1.8, alpha_neg=0.6, gamma_pos=2.0, gamma_neg=2.0
        )
        avoidance = phi_transform(
            -epsilon, alpha_pos=1.8, alpha_neg=0.6, gamma_pos=2.0, gamma_neg=2.0
        )
        assert approach > abs(avoidance)

    def test_avoidance_can_suppress_signal(self):
        """Negative φ(ε) for aversive errors — suppression possible (§6 property 3)."""
        result = phi_transform(-0.5, alpha_neg=1.0, alpha_pos=1.0, gamma_neg=2.0, gamma_pos=2.0)
        assert result < 0.0

    def test_different_gamma_changes_saturation_point(self):
        """Higher γ saturates faster: φ(1, γ=5) ≈ φ(large, γ=1)."""
        steep = phi_transform(0.5, alpha_pos=1.0, alpha_neg=1.0, gamma_pos=5.0, gamma_neg=1.0)
        shallow = phi_transform(0.5, alpha_pos=1.0, alpha_neg=1.0, gamma_pos=1.0, gamma_neg=1.0)
        assert steep > shallow
