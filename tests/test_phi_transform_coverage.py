import numpy as np
import pytest

from core.phi_transform import phi_transform, phi_transform_array, validate_phi_params


def test_validate_phi_params():
    # Valid
    validate_phi_params(1.0, 1.0, 2.0, 2.0)

    # Invalid alpha_pos
    with pytest.raises(ValueError, match="alpha_plus"):
        validate_phi_params(0.1, 1.0, 2.0, 2.0)
    with pytest.raises(ValueError, match="alpha_plus"):
        validate_phi_params(3.0, 1.0, 2.0, 2.0)

    # Invalid alpha_neg
    with pytest.raises(ValueError, match="alpha_minus"):
        validate_phi_params(1.0, 0.1, 2.0, 2.0)
    with pytest.raises(ValueError, match="alpha_minus"):
        validate_phi_params(1.0, 3.0, 2.0, 2.0)

    # Invalid gamma_pos
    with pytest.raises(ValueError, match="gamma_plus"):
        validate_phi_params(1.0, 1.0, 0.5, 2.0)
    with pytest.raises(ValueError, match="gamma_plus"):
        validate_phi_params(1.0, 1.0, 6.0, 2.0)

    # Invalid gamma_neg
    with pytest.raises(ValueError, match="gamma_minus"):
        validate_phi_params(1.0, 1.0, 2.0, 0.5)
    with pytest.raises(ValueError, match="gamma_minus"):
        validate_phi_params(1.0, 1.0, 2.0, 6.0)


def test_phi_transform():
    # Positive
    val_pos = phi_transform(1.0, alpha_pos=1.0, gamma_pos=2.0)
    assert val_pos == np.tanh(2.0)

    # Negative
    val_neg = phi_transform(-1.0, alpha_neg=1.0, gamma_neg=2.0)
    assert val_neg == np.tanh(-2.0)


def test_phi_transform_array():
    arr = np.array([-1.0, 0.0, 1.0])
    res = phi_transform_array(arr, alpha_pos=1.5, alpha_neg=0.5, gamma_pos=1.0, gamma_neg=3.0)

    assert res[0] == 0.5 * np.tanh(-3.0)
    assert res[1] == 1.5 * np.tanh(0.0)  # 0.0
    assert res[2] == 1.5 * np.tanh(1.0)
