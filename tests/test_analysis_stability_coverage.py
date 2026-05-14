import pytest
import numpy as np
from analysis.stability import (
    compute_jacobian_discrete,
    compute_eigenvalues,
    check_stability,
    compute_fixed_point,
    analyze_bifurcation,
    validate_system_dynamics,
    StabilityAnalyzer,
)


def test_compute_jacobian_discrete():
    J = compute_jacobian_discrete(0.2, 0.15, 0.2, 0.1)
    assert J.shape == (2, 2)
    assert J[0, 0] == pytest.approx(0.8)
    assert J[0, 1] == 0.0


def test_compute_eigenvalues():
    J = np.array([[0.5, 0], [0.1, 0.5]])
    eigs, vecs = compute_eigenvalues(J)
    assert len(eigs) == 2
    assert np.all(eigs == 0.5)

    # Fallback case
    from unittest.mock import patch

    with patch("numpy.linalg.eig", side_effect=np.linalg.LinAlgError):
        e_fb, v_fb = compute_eigenvalues(J)
        assert len(e_fb) == 2
        assert e_fb[0] == 0.5


def test_check_stability(capsys):
    cfg = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
    res = check_stability(cfg, verbose=True)
    assert res["stable"] is True
    captured = capsys.readouterr()
    assert "Stability Analysis" in captured.out


def test_compute_fixed_point():
    cfg = {"lam": 0.2, "theta_base": 1.0}
    fp = compute_fixed_point(cfg)
    assert fp["S_star"] == pytest.approx(5.0)
    assert fp["theta_star"] == 1.0


def test_analyze_bifurcation():
    cfg = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
    # Test lambda variation (always stable in (0,1))
    res = analyze_bifurcation(cfg, "lam", (0.1, 0.9), n_points=5)
    assert len(res["parameter_values"]) == 5
    assert all(res["stability"])

    # Test unstable region (negative kappa)
    res_uns = analyze_bifurcation(cfg, "kappa", (-0.1, 0.1), n_points=10)
    assert not all(res_uns["stability"])
    assert len(res_uns["bifurcation_points"]) > 0


def test_validate_system_dynamics():
    cfg = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1, "theta_base": 1.0}
    # Insufficient data
    res_short = validate_system_dynamics(cfg, np.zeros(10), np.zeros(10))
    assert res_short["valid"] is False

    # Near fixed point
    S = np.full(200, 5.0) + np.random.randn(200) * 0.01
    th = np.full(200, 1.0) + np.random.randn(200) * 0.01
    res = validate_system_dynamics(cfg, S, th)
    assert res["valid"] is True
    assert "prediction_error" in res


def test_stability_analyzer():
    cfg = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
    ana = StabilityAnalyzer(cfg)
    for _ in range(50):
        ana.step(5.0, 1.0)

    # Analysis with insufficient history
    res1 = ana.analyze()
    assert res1["dynamics_validation"]["valid"] is False

    # With enough history
    for _ in range(100):
        ana.step(5.0, 1.0)
    res2 = ana.analyze(verbose=True)
    assert res2["dynamics_validation"]["valid"] is True
