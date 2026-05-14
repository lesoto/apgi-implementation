import numpy as np

from analysis.stability import (
    StabilityAnalyzer,
    analyze_bifurcation,
    check_stability,
    compute_eigenvalues,
    compute_fixed_point,
    compute_jacobian_discrete,
    validate_system_dynamics,
)


def test_jacobian_and_eigenvalues():
    J = compute_jacobian_discrete(lam=0.2, kappa=0.15, c1=0.2, eta=0.1)
    assert J.shape == (2, 2)
    assert J[0, 1] == 0.0

    eigs, vecs = compute_eigenvalues(J)
    assert len(eigs) == 2
    assert vecs.shape == (2, 2)

    # Test error fallback
    eigs2, vecs2 = compute_eigenvalues(np.array([[np.nan, 0], [0, 1]]))
    assert len(eigs2) == 2


def test_stability_check():
    config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
    res = check_stability(config, verbose=True)
    assert res["stable"] is True

    # Test unstable
    config_unstable = {"lam": -0.1, "kappa": -0.1}
    res_unstable = check_stability(config_unstable)
    assert res_unstable["stable"] is False


def test_fixed_point():
    config = {"lam": 0.2, "theta_base": 1.5}
    fp = compute_fixed_point(config)
    assert fp["theta_star"] == 1.5
    assert fp["S_star"] > 0


def test_bifurcation_analysis():
    config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1}
    res = analyze_bifurcation(config, "lam", (0.0, 1.0))
    assert len(res["parameter_values"]) == 50
    assert "bifurcation_points" in res


def test_dynamics_validation():
    config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1, "theta_base": 1.0}

    # Insufficient data
    res = validate_system_dynamics(config, np.zeros(10), np.zeros(10))
    assert res["valid"] is False

    # Sufficient data
    S = np.random.randn(150)
    theta = np.random.randn(150)
    res2 = validate_system_dynamics(config, S, theta)
    assert res2["valid"] is True


def test_stability_analyzer():
    config = {"lam": 0.2, "kappa": 0.15, "c1": 0.2, "eta": 0.1, "theta_base": 1.0}
    sa = StabilityAnalyzer(config)
    for _ in range(110):
        sa.step(S=1.0, theta=1.2)

    res = sa.analyze(verbose=True)
    assert res["stability"]["stable"] is True
    assert res["dynamics_validation"]["valid"] is True

    # Small history
    sa2 = StabilityAnalyzer(config)
    sa2.step(1.0, 1.2)
    res2 = sa2.analyze()
    assert res2["dynamics_validation"]["valid"] is False
