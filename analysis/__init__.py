"""Analysis module for APGI stability and dynamics."""

from analysis.stability import (
    StabilityAnalyzer,
    analyze_bifurcation,
    check_stability,
    compute_eigenvalues,
    compute_fixed_point,
    compute_jacobian_discrete,
    validate_system_dynamics,
)

__all__ = [
    "compute_jacobian_discrete",
    "compute_eigenvalues",
    "check_stability",
    "compute_fixed_point",
    "analyze_bifurcation",
    "validate_system_dynamics",
    "StabilityAnalyzer",
]
