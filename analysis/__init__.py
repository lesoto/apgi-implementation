"""Analysis module for APGI stability and dynamics."""

from analysis.stability import (
    compute_jacobian_discrete,
    compute_eigenvalues,
    check_stability,
    compute_fixed_point,
    analyze_bifurcation,
    validate_system_dynamics,
    StabilityAnalyzer,
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
