"""Validation module for APGI observable mapping and testing."""

from validation.falsification_registry import (
    FALSIFICATION_TABLE,
    FalsificationClaim,
    check_criticality_avalanche,
    check_discrete_ignition,
    check_long_range_correlations,
    check_oscillatory_gate,
    check_phase_transition_signatures,
    check_thermodynamic_bridge,
    check_timescale_ordering,
    check_valence_asymmetry,
    summarize_registry,
)
from validation.observable_mapping import (
    BehavioralObservableExtractor,
    KeyTestablePredictionValidator,
    NeuralObservableExtractor,
    ParameterIdentifiabilityAnalyzer,
)

__all__ = [
    "NeuralObservableExtractor",
    "BehavioralObservableExtractor",
    "KeyTestablePredictionValidator",
    "ParameterIdentifiabilityAnalyzer",
    "FALSIFICATION_TABLE",
    "FalsificationClaim",
    "check_criticality_avalanche",
    "check_discrete_ignition",
    "check_long_range_correlations",
    "check_valence_asymmetry",
    "check_thermodynamic_bridge",
    "check_phase_transition_signatures",
    "check_oscillatory_gate",
    "check_timescale_ordering",
    "summarize_registry",
]
