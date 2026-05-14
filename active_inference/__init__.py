"""Active Inference Action Loop — §19 of APGI Full Specs."""

from active_inference.policy import (
    ActionFeedback,
    ActiveInferenceAgent,
    PolicyResult,
    compute_expected_free_energy,
    default_action_params,
)

__all__ = [
    "ActiveInferenceAgent",
    "ActionFeedback",
    "PolicyResult",
    "compute_expected_free_energy",
    "default_action_params",
]
