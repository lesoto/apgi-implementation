"""Active Inference Action Loop — §19 of APGI Full Specs."""

from active_inference.policy import (
    ActionFeedback,
    ActiveInferenceAgent,
    PolicyResult,
    compute_expected_free_energy,
    default_action_params,
)
from active_inference.somatic_agent_sim import (
    APGISomaticAgent,
    ThreeArmedBanditEnv,
    compute_somatic_marker_estimate,
    run_protocol_2,
)

__all__ = [
    "ActiveInferenceAgent",
    "ActionFeedback",
    "PolicyResult",
    "compute_expected_free_energy",
    "default_action_params",
    "APGISomaticAgent",
    "ThreeArmedBanditEnv",
    "compute_somatic_marker_estimate",
    "run_protocol_2",
]
