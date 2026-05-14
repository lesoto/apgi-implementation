import pytest
import numpy as np
from active_inference.policy import (
    ActiveInferenceAgent,
    compute_expected_free_energy,
    default_action_params,
    PolicyResult,
    ActionFeedback,
)


def test_default_params():
    params = default_action_params()
    assert params.shape == (3, 3)


def test_expected_free_energy():
    # F = w_epi*sigma2_after - w_prag*pragmatic_gain + w_met*metabolic_cost
    # sigma2_after = max(0.5+0.5 - 0.2, 1e-6) = 0.8
    # pragmatic_gain = 0.5 * max(1.0-0.5, 0) = 0.25
    # F = 1.0*0.8 - 1.0*0.25 + 0.5*0.1 = 0.8 - 0.25 + 0.05 = 0.6
    res = compute_expected_free_energy(
        sigma2_e=0.5,
        sigma2_i=0.5,
        S=1.0,
        theta=0.5,
        action_sensory_shift=0.5,
        action_epistemic_gain=0.2,
        action_metabolic_cost=0.1,
    )
    assert pytest.approx(res) == 0.6

    # Negative margin -> zero pragmatic gain
    res_neg = compute_expected_free_energy(
        sigma2_e=0.5,
        sigma2_i=0.5,
        S=0.1,
        theta=0.5,
        action_sensory_shift=0.5,
        action_epistemic_gain=0.2,
        action_metabolic_cost=0.1,
    )
    # F = 1.0*0.8 - 0 + 0.05 = 0.85
    assert pytest.approx(res_neg) == 0.85


def test_agent_init():
    agent = ActiveInferenceAgent(n_actions=5)
    assert agent.n_actions == 5
    assert agent.action_params.shape == (5, 3)

    custom_params = np.zeros((2, 3))
    agent_custom = ActiveInferenceAgent(n_actions=2, action_params=custom_params)
    assert agent_custom.n_actions == 2

    with pytest.raises(ValueError, match="must have shape"):
        ActiveInferenceAgent(n_actions=3, action_params=custom_params)


def test_select_policy():
    agent = ActiveInferenceAgent()
    res = agent.select_policy(sigma2_e=0.1, sigma2_i=0.1, S=1.0, theta=0.5)
    assert isinstance(res, PolicyResult)
    assert res.action_idx in [0, 1, 2]
    assert len(res.F_values) == 3
    assert len(res.p_policies) == 3
    assert len(agent.action_history) == 1

    # Test high precision -> near-MAP
    agent_high = ActiveInferenceAgent(policy_precision=100.0)
    res_high = agent_high.select_policy(0.1, 0.1, 1.0, 0.5)
    # argmin of F should have p near 1.0
    best_idx = np.argmin(res_high.F_values)
    assert res_high.p_policies[best_idx] > 0.99

    # Out of labels case
    agent_many = ActiveInferenceAgent(n_actions=10)
    res_many = agent_many.select_policy(0.1, 0.1, 1.0, 0.5)
    # If action_idx >= 3, label should be string of index
    if res_many.action_idx >= 3:
        assert res_many.action_label == str(res_many.action_idx)


def test_apply_action_feedback():
    agent = ActiveInferenceAgent()
    # Action 0: [0.10, 0.80, 0.60] (explore)
    fb = agent.apply_action_feedback(0, z_e=1.0, z_i=-1.0, sigma2_e=1.0, sigma2_i=1.0, M=1.0)
    assert isinstance(fb, ActionFeedback)
    # delta_x_hat_e = 0.1 * 0.1 * 1.0 = 0.01
    assert pytest.approx(fb.delta_x_hat_e) == 0.01
    # delta_x_hat_i = 0.1 * 0.1 * -1.0 = -0.01
    assert pytest.approx(fb.delta_x_hat_i) == -0.01
    # delta_M = -0.1 * 0.60 = -0.06
    assert pytest.approx(fb.delta_M) == -0.06
    # reduction = 0.05 * 0.80 = 0.04
    # delta_sigma = -0.04 * 1.0 = -0.04
    assert pytest.approx(fb.delta_sigma2_e) == -0.04
    assert pytest.approx(fb.delta_sigma2_i) == -0.04


def test_reset():
    agent = ActiveInferenceAgent()
    agent.select_policy(0.1, 0.1, 1.0, 0.5)
    assert len(agent.action_history) == 1
    agent.reset()
    assert len(agent.action_history) == 0
