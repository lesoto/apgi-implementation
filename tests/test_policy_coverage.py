import numpy as np
import pytest

from active_inference.policy import (
    ActionFeedback,
    ActiveInferenceAgent,
    PolicyResult,
    _softmax,
    compute_expected_free_energy,
    default_action_params,
)


def test_compute_expected_free_energy():
    # Test normal case
    f = compute_expected_free_energy(
        sigma2_e=1.0,
        sigma2_i=1.0,
        S=1.0,
        theta=0.5,
        action_sensory_shift=0.5,
        action_epistemic_gain=0.1,
        action_metabolic_cost=0.2,
    )
    assert isinstance(f, float)

    # Test low S case
    f2 = compute_expected_free_energy(
        sigma2_e=1.0,
        sigma2_i=1.0,
        S=0.1,
        theta=0.5,
        action_sensory_shift=0.5,
        action_epistemic_gain=0.1,
        action_metabolic_cost=0.2,
    )
    assert f2 > f  # lower S means less pragmatic gain


def test_softmax():
    x = np.array([1.0, 2.0, 3.0])
    p = _softmax(x, gamma=1.0)
    assert np.allclose(np.sum(p), 1.0)
    assert p[0] > p[1] > p[2]


def test_agent_initialization():
    # Test default
    agent = ActiveInferenceAgent()
    assert agent.n_actions == 3

    # Test custom n_actions
    agent2 = ActiveInferenceAgent(n_actions=5)
    assert agent2.n_actions == 5
    assert agent2.action_params.shape == (5, 3)

    # Test custom params
    params = np.random.rand(2, 3)
    agent3 = ActiveInferenceAgent(n_actions=2, action_params=params)
    assert np.allclose(agent3.action_params, params)

    # Test invalid shape
    with pytest.raises(ValueError, match="action_params must have shape"):
        ActiveInferenceAgent(n_actions=3, action_params=np.random.rand(2, 3))


def test_agent_policy_selection():
    agent = ActiveInferenceAgent()
    res = agent.select_policy(sigma2_e=1.0, sigma2_i=1.0, S=1.0, theta=0.5)
    assert isinstance(res, PolicyResult)
    assert 0 <= res.action_idx < 3
    assert len(res.F_values) == 3

    # Test with more actions to cover label fallback
    agent5 = ActiveInferenceAgent(n_actions=5)
    res5 = agent5.select_policy(sigma2_e=1.0, sigma2_i=1.0, S=1.0, theta=0.5)
    assert isinstance(res5.action_label, str)


def test_agent_feedback():
    agent = ActiveInferenceAgent()
    fb = agent.apply_action_feedback(
        action_idx=0, z_e=0.5, z_i=-0.5, sigma2_e=1.0, sigma2_i=1.0, M=0.0
    )
    assert isinstance(fb, ActionFeedback)
    assert fb.delta_x_hat_e > 0
    assert fb.delta_x_hat_i < 0
    assert fb.delta_M < 0
    assert fb.delta_sigma2_e < 0


def test_agent_reset():
    agent = ActiveInferenceAgent()
    agent.select_policy(1, 1, 1, 1)
    assert len(agent.action_history) == 1
    agent.reset()
    assert len(agent.action_history) == 0


def test_default_action_params():
    params = default_action_params()
    assert params.shape == (3, 3)
