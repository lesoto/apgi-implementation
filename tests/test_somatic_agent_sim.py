"""Tests for active_inference.somatic_agent_sim (Protocol 2: Somatic-AgentSim)."""

import numpy as np

from active_inference.somatic_agent_sim import (
    BETA_SM_DEFAULT,
    GAMMA_A_SM_DEFAULT,
    GAMMA_V_DEFAULT,
    APGISomaticAgent,
    ThreeArmedBanditEnv,
    compute_interoceptive_dominance,
    compute_m_hat_lead_lag,
    compute_somatic_marker_estimate,
    estimate_convergence_trial,
    run_agent_on_volatility_schedule,
    run_protocol_2,
)


def test_canonical_defaults_match_protocol_2():
    """Protocol 2: 2:1 value-to-arousal weighting (gamma_V=0.6, gamma_A_SM=0.3)."""
    assert GAMMA_V_DEFAULT == 0.6
    assert GAMMA_A_SM_DEFAULT == 0.3
    assert 0.1 <= BETA_SM_DEFAULT <= 1.2  # canonical beta_SM range


def test_somatic_marker_formula():
    m_hat = compute_somatic_marker_estimate(V=1.0, A=0.5, gamma_V=0.6, gamma_A_sm=0.3)
    assert m_hat == 0.6 * 1.0 + 0.3 * 0.5


class TestThreeArmedBanditEnv:
    def test_three_arms(self):
        env = ThreeArmedBanditEnv(sigma_env=0.3, rng=np.random.default_rng(0))
        assert len(env.probs) == 3
        assert np.all((env.probs >= env.p_min) & (env.probs <= env.p_max))

    def test_pull_returns_binary_reward(self):
        env = ThreeArmedBanditEnv(sigma_env=0.1, rng=np.random.default_rng(0))
        r = env.pull(0)
        assert r in (0.0, 1.0)

    def test_higher_volatility_more_drift(self):
        rng = np.random.default_rng(0)
        env_lo = ThreeArmedBanditEnv(sigma_env=0.01, rng=rng)
        env_hi = ThreeArmedBanditEnv(sigma_env=0.6, rng=rng)
        p0_lo, p0_hi = env_lo.probs.copy(), env_hi.probs.copy()
        for _ in range(50):
            env_lo.step_env()
            env_hi.step_env()
        drift_lo = np.mean(np.abs(env_lo.probs - p0_lo))
        drift_hi = np.mean(np.abs(env_hi.probs - p0_hi))
        assert drift_hi > drift_lo


class TestAPGISomaticAgent:
    def test_beta_sm_lesion_zeroes_beta(self):
        agent = APGISomaticAgent(agent_type="beta_sm_lesion", rng=np.random.default_rng(0))
        assert agent.beta_sm == 0.0

    def test_full_agent_uses_nonzero_beta(self):
        agent = APGISomaticAgent(agent_type="full", rng=np.random.default_rng(0))
        assert agent.beta_sm == BETA_SM_DEFAULT

    def test_pi_lesion_freezes_baseline(self):
        agent = APGISomaticAgent(agent_type="pi_lesion", rng=np.random.default_rng(0))
        assert agent.freeze_pi_baseline is True
        assert agent._pi_i_baseline(0) == 1.0
        agent.update(arm=0, reward=1.0, m_hat=0.1, pi_i_eff=1.0)
        # Frozen baseline: sigma2 must not update even after an update() call
        assert agent._pi_i_baseline(0) == 1.0

    def test_alpha_lesion_disables_metabolic_weighting(self):
        agent = APGISomaticAgent(agent_type="alpha_lesion", rng=np.random.default_rng(0))
        assert agent.disable_metabolic_weighting is True

    def test_random_agent_selects_uniformly_over_many_trials(self):
        agent = APGISomaticAgent(agent_type="random", rng=np.random.default_rng(0))
        choices = [agent.select_action(t, 300)[0] for t in range(300)]
        counts = np.bincount(choices, minlength=3)
        # Roughly uniform (not a strict statistical test, just a sanity bound)
        assert np.all(counts > 50)

    def test_select_action_returns_valid_arm(self):
        agent = APGISomaticAgent(agent_type="full", rng=np.random.default_rng(0))
        arm, m_hat, pi_i_eff = agent.select_action(0, 100)
        assert 0 <= arm < agent.n_arms
        assert isinstance(m_hat, float)
        assert pi_i_eff > 0

    def test_update_logs_ignition_diagnostics(self):
        agent = APGISomaticAgent(agent_type="full", rng=np.random.default_rng(0))
        arm, m_hat, pi_i_eff = agent.select_action(0, 100)
        agent.update(arm, reward=1.0, m_hat=m_hat, pi_i_eff=pi_i_eff)
        assert len(agent.log.action) == 1
        assert agent.log.ignition[0] in (0, 1)


class TestRunAgentOnVolatilitySchedule:
    def test_produces_500_trials_by_default(self):
        log = run_agent_on_volatility_schedule("full", seed=0)
        assert len(log.action) == 500
        assert len(log.reward) == 500
        assert len(log.optimal) == 500

    def test_custom_schedule_length(self):
        log = run_agent_on_volatility_schedule(
            "full", volatility_schedule=[0.1, 0.6], trials_per_block=20, seed=0
        )
        assert len(log.action) == 40


class TestConvergenceEstimate:
    def test_none_when_never_converges(self):
        optimal = [False] * 100
        assert estimate_convergence_trial(optimal) is None

    def test_detects_convergence(self):
        optimal = [False] * 30 + [True] * 100
        trial = estimate_convergence_trial(optimal, window=20, threshold=0.6)
        assert trial is not None
        assert 30 <= trial <= 60

    def test_none_for_short_series(self):
        assert estimate_convergence_trial([True] * 5, window=20) is None


class TestInteroceptiveDominance:
    def test_all_dominant(self):
        from active_inference.somatic_agent_sim import TrialLog

        log = TrialLog()
        log.pi_i_eff = [5.0, 5.0]
        log.z_i = [1.0, 1.0]
        log.pi_e = [1.0, 1.0]
        log.z_e = [1.0, 1.0]
        assert compute_interoceptive_dominance(log) == 1.0

    def test_none_dominant(self):
        from active_inference.somatic_agent_sim import TrialLog

        log = TrialLog()
        log.pi_i_eff = [1.0, 1.0]
        log.z_i = [1.0, 1.0]
        log.pi_e = [5.0, 5.0]
        log.z_e = [1.0, 1.0]
        assert compute_interoceptive_dominance(log) == 0.0

    def test_empty_log_returns_nan(self):
        from active_inference.somatic_agent_sim import TrialLog

        log = TrialLog()
        assert np.isnan(compute_interoceptive_dominance(log))


class TestMHatLeadLag:
    def test_leading_signal_detected(self):
        from active_inference.somatic_agent_sim import TrialLog

        rng = np.random.default_rng(0)
        n = 100
        m_hat = rng.normal(size=n)
        # ignition series lags m_hat by 2 steps
        b = np.zeros(n)
        b[2:] = (m_hat[:-2] > 0).astype(float)
        log = TrialLog()
        log.m_hat = m_hat.tolist()
        log.ignition = b.astype(int).tolist()
        result = compute_m_hat_lead_lag(log, max_lag=5)
        assert result["peak_lag"] is not None
        assert result["leads"] is True

    def test_too_short_returns_none(self):
        from active_inference.somatic_agent_sim import TrialLog

        log = TrialLog()
        log.m_hat = [0.1, 0.2]
        log.ignition = [0, 1]
        result = compute_m_hat_lead_lag(log, max_lag=5)
        assert result["peak_lag"] is None


class TestRunProtocol2:
    def test_all_five_agent_types_present(self):
        results = run_protocol_2(seed=1, trials_per_block=20)
        for agent_type in ("full", "beta_sm_lesion", "pi_lesion", "alpha_lesion", "random"):
            assert agent_type in results
            assert "total_reward" in results[agent_type]
            assert "interoceptive_dominance" in results[agent_type]
            assert "m_hat_lead_lag" in results[agent_type]

    def test_reward_ratio_computed(self):
        results = run_protocol_2(seed=1, trials_per_block=20)
        assert "reward_ratio_full_vs_beta_sm_lesion" in results
        assert results["reward_ratio_full_vs_beta_sm_lesion"] > 0

    def test_lesions_produce_different_dynamics_than_full(self):
        """Structural check: lesioning a pathway must measurably change
        agent behavior (not necessarily 'full wins' — that is itself the
        empirical question Protocol 2 poses, not a foregone conclusion)."""
        results = run_protocol_2(seed=7, trials_per_block=40)
        full_dom = results["full"]["interoceptive_dominance"]
        lesion_dom = results["beta_sm_lesion"]["interoceptive_dominance"]
        assert full_dom != lesion_dom
