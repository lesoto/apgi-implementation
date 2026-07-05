"""Tests to achieve 100% coverage for pipeline.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestPipeline100Coverage:
    """Tests to cover remaining lines in pipeline.py."""

    def test_pipeline_with_circadian_regulator(self):
        """Test pipeline with circadian regulator to cover lines 952-953."""
        try:
            from core.circadian import CircadianRegulator
            from pipeline import Pipeline
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")

        # Create minimal config
        config = {
            "theta_base": 0.5,
            "lambda_s": 0.1,
            "lambda_c": 0.01,
            "V_max": 1.0,
            "C_max": 1.0,
            "omega_ne": 0.1,
            "omega_5ht": 0.05,
            "sigma_noise": 0.01,
        }

        # Create pipeline
        pipeline = Pipeline(config)

        # Add circadian regulator
        pipeline.circadian_regulator = CircadianRegulator(
            tau_circadian=24.0,
            amplitude=0.1,
            phase_shift=0.0,
        )

        # Run step - this should call tick()
        result = pipeline.step(u=0.5)

        # Verify step completed
        assert result is not None

    def test_pipeline_with_reservoir_additive_mode(self):
        """Test pipeline with reservoir in additive mode to cover line 1119."""
        try:
            from pipeline import Pipeline
        except ImportError as e:
            pytest.skip(f"Could not import Pipeline: {e}")

        # Create config with reservoir enabled but not replacing signal
        config = {
            "theta_base": 0.5,
            "lambda_s": 0.1,
            "lambda_c": 0.01,
            "V_max": 1.0,
            "C_max": 1.0,
            "omega_ne": 0.1,
            "omega_5ht": 0.05,
            "sigma_noise": 0.01,
            "use_reservoir": True,
            "reservoir_replaces_signal": False,  # This should trigger the else block
            "reservoir_weight": 0.2,  # Explicit weight for line 1119
            "reservoir_amplification": 0.0,
            "reservoir_readout_method": "linear",
        }

        # Create pipeline
        pipeline = Pipeline(config)

        # Mock the reservoir to avoid complex dependencies
        mock_reservoir = MagicMock()
        mock_reservoir.step.return_value = None
        mock_reservoir.readout.return_value = 0.5
        pipeline.reservoir = mock_reservoir

        # Run step - this should use config.get("reservoir_weight", 0.1)
        result = pipeline.step(u=0.5)

        # Verify step completed
        assert result is not None

        # Verify reservoir was called
        mock_reservoir.step.assert_called_once()
        mock_reservoir.readout.assert_called_once()

    def test_pipeline_with_reservoir_default_weight(self):
        """Test pipeline with reservoir using default weight."""
        try:
            from pipeline import Pipeline
        except ImportError as e:
            pytest.skip(f"Could not import Pipeline: {e}")

        # Create config with reservoir but no explicit weight (should use default 0.1)
        config = {
            "theta_base": 0.5,
            "lambda_s": 0.1,
            "lambda_c": 0.01,
            "V_max": 1.0,
            "C_max": 1.0,
            "omega_ne": 0.1,
            "omega_5ht": 0.05,
            "sigma_noise": 0.01,
            "use_reservoir": True,
            "reservoir_replaces_signal": False,
            # No reservoir_weight - should use default 0.1
        }

        # Create pipeline
        pipeline = Pipeline(config)

        # Mock the reservoir
        mock_reservoir = MagicMock()
        mock_reservoir.step.return_value = None
        mock_reservoir.readout.return_value = 0.5
        pipeline.reservoir = mock_reservoir

        # Run step
        result = pipeline.step(u=0.5)

        # Verify step completed
        assert result is not None

    def test_pipeline_with_hierarchical_mode(self):
        """Test pipeline with hierarchical mode enabled."""
        try:
            from pipeline import Pipeline
        except ImportError as e:
            pytest.skip(f"Could not import Pipeline: {e}")

        # Create config with hierarchical mode
        config = {
            "theta_base": 0.5,
            "lambda_s": 0.1,
            "lambda_c": 0.01,
            "V_max": 1.0,
            "C_max": 1.0,
            "omega_ne": 0.1,
            "omega_5ht": 0.05,
            "sigma_noise": 0.01,
            "use_hierarchical": True,
        }

        # Create pipeline
        pipeline = Pipeline(config)

        # Mock hierarchical system
        mock_hierarchical = MagicMock()
        mock_hierarchical.step.return_value = (np.array([0.5]), np.array([0.6]))
        pipeline.hierarchical = mock_hierarchical

        # Run step
        result = pipeline.step(u=0.5)

        # Verify step completed
        assert result is not None

    def test_pipeline_all_edge_cases(self):
        """Test pipeline with multiple edge cases."""
        try:
            from core.circadian import CircadianRegulator
            from pipeline import Pipeline
        except ImportError as e:
            pytest.skip(f"Could not import required modules: {e}")

        # Create comprehensive config
        config = {
            "theta_base": 0.5,
            "lambda_s": 0.1,
            "lambda_c": 0.01,
            "V_max": 1.0,
            "C_max": 1.0,
            "omega_ne": 0.1,
            "omega_5ht": 0.05,
            "sigma_noise": 0.01,
            "use_reservoir": True,
            "reservoir_replaces_signal": False,
            "reservoir_weight": 0.15,
            "use_hierarchical": True,
            "use_active_inference": True,
            "ai_on_ignition_only": False,
        }

        # Create pipeline
        pipeline = Pipeline(config)

        # Add circadian regulator
        pipeline.circadian_regulator = CircadianRegulator(
            tau_circadian=24.0,
            amplitude=0.1,
            phase_shift=0.0,
        )

        # Mock dependencies
        mock_reservoir = MagicMock()
        mock_reservoir.step.return_value = None
        mock_reservoir.readout.return_value = 0.3
        pipeline.reservoir = mock_reservoir

        mock_hierarchical = MagicMock()
        mock_hierarchical.step.return_value = (np.array([0.4]), np.array([0.5]))
        pipeline.hierarchical = mock_hierarchical

        mock_ai_policy = MagicMock()
        mock_ai_policy.select_action.return_value = 0
        pipeline.ai_policy = mock_ai_policy

        # Run multiple steps to cover various code paths
        for _ in range(3):
            result = pipeline.step(u=np.random.randn())
            assert result is not None


class TestLiquidNetwork100Coverage:
    """Tests to cover remaining lines in liquid_network.py."""

    def test_liquid_network_suprathreshold_error(self):
        """Test liquid network line 141 - error message."""
        from reservoir.liquid_network import LiquidNetwork

        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)

        # Setup conditions to trigger the error
        network.x = np.ones(100) * 5.0  # High state
        S_target = 10.0  # Much higher than theta
        theta = 1.0  # Low threshold
        A_amp = 10.0  # High amplification

        # This should trigger the error with saturating_amplification=False
        with pytest.raises(ValueError, match="Suprathreshold amplification unstable"):
            network.step(
                u=0.5,
                tau=1.0,
                S_target=S_target,
                theta=theta,
                A_amp=A_amp,
                saturating_amplification=False,
                alpha_res=1.0,  # Small alpha_res to trigger error
            )

    def test_liquid_network_with_saturating_amplification(self):
        """Test liquid network with saturating amplification (should not error)."""
        from reservoir.liquid_network import LiquidNetwork

        network = LiquidNetwork(n_units=100, spectral_radius=0.9, res_sparsity=0.15)

        # Same conditions but with saturating_amplification=True
        network.x = np.ones(100) * 5.0
        S_target = 10.0
        theta = 1.0
        A_amp = 10.0

        # Should not raise error with saturating_amplification=True
        result = network.step(
            u=0.5,
            tau=1.0,
            S_target=S_target,
            theta=theta,
            A_amp=A_amp,
            saturating_amplification=True,  # Default is True
            alpha_res=1.0,
        )

        assert result is not None


class TestLiquidStateMachine100Coverage:
    """Tests to cover remaining lines in liquid_state_machine.py."""

    def test_lsm_with_seed(self):
        """Test liquid state machine line 157 - seed initialization."""
        from reservoir.liquid_state_machine import LiquidStateMachine

        # Test with seed
        lsm = LiquidStateMachine(
            N=100,
            spectral_radius=0.9,
            res_sparsity=0.15,
            seed=42,
        )
        assert lsm.N == 100

        # Test without seed
        lsm2 = LiquidStateMachine(
            N=50,
            spectral_radius=0.8,
            res_sparsity=0.2,
            # No seed
        )
        assert lsm2.N == 50

    def test_lsm_suprathreshold_error(self):
        """Test liquid state machine line 320 - error message."""
        from reservoir.liquid_state_machine import LiquidStateMachine

        lsm = LiquidStateMachine(
            N=100,
            spectral_radius=0.9,
            res_sparsity=0.15,
        )

        # Setup conditions to trigger the error
        lsm.x = np.ones(100) * 5.0  # High state
        S_target = 10.0  # Much higher than theta
        theta = 1.0  # Low threshold
        A_amp = 10.0  # High amplification

        # This should trigger the error with saturating_amplification=False
        with pytest.raises(ValueError, match="Suprathreshold amplification unstable"):
            lsm.step(
                u=0.5,
                tau=1.0,
                S_target=S_target,
                theta=theta,
                A_amp=A_amp,
                saturating_amplification=False,
                alpha_res=1.0,  # Small alpha_res to trigger error
            )

    def test_lsm_res_sparsity_validation(self):
        """Test liquid state machine res_sparsity validation."""
        from reservoir.liquid_state_machine import LiquidStateMachine

        # Test invalid res_sparsity
        with pytest.raises(ValueError, match="res_sparsity must be in"):
            LiquidStateMachine(
                N=100,
                spectral_radius=0.9,
                res_sparsity=0.0,  # Invalid
            )

        with pytest.raises(ValueError, match="res_sparsity must be in"):
            LiquidStateMachine(
                N=100,
                spectral_radius=0.9,
                res_sparsity=1.5,  # Invalid
            )
