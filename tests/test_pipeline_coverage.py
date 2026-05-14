import pytest

from core.config_schema import APGIConfig
from core.validation import ValidationError
from pipeline import APGIPipeline, HierarchicalState, PrecisionState


def get_base_config():
    return APGIConfig().model_dump()


def test_pipeline_initialization_presets():
    # Test hierarchical presets
    for mode in ["off", "basic", "advanced", "full"]:
        config = get_base_config()
        config["hierarchical_mode"] = mode
        config["strict_mode"] = False
        p = APGIPipeline(config)
        if mode == "off":
            assert p.config["use_hierarchical"] is False
        else:
            assert p.config["use_hierarchical"] is True

    # Test unknown mode
    config = get_base_config()
    config["hierarchical_mode"] = "invalid"
    with pytest.raises(ValueError, match="Unknown hierarchical_mode"):
        APGIPipeline(config)


def test_pipeline_validation_errors():
    # Test strict mode validation failure
    config = get_base_config()
    config["dt"] = 1.0  # dt too large for tau_s=5
    config["strict_mode"] = True
    with pytest.raises(ValidationError):
        APGIPipeline(config)

    # Test NE double counting
    config = get_base_config()
    config["ne_on_precision"] = True
    config["ne_on_threshold"] = True
    config["strict_mode"] = False
    with pytest.raises(ValidationError, match="double-counts"):
        APGIPipeline(config)


def test_pipeline_step_modes():
    # Test basic step
    config = get_base_config()
    config["use_realistic_cost"] = True
    config["signal_log_nonlinearity"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test sliding window variance
    config["variance_method"] = "sliding_window"
    config["T_win"] = 10
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test somatic precision
    config["use_somatic_precision"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test NE on precision
    config["use_somatic_precision"] = False
    config["ne_on_precision"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test discrete mode
    config = get_base_config()
    config["use_canonical_discrete_mode"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test thermodynamic cost
    config = get_base_config()
    config["use_thermodynamic_cost"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test reservoir as threshold
    config = get_base_config()
    config["use_reservoir"] = True
    config["reservoir_as_threshold"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)


def test_pipeline_hierarchical_step():
    # Test full hierarchical mode
    config = get_base_config()
    config["hierarchical_mode"] = "full"
    config["use_resonance"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test advanced hierarchical mode
    config = get_base_config()
    config["hierarchical_mode"] = "advanced"
    config["use_resonance"] = False
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)


def test_pipeline_other_features():
    config = get_base_config()
    config["use_kuramoto"] = True
    config["use_observable_mapping"] = True
    config["use_stability_analysis"] = True
    config["use_active_inference"] = True
    config["ai_n_actions"] = 3
    config["ai_action_params"] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    config["use_bold_calibration"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)

    # Test continuous threshold ODE
    config = get_base_config()
    config["use_continuous_threshold_ode"] = True
    config["use_ode_refractory_drift"] = True
    p = APGIPipeline(config)
    p.step(x_e=1.5, x_i=0.5)


def test_dataclasses():
    ps = PrecisionState(sigma2_e=1.0, sigma2_i=1.0)
    assert ps.pi_e == 1.0

    hs = HierarchicalState(n_levels=2)
    assert len(hs.pis) == 2
