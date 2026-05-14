import os
import json
import numpy as np
from main import run_standard_pipeline, run_multiscale_pipeline, save_results, main
from unittest.mock import patch

from core.config_schema import APGIConfig


def get_full_config():
    return APGIConfig().model_dump()


def test_main_standard_ring_buffer():
    # Test standard pipeline with max_history < n_steps
    config = get_full_config()
    results = run_standard_pipeline(n_steps=20, max_history=10, config=config)
    assert len(results["history"]["S"]) == 10


def test_main_multiscale_ring_buffer():
    # Test multiscale pipeline with max_history < n_steps
    config = get_full_config()
    results = run_multiscale_pipeline(n_steps=20, n_levels=3, max_history=10, config=config)
    assert len(results["history"]["S_multiscale"]) == 10


def test_main_multiscale_progress_log():
    # Test multiscale progress logging (> 1000 steps)
    config = get_full_config()
    # 1100 steps should be fast enough
    run_multiscale_pipeline(n_steps=1100, n_levels=3, config=config)


def test_main_save_results(tmp_path):
    # Test save_results with numpy arrays
    filepath = tmp_path / "results.json"
    results = {
        "arr": np.array([1, 2, 3]),
        "nested": {"val": np.array([0.1])},
        "list": [np.array([1])],
    }
    save_results(results, str(filepath))
    assert os.path.exists(filepath)
    with open(filepath, "r") as f:
        data = json.load(f)
    assert data["arr"] == [1, 2, 3]


def test_main_cli_gamma_ne():
    # Test CLI --gamma-ne override
    with patch("sys.argv", ["main.py", "--gamma-ne", "0.05", "--steps", "10"]):
        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.return_value = {"history": {"S": [0] * 10}, "config": {}, "ignition_count": 0}
            main()
            args, kwargs = mock_run.call_args
            assert kwargs["config"]["gamma_ne"] == 0.05


def test_main_cli_ne_on_threshold_gamma_ne():
    # Test CLI --ne-on-threshold with explicit --gamma-ne
    with patch(
        "sys.argv", ["main.py", "--ne-on-threshold", "--gamma-ne", "0.005", "--steps", "10"]
    ):
        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.return_value = {"history": {"S": [0] * 10}, "config": {}, "ignition_count": 0}
            main()
            args, kwargs = mock_run.call_args
            assert kwargs["config"]["ne_on_threshold"] is True
            assert kwargs["config"]["gamma_ne"] == 0.005


def test_main_cli_output(tmp_path):
    # Test CLI --output
    output_file = tmp_path / "out.json"
    with patch("sys.argv", ["main.py", "--output", str(output_file), "--steps", "10"]):
        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.return_value = {
                "history": {"S": [0] * 10},
                "config": {},
                "ignition_count": 0,
                "n_steps": 10,
            }
            main()
            assert os.path.exists(output_file)


def test_main_cli_multiscale():
    # Test CLI --multiscale
    with patch("sys.argv", ["main.py", "--multiscale", "--levels", "3", "--steps", "10"]):
        with patch("main.run_multiscale_pipeline") as mock_run:
            mock_run.return_value = {
                "history": {"S_multiscale": [0] * 10},
                "config": {},
                "ignition_count": 0,
                "n_steps": 10,
            }
            main()
            mock_run.assert_called_once()


def test_main_cli_interrupted():
    # Test main() with KeyboardInterrupt
    with patch("sys.argv", ["main.py", "--steps", "10"]):
        with patch("main.run_standard_pipeline", side_effect=KeyboardInterrupt):
            assert main() == 130


def test_main_cli_failed():
    # Test main() with general exception
    with patch("sys.argv", ["main.py", "--steps", "10"]):
        with patch("main.run_standard_pipeline", side_effect=Exception("failed")):
            assert main() == 1


def test_main_cli_demo():
    # Test CLI --demo
    with patch("sys.argv", ["main.py", "--demo", "--steps", "10"]):
        # Demo calls run_standard_pipeline
        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.return_value = {
                "history": {"S": [0] * 10},
                "config": {},
                "ignition_count": 0,
                "n_steps": 10,
            }
            main()
            mock_run.assert_called_once()


def test_main_hurst_success_log():
    # Test successful Hurst estimation in run_standard_pipeline
    config = get_full_config()
    # Need at least 256 points
    run_standard_pipeline(n_steps=300, config=config)


def test_main_cli_seed_gamma():
    # Test CLI --seed and --gamma-ne
    with patch("sys.argv", ["main.py", "--seed", "42", "--gamma-ne", "0.01", "--steps", "10"]):
        with patch("main.run_standard_pipeline") as mock_run:
            mock_run.return_value = {
                "history": {"S": [0] * 10},
                "config": {},
                "ignition_count": 0,
                "n_steps": 10,
            }
            main()
            mock_run.assert_called_once()
            # Check if seed was set
            # We can't easily check if random.seed was called, but we covered the line.
