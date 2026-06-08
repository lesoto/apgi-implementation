"""Tests achieving 100% coverage of the examples folder.

Each example is loaded via importlib so coverage can track executed lines.
Matplotlib is forced to the non-interactive Agg backend and plt.show / savefig
are patched to avoid display or file side-effects.

Strategy:
  - Regular _load() tests exercise function-level logic via main() or helper calls.
  - _run_as_main() tests use runpy so the `if __name__ == "__main__":` blocks execute.
  - Targeted patch tests force specific conditional branches that regular runs skip.
"""

import contextlib
import importlib.util
import io
import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib
import numpy as np

matplotlib.use("Agg")

EXAMPLES_DIR = Path(__file__).parent.parent / "examples"


def _load(filename: str) -> types.ModuleType:
    """Load an example file as a fresh module via importlib."""
    path = EXAMPLES_DIR / filename
    spec = importlib.util.spec_from_file_location(f"_ex_{filename}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def _run(func, *args, **kwargs):
    """Call func capturing stdout; return (return_value, captured_output)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        ret = func(*args, **kwargs)
    return ret, buf.getvalue()


def _run_as_main(filename: str) -> str:
    """Execute an example file as __main__ via runpy, capturing stdout.

    This covers the ``if __name__ == '__main__':`` blocks which importlib skips.
    """
    path = str(EXAMPLES_DIR / filename)
    buf = io.StringIO()
    with (
        contextlib.redirect_stdout(buf),
        patch("matplotlib.pyplot.show"),
        patch("matplotlib.pyplot.savefig"),
    ):
        runpy.run_path(path, run_name="__main__")
    return buf.getvalue()


# ===========================================================================
# Example 01 – basic usage
# ===========================================================================


def test_example_01_basic_usage():
    mod = _load("01_basic_usage.py")
    _, out = _run(mod.main)
    assert "completed successfully" in out


def test_example_01_as_main():
    """Cover `if __name__ == '__main__': main()` (line 108)."""
    out = _run_as_main("01_basic_usage.py")
    assert "completed successfully" in out


# ===========================================================================
# Example 02 – advanced features
# ===========================================================================


def test_example_02_advanced_features():
    mod = _load("02_advanced_features.py")
    _, out = _run(mod.main)
    assert "completed successfully" in out


def test_example_02_as_main():
    """Cover `if __name__ == '__main__': main()` (line 168)."""
    out = _run_as_main("02_advanced_features.py")
    assert "completed successfully" in out


def test_example_02_stability_eigenvalues():
    """Cover lines 101, 127-136: stability_eigenvalues branch.

    The pipeline never adds stability_eigenvalues per-step on its own, so we
    monkey-patch APGIPipeline.step to inject the key.
    """
    from pipeline import APGIPipeline

    mod = _load("02_advanced_features.py")
    orig_step = APGIPipeline.step

    def _step_with_eigs(self, *args, **kwargs):
        r = orig_step(self, *args, **kwargs)
        r["stability_eigenvalues"] = np.array([0.5, 0.3, 0.1])
        return r

    APGIPipeline.step = _step_with_eigs
    try:
        _, out = _run(mod.main)
    finally:
        APGIPipeline.step = orig_step
    assert "Stability Analysis Results" in out


# ===========================================================================
# Example 03 – observable mapping
# ===========================================================================


def test_example_03_observable_mapping():
    mod = _load("03_observable_mapping.py")
    _, out = _run(mod.main)
    assert "completed successfully" in out


def test_example_03_as_main():
    """Cover `if __name__ == '__main__': main()` (line 208)."""
    out = _run_as_main("03_observable_mapping.py")
    assert "completed successfully" in out


def test_example_03_no_prediction_margin():
    """Cover line 75: fallback when prediction_margin absent from step result."""
    from pipeline import APGIPipeline

    mod = _load("03_observable_mapping.py")
    orig_step = APGIPipeline.step

    def _step_without_margin(self, *args, **kwargs):
        r = orig_step(self, *args, **kwargs)
        r.pop("prediction_margin", None)
        return r

    APGIPipeline.step = _step_without_margin
    try:
        _, out = _run(mod.main)
    finally:
        APGIPipeline.step = orig_step
    assert "completed successfully" in out


def test_example_03_corr_delta_less_than_corr_signal():
    """Cover line 200: else branch where margin does NOT outperform signal."""
    mod = _load("03_observable_mapping.py")
    call_count = [0]
    orig_corrcoef = np.corrcoef

    def mock_corrcoef(a, b):
        call_count[0] += 1
        # First call: corr_S=0.9, Second call: corr_delta=0.4 → else branch
        if call_count[0] == 1:
            return np.array([[1.0, 0.9], [0.9, 1.0]])
        return np.array([[1.0, 0.4], [0.4, 1.0]])

    mod.np.corrcoef = mock_corrcoef
    try:
        _, out = _run(mod.main)
    finally:
        mod.np.corrcoef = orig_corrcoef
    assert "Prediction not validated" in out


# ===========================================================================
# Example 04 – thermodynamics
# ===========================================================================


def test_example_04_thermodynamics():
    mod = _load("04_thermodynamics.py")
    _, out = _run(mod.main)
    assert "completed successfully" in out


def test_example_04_as_main():
    """Cover `if __name__ == '__main__': main()` (line 216)."""
    out = _run_as_main("04_thermodynamics.py")
    assert "completed successfully" in out


def test_example_04_constraint_satisfied():
    """Cover line 168: 'Thermodynamic constraint satisfied' branch.

    Patch compute_landauer_cost to return near-zero so all C values exceed it.
    """
    mod = _load("04_thermodynamics.py")
    mod.compute_landauer_cost = (
        lambda *args, **kwargs: 0.0
    )  # zero landauer → C >= 0 always satisfied

    try:
        _, out = _run(mod.main)
    finally:
        from core.thermodynamics import compute_landauer_cost as _orig

        mod.compute_landauer_cost = _orig
    assert "constraint satisfied" in out


# ===========================================================================
# Example 05 – BOLD thermodynamics
# ===========================================================================


def test_example_05_bold_thermodynamics():
    mod = _load("05_bold_thermodynamics.py")
    _, out = _run(mod.main)
    assert "Summary" in out


def test_example_05_as_main():
    """Cover `if __name__ == '__main__': main()` (line 212)."""
    out = _run_as_main("05_bold_thermodynamics.py")
    assert "Summary" in out


def test_example_05_zero_landauer():
    """Cover line 174: efficiency=0.0 when C_landauer==0.

    Patch APGIPipeline.step to return C_landauer=0 for some steps.
    """
    from pipeline import APGIPipeline

    mod = _load("05_bold_thermodynamics.py")
    orig_step = APGIPipeline.step

    def _step_zero_landauer(self, *args, **kwargs):
        r = orig_step(self, *args, **kwargs)
        r["C_landauer"] = 0.0
        return r

    APGIPipeline.step = _step_zero_landauer
    try:
        _, out = _run(mod.main)
    finally:
        APGIPipeline.step = orig_step
    assert "Summary" in out


# ===========================================================================
# Example 06 – hierarchical system (has main())
# ===========================================================================


def test_example_06_hierarchical_system():
    mod = _load("06_hierarchical_system.py")
    _, out = _run(mod.main)
    assert "Hierarchical Examples Complete" in out


def test_example_06_as_main():
    """Cover `if __name__ == '__main__': main()` (line 343)."""
    out = _run_as_main("06_hierarchical_system.py")
    assert "Hierarchical Examples Complete" in out


# ===========================================================================
# Example 07 – hierarchical system (no main(); if __name__ == "__main__")
# ===========================================================================


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_generate_hierarchical_input(mock_savefig, mock_show):
    mod = _load("07_hierarchical_system.py")
    x_e, x_i = mod.generate_hierarchical_input(n_steps=100)
    assert x_e.shape == (100,)
    assert x_i.shape == (100,)


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_run_simulation_no_plot(mock_savefig, mock_show):
    mod = _load("07_hierarchical_system.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = mod.run_hierarchical_simulation(hierarchical_mode="off", n_steps=300, plot=False)
    assert "S" in results and "theta" in results


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_run_simulation_with_plot(mock_savefig, mock_show):
    """Cover the plot_results branch by passing plot=True."""
    mod = _load("07_hierarchical_system.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = mod.run_hierarchical_simulation(hierarchical_mode="full", n_steps=300, plot=True)
    mock_show.assert_called()
    assert "spectral_result" in results


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_plot_results_directly(mock_savefig, mock_show):
    """Call plot_results directly to cover all plotting lines."""
    mod = _load("07_hierarchical_system.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = mod.run_hierarchical_simulation(
            hierarchical_mode="basic", n_steps=300, plot=False
        )
    mod.plot_results(results, "test")
    mock_show.assert_called()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_compare_modes(mock_savefig, mock_show):
    """Cover compare_modes() which iterates over all 4 hierarchical modes."""
    mod = _load("07_hierarchical_system.py")
    original = mod.run_hierarchical_simulation

    def fast_sim(hierarchical_mode="full", n_steps=200, plot=False):
        return original(hierarchical_mode=hierarchical_mode, n_steps=200, plot=False)

    mod.run_hierarchical_simulation = fast_sim
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.compare_modes()
    mod.run_hierarchical_simulation = original


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_progress_print(mock_savefig, mock_show):
    """Cover line 174: progress print at 2000-step multiples."""
    mod = _load("07_hierarchical_system.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.run_hierarchical_simulation(hierarchical_mode="off", n_steps=2001, plot=False)
    assert "Step 2000" in buf.getvalue()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_07_as_main(mock_savefig, mock_show):
    """Cover the if __name__ == '__main__' block (lines 382-412).

    Uses runpy so those file-level lines are actually traced by coverage.
    Accepts ~7 s for the heavy simulations.
    """
    path = str(EXAMPLES_DIR / "07_hierarchical_system.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        runpy.run_path(path, run_name="__main__")
    assert "Examples complete" in buf.getvalue()


# ===========================================================================
# Example 08 – spectral validation (has main())
# ===========================================================================


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_08_spectral_validation(mock_savefig, mock_show):
    mod = _load("08_spectral_validation.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    assert "Spectral Validation Complete" in buf.getvalue()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_08_as_main(mock_savefig, mock_show):
    """Cover `if __name__ == '__main__': main()` (line 453)."""
    out = _run_as_main("08_spectral_validation.py")
    assert "Spectral Validation Complete" in out


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_08_demonstrate_analytic_formula(mock_savefig, mock_show):
    mod = _load("08_spectral_validation.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.demonstrate_analytic_formula(n_levels=3)
    assert "Lorentzian" in buf.getvalue()


@patch("matplotlib.pyplot.show")
@patch("matplotlib.pyplot.savefig")
def test_example_08_run_and_validate(mock_savefig, mock_show):
    mod = _load("08_spectral_validation.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pipeline, eps_e, eps_i = mod.run_hierarchical_simulation(n_steps=500, n_levels=3)
        results = mod.validate_lorentzian_superposition(pipeline, n_levels=3)
    assert "beta_observed" in results


def test_example_08_no_multiscale_signal_history():
    """Cover lines 194-195: fallback when multiscale_signal_history absent."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
        mod = _load("08_spectral_validation.py")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline, _, _ = mod.run_hierarchical_simulation(n_steps=300, n_levels=3)
        # Remove the history attribute to trigger the fallback branch
        del pipeline.multiscale_signal_history
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            results = mod.validate_lorentzian_superposition(pipeline, n_levels=3)
    assert "main signal S" in buf2.getvalue()


def test_example_08_no_per_level_history():
    """Cover lines 219-220: fallback variances when per_level_history absent."""
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
        mod = _load("08_spectral_validation.py")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            pipeline, _, _ = mod.run_hierarchical_simulation(n_steps=300, n_levels=3)
        # Remove per_level_history to trigger fallback variance estimation
        del pipeline.per_level_history
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2):
            results = mod.validate_lorentzian_superposition(pipeline, n_levels=3)
    assert "estimated variances" in buf2.getvalue()


def test_example_08_matplotlib_import_error():
    """Cover lines 444-445: except ImportError when matplotlib unavailable.

    We set sys.modules['matplotlib'] = None so the re-import inside main()'s
    try-block raises ImportError while the already-imported top-level references
    remain intact.
    """
    with patch("matplotlib.pyplot.show"), patch("matplotlib.pyplot.savefig"):
        mod = _load("08_spectral_validation.py")

    # Now disable matplotlib so the try-block import inside main() fails
    buf = io.StringIO()
    with patch.dict(sys.modules, {"matplotlib": None, "matplotlib.pyplot": None}):
        with contextlib.redirect_stdout(buf):
            mod.main()
    assert "not available" in buf.getvalue()


# ===========================================================================
# Example 09 – Kuramoto coupling (has main())
# ===========================================================================


def test_example_09_kuramoto_coupling():
    mod = _load("09_kuramoto_coupling.py")
    _, out = _run(mod.main)
    assert "Kuramoto" in out


def test_example_09_as_main():
    """Cover `if __name__ == '__main__': main()` (line 400)."""
    out = _run_as_main("09_kuramoto_coupling.py")
    assert "Kuramoto" in out


# ===========================================================================
# Example 10 – reservoir as threshold (module-level script, no functions)
# ===========================================================================


def test_example_10_reservoir_as_threshold():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _load("10_reservoir_as_threshold.py")
    assert "Reservoir" in buf.getvalue()


# ===========================================================================
# Example 11 – maturity assessment (no main(); if __name__ == "__main__")
# ===========================================================================


def test_example_11_assess_system_maturity():
    mod = _load("11_maturity_assessment.py")
    original_sim = mod.run_hierarchical_simulation_for_assessment

    def fast_sim(n_steps=500, n_levels=4):
        return original_sim(n_steps=500, n_levels=n_levels)

    mod.run_hierarchical_simulation_for_assessment = fast_sim
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        score = mod.assess_system_maturity()
    mod.run_hierarchical_simulation_for_assessment = original_sim
    assert hasattr(score, "overall_score")


def test_example_11_compare_configurations():
    mod = _load("11_maturity_assessment.py")
    original_sim = mod.run_hierarchical_simulation_for_assessment

    def fast_sim(n_steps=500, n_levels=4):
        return original_sim(n_steps=500, n_levels=n_levels)

    mod.run_hierarchical_simulation_for_assessment = fast_sim
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.compare_configurations()
    mod.run_hierarchical_simulation_for_assessment = original_sim
    assert "SUMMARY" in buf.getvalue()


def test_example_11_as_main():
    """Cover the if __name__ == '__main__' block (lines 371-379).

    Uses runpy.run_path so coverage sees those literal file lines execute.
    The __main__ block calls assess_system_maturity() (n_steps=5000) and
    compare_configurations() (4 configs × n_steps=2000), totalling ~13k steps
    which complete in ~10 s.
    """
    out = _run_as_main("11_maturity_assessment.py")
    assert "ASSESSMENT COMPLETE" in out


def test_example_11_phi_fallback_path():
    """Cover line 116: fallback when phi_e_levels / phi_i_levels absent (for ell > 0).

    Strategy: restore phi before each step (so the step can write them), then
    delete from the instance __dict__ after the step returns.  The post-step
    hasattr(pipeline, "phi_e_levels") check in ex11 then sees False → line 116.
    """
    from pipeline import APGIPipeline

    mod = _load("11_maturity_assessment.py")
    orig_step = APGIPipeline.step

    def _step_hide_phi(self, *a, **kw):
        # Restore saved phi before running the real step (step writes to them).
        if hasattr(self, "_phi_e_saved"):
            self.__dict__["phi_e_levels"] = self._phi_e_saved
            self.__dict__["phi_i_levels"] = self._phi_i_saved
        r = orig_step(self, *a, **kw)
        # Hide phi after step so the caller's hasattr sees False.
        self._phi_e_saved = self.__dict__.pop("phi_e_levels", None)
        self._phi_i_saved = self.__dict__.pop("phi_i_levels", None)
        return r

    APGIPipeline.step = _step_hide_phi
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.run_hierarchical_simulation_for_assessment(n_steps=5, n_levels=4)
    finally:
        APGIPipeline.step = orig_step


def test_example_11_hierarchical_none_path():
    """Cover lines 132-133: else when pipeline.hierarchical is None/falsy.

    Restore the real hierarchical before each step (step uses it), then set it
    to None after the step so example 11's hasattr+value check sees False → else.
    """
    from pipeline import APGIPipeline

    mod = _load("11_maturity_assessment.py")
    orig_step = APGIPipeline.step

    def _step_hide_hierarchical(self, *a, **kw):
        # Restore real hierarchical so the step can use it
        if hasattr(self, "_hierarchical_saved"):
            self.__dict__["hierarchical"] = self._hierarchical_saved
        r = orig_step(self, *a, **kw)
        # After step: save and set to None so outer check triggers else-branch
        self._hierarchical_saved = self.__dict__.get("hierarchical")
        self.__dict__["hierarchical"] = None
        return r

    APGIPipeline.step = _step_hide_hierarchical
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.run_hierarchical_simulation_for_assessment(n_steps=5, n_levels=4)
    finally:
        APGIPipeline.step = orig_step


def test_example_11_short_thetas_path():
    """Cover lines 126-131: else when ell >= len(hierarchical.thetas)."""
    from pipeline import APGIPipeline

    mod = _load("11_maturity_assessment.py")
    orig_step = APGIPipeline.step

    class _ShortThetas:
        thetas: list = []  # empty → ell < len(thetas) always False

    def _step_short_thetas(self, *a, **kw):
        r = orig_step(self, *a, **kw)
        if self.__dict__.get("hierarchical") is not None:
            self.__dict__["hierarchical"] = _ShortThetas()
        return r

    APGIPipeline.step = _step_short_thetas
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            mod.run_hierarchical_simulation_for_assessment(n_steps=5, n_levels=4)
    finally:
        APGIPipeline.step = orig_step


def test_example_11_statistical_priority_branch():
    """Cover lines 261-264: 'Priority 1: Improve Statistical Validation' branch.

    Mock assess_overall_maturity to return a score where the statistical gap
    is larger than the hierarchical gap, forcing the else branch.
    """
    from stats.maturity_assessment import MaturityScore

    mod = _load("11_maturity_assessment.py")
    original_sim = mod.run_hierarchical_simulation_for_assessment

    def fast_sim(n_steps=500, n_levels=4):
        return original_sim(n_steps=500, n_levels=n_levels)

    mod.run_hierarchical_simulation_for_assessment = fast_sim

    # Build a mock score: hierarchical_score=90 (small gap), statistical_score=30 (large gap)
    mock_score = MaturityScore(
        overall_score=60.0,
        hierarchical_score=90.0,
        statistical_score=30.0,
        pac_score=90.0,
        cascade_score=90.0,
        coherence_score=90.0,
        spectral_score=30.0,
        spectral_signature=None,
        issues=["Low spectral score"],
        recommendations=["Improve spectral", "Check coupling"],
    )

    with patch.object(mod, "assess_overall_maturity", return_value=mock_score):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            score = mod.assess_system_maturity()

    mod.run_hierarchical_simulation_for_assessment = original_sim
    assert "Statistical Validation" in buf.getvalue()
    assert "Improve spectral" in buf.getvalue()  # line 272: recommendations loop


# ===========================================================================
# Example 12 – maturity demo (has main())
# ===========================================================================


def test_example_12_maturity_demo():
    mod = _load("12_maturity_demo.py")
    _, out = _run(mod.main)
    assert "Maturity" in out


def test_example_12_as_main():
    """Cover `if __name__ == '__main__': main()` (line 171)."""
    out = _run_as_main("12_maturity_demo.py")
    assert "Maturity" in out


def test_example_12_generate_pink_noise():
    mod = _load("12_maturity_demo.py")
    signal = mod.generate_pink_noise(n_samples=1000, seed=0)
    assert len(signal) == 1000


def test_example_12_generate_hierarchical_signals():
    mod = _load("12_maturity_demo.py")
    signals = mod.generate_hierarchical_signals(n_samples=1000, n_levels=3, seed=0)
    assert len(signals) == 3


def test_example_12_demo_functions():
    mod = _load("12_maturity_demo.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.demo_spectral_extraction()
        mod.demo_hierarchical_assessment()
        mod.demo_configuration_comparison()
    assert "Spectral" in buf.getvalue()


def test_example_12_demo_configuration_exception():
    """Cover lines 139-140: except block in demo_configuration_comparison.

    Patch extract_1f_signature to raise on the first call.
    """
    mod = _load("12_maturity_demo.py")
    call_count = [0]
    real_extract = mod.extract_1f_signature

    def mock_extract(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise ValueError("Forced extract error")
        return real_extract(*args, **kwargs)

    mod.extract_1f_signature = mock_extract
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.demo_configuration_comparison()
    assert "Error" in buf.getvalue()


# ===========================================================================
# Example 13 – end-to-end validation (has main())
# ===========================================================================


def test_example_13_validation_e2e():
    mod = _load("13_validation_e2e.py")
    _, out = _run(mod.main)
    assert "Validation Complete" in out


def test_example_13_as_main():
    """Cover `if __name__ == '__main__': main()` (line 177)."""
    out = _run_as_main("13_validation_e2e.py")
    assert "Validation Complete" in out


def test_example_13_generate_synthetic_data():
    mod = _load("13_validation_e2e.py")
    x_e, x_i = mod.generate_synthetic_data(n_steps=100)
    assert len(x_e) == 100
    assert len(x_i) == 100


def test_example_13_insufficient_data():
    """Cover lines 58-59: early return when history has < 64 entries."""
    from config import CONFIG
    from pipeline import APGIPipeline

    mod = _load("13_validation_e2e.py")

    # Build a pipeline with only 10 theta entries
    config = dict(CONFIG)
    pipeline = APGIPipeline(config)
    for _ in range(10):
        pipeline.step(0.1, 0.1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        result = mod.validate_spectral_characteristics(pipeline)
    assert "Insufficient data" in buf.getvalue()
    assert result == (0.0, None)


def test_example_13_pink_noise_exception():
    """Cover lines 83-86: except block when validate_pink_noise raises.

    The function does `from stats.spectral_model import validate_pink_noise`
    locally, so we patch at the source module level.
    """
    from config import CONFIG
    from pipeline import APGIPipeline

    mod = _load("13_validation_e2e.py")

    config = dict(CONFIG)
    pipeline = APGIPipeline(config)
    for _ in range(100):
        pipeline.step(0.1, 0.1)

    buf = io.StringIO()
    with patch("stats.spectral_model.validate_pink_noise", side_effect=ValueError("Forced")):
        with contextlib.redirect_stdout(buf):
            hurst, pink = mod.validate_spectral_characteristics(pipeline)
    assert "validation skipped" in buf.getvalue()
    assert pink is None


def test_example_13_observable_mapping_not_enabled():
    """Cover lines 93-94: early return when observables are None."""
    from config import CONFIG
    from pipeline import APGIPipeline

    mod = _load("13_validation_e2e.py")

    # Pipeline without observable mapping
    config = dict(CONFIG)
    pipeline = APGIPipeline(config)
    pipeline.step(0.1, 0.1)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.validate_observable_predictions(pipeline)
    assert "not enabled" in buf.getvalue()


def test_example_13_observable_predictions_with_data():
    """Cover lines 104-112: printing observable stats from pipeline.history."""
    from config import CONFIG
    from pipeline import APGIPipeline

    mod = _load("13_validation_e2e.py")

    config = dict(CONFIG)
    config["use_observable_mapping"] = True
    pipeline = APGIPipeline(config)
    pipeline.step(0.5, 0.2)

    # Inject observable data into history (pipeline stores per-step results
    # only in the result dict, not in history — inject manually for coverage)
    pipeline.history["neural_gamma_power"] = [0.5, 0.6]
    pipeline.history["neural_erp_amplitude"] = [1.0, 1.1]
    pipeline.history["neural_ignition_rate"] = [0.1, 0.2]
    pipeline.history["behavioral_rt_variability"] = [0.4, 0.5]
    pipeline.history["behavioral_response_criterion"] = [0.9, 1.0]

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.validate_observable_predictions(pipeline)
    assert "gamma power" in buf.getvalue()


def test_example_13_spectral_validation_exception():
    """Cover lines 150-153: except block when validate_spectral_characteristics raises."""
    from config import CONFIG
    from pipeline import APGIPipeline

    mod = _load("13_validation_e2e.py")

    def mock_validate_spectral(pipeline):
        raise RuntimeError("Forced spectral failure")

    mod.validate_spectral_characteristics = mock_validate_spectral

    config = dict(CONFIG)
    config["use_observable_mapping"] = True
    config["use_stability_analysis"] = True
    config["stochastic_ignition"] = False
    pipeline = APGIPipeline(config)

    x_e, x_i = mod.generate_synthetic_data(n_steps=200)
    for i in range(200):
        pipeline.step(x_e[i], x_i[i])

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    # The spectral exception is caught; validation still completes
    assert "Validation Complete" in buf.getvalue()


# ===========================================================================
# Example 14 – BOLD calibration (has main())
# ===========================================================================


def test_example_14_bold_calibration():
    mod = _load("14_bold_calibration.py")
    _, out = _run(mod.main)
    assert "BOLD" in out


def test_example_14_as_main():
    """Cover `if __name__ == '__main__': main()` (line 181)."""
    out = _run_as_main("14_bold_calibration.py")
    assert "BOLD" in out


# ===========================================================================
# Example 15 – hierarchy power spectrum (has main())
# ===========================================================================


def test_example_15_hierarchy_power_spectrum():
    mod = _load("15_hierarchy_power_spectrum.py")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    assert "Summary" in buf.getvalue()


def test_example_15_as_main():
    """Cover `if __name__ == '__main__': main()` (line 99)."""
    out = _run_as_main("15_hierarchy_power_spectrum.py")
    assert "Summary" in out


def test_example_15_dfa_low_alpha_warning():
    """Cover line 83: warning when DFA alpha < 0.55."""
    mod = _load("15_hierarchy_power_spectrum.py")

    # Mock estimate_hurst_dfa to return a low alpha (< 0.55)
    mod.estimate_hurst_dfa = lambda signal: (0.4, 0.9)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    assert "Markovian" in buf.getvalue()


def test_example_15_dfa_exception():
    """Cover lines 86-87: except block when estimate_hurst_dfa raises."""
    mod = _load("15_hierarchy_power_spectrum.py")

    def mock_hurst(signal):
        raise RuntimeError("Forced DFA failure")

    mod.estimate_hurst_dfa = mock_hurst
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mod.main()
    assert "Failed to compute DFA" in buf.getvalue()
