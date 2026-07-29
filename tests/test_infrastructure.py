"""Tests for the provenance, configuration and numerical-guard modules.

These support reproducibility rather than the model itself: the run manifest
(so a figure can be traced to the code and seed that made it), externalised
configuration (so dev/staging/prod differ by file, not by edited source), and
the degeneracy guards (so constant series produce an explicit error instead of
NaN or LAPACK noise on stderr).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from config import CONFIG
from core.config_io import (
    ENV_PREFIX,
    apply_env_overrides,
    load_config,
    load_config_file,
)
from core.manifest import (
    build_manifest,
    config_hash,
    dependency_versions,
    git_revision,
    write_manifest,
)
from core.numerics import (
    finite_mask,
    is_degenerate,
    safe_corrcoef,
    safe_log,
    safe_nperseg,
)
from pipeline import APGIPipeline


class TestManifest:
    def test_manifest_records_everything_needed_to_reproduce(self):
        m = build_manifest({**CONFIG, "seed": 42})
        assert m["seed"] == 42
        assert len(m["config_sha256"]) == 64
        assert m["python"]["version"]
        assert "numpy" in m["dependencies"]
        assert set(m["git"]) == {"commit", "dirty", "branch"}

    def test_config_hash_is_order_independent_but_content_sensitive(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})

    def test_config_hash_survives_unserialisable_values(self):
        """A live Generator in the config must not break hashing mid-run."""
        digest = config_hash({"rng": np.random.default_rng(0), "seed": 1})
        assert len(digest) == 64

    def test_spec_conformant_flag_tracks_validation_warnings(self):
        assert build_manifest({}, validation_warnings=[])["spec_conformant"] is True
        assert build_manifest({}, validation_warnings=["x"])["spec_conformant"] is False

    def test_manifest_is_json_serialisable(self):
        json.dumps(build_manifest(dict(CONFIG)))

    def test_write_manifest_creates_parents_and_valid_json(self, tmp_path):
        target = write_manifest(tmp_path / "nested" / "run.json", {**CONFIG, "seed": 7})
        assert target.is_file()
        assert json.loads(target.read_text())["seed"] == 7

    def test_pipeline_exposes_its_own_manifest(self, tmp_path):
        pipeline = APGIPipeline({**CONFIG, "seed": 11})
        for _ in range(10):
            pipeline.step(x_e=0.4, x_i=0.2)
        assert pipeline.manifest(n_steps=10)["n_steps"] == 10
        written = pipeline.write_manifest(str(tmp_path / "m.json"), n_steps=10)
        assert json.loads(Path(written).read_text())["seed"] == 11

    def test_git_revision_degrades_gracefully_outside_a_repo(self, tmp_path):
        """An installed wheel has no git metadata; that is expected."""
        info = git_revision(tmp_path)
        assert set(info) == {"commit", "dirty", "branch"}

    def test_dependency_versions_reports_absent_packages_as_none(self):
        versions = dependency_versions()
        assert versions["numpy"] is not None
        assert all(v is None or isinstance(v, str) for v in versions.values())


class TestConfigIO:
    def test_defaults_load(self):
        assert load_config(use_env=False)["pi_min"] == CONFIG["pi_min"]

    def test_env_override_is_type_coerced(self):
        env = {f"{ENV_PREFIX}SEED": "123", f"{ENV_PREFIX}USE_RESERVOIR": "true"}
        resolved = apply_env_overrides(dict(CONFIG), env)
        assert resolved["seed"] == 123
        assert resolved["use_reservoir"] is True

    def test_env_override_parses_none_and_floats(self):
        env = {f"{ENV_PREFIX}SEED": "none", f"{ENV_PREFIX}LAM": "0.25"}
        resolved = apply_env_overrides(dict(CONFIG), env)
        assert resolved["seed"] is None
        assert resolved["lam"] == pytest.approx(0.25)

    def test_unknown_env_override_is_rejected_not_ignored(self):
        """Silently dropping a typo would give a run the operator did not configure."""
        with pytest.raises(ValueError, match="unrecognised APGI_"):
            apply_env_overrides(dict(CONFIG), {f"{ENV_PREFIX}NOT_A_KEY": "1"})

    def test_precedence_is_defaults_then_file_then_env_then_overrides(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text(json.dumps({"seed": 1, "lam": 0.2}))
        resolved = load_config(path, overrides={"seed": 3}, use_env=False)
        assert resolved["seed"] == 3, "explicit overrides win"
        assert resolved["lam"] == pytest.approx(0.2), "file beats defaults"

    def test_toml_apgi_table_is_unwrapped(self, tmp_path):
        path = tmp_path / "c.toml"
        path.write_text("[apgi]\nseed = 5\n")
        assert load_config_file(path)["seed"] == 5

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="config file not found"):
            load_config_file(tmp_path / "absent.toml")

    def test_unsupported_extension_raises(self, tmp_path):
        path = tmp_path / "c.yaml"
        path.write_text("seed: 1")
        with pytest.raises(ValueError, match="unsupported config format"):
            load_config_file(path)

    def test_non_mapping_file_raises(self, tmp_path):
        path = tmp_path / "c.json"
        path.write_text("[1, 2, 3]")
        with pytest.raises(ValueError, match="must contain a mapping"):
            load_config_file(path)

    @pytest.mark.parametrize("name", ["dev", "staging", "prod"])
    def test_shipped_environment_configs_are_valid_and_run(self, name):
        """Every environment config must load, validate and drive the pipeline."""
        config = load_config(f"configs/{name}.toml", use_env=False)
        pipeline = APGIPipeline(config)
        for t in range(20):
            pipeline.step(x_e=float(np.sin(0.05 * t)), x_i=0.3)
        assert pipeline.manifest()["spec_conformant"] is True


class TestNumericalGuards:
    def test_safe_corrcoef_matches_numpy_on_well_posed_input(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = np.array([2.0, 4.1, 5.9, 8.2])
        assert safe_corrcoef(x, y) == pytest.approx(np.corrcoef(x, y)[0, 1])

    def test_safe_corrcoef_returns_default_on_constant_series(self):
        """np.corrcoef divides by zero here and yields NaN, which then
        propagates silently into reported validation metrics."""
        assert safe_corrcoef(np.ones(10), np.arange(10.0)) == 0.0

    def test_safe_corrcoef_handles_nan_and_short_input(self):
        assert safe_corrcoef(np.array([1.0, np.nan, 3.0]), np.arange(3.0)) == 0.0
        assert safe_corrcoef(np.array([1.0]), np.array([2.0])) == 0.0

    def test_safe_corrcoef_rejects_shape_mismatch(self):
        with pytest.raises(ValueError, match="same shape"):
            safe_corrcoef(np.arange(3.0), np.arange(4.0))

    def test_is_degenerate_detection(self):
        assert is_degenerate(np.ones(5))
        assert is_degenerate(np.array([1.0]))
        assert is_degenerate(np.array([1.0, np.inf]))
        assert not is_degenerate(np.arange(5.0))

    def test_safe_nperseg_never_exceeds_the_signal(self):
        assert safe_nperseg(5) <= 5
        assert safe_nperseg(5, requested=256) <= 5
        assert safe_nperseg(10_000, requested=256) == 256

    def test_safe_nperseg_rejects_empty_signal(self):
        with pytest.raises(ValueError, match="signal_length must be >= 1"):
            safe_nperseg(0)

    def test_safe_log_floors_zero_instead_of_returning_inf(self):
        assert np.all(np.isfinite(safe_log(np.array([0.0, 1.0, 10.0]))))

    def test_finite_mask_selects_common_finite_positions(self):
        a = np.array([1.0, np.nan, 3.0])
        b = np.array([1.0, 2.0, np.inf])
        assert finite_mask(a, b).tolist() == [True, False, False]

    def test_finite_mask_validates_inputs(self):
        with pytest.raises(ValueError, match="at least one array"):
            finite_mask()
        with pytest.raises(ValueError, match="shape mismatch"):
            finite_mask(np.arange(3.0), np.arange(4.0))


class TestBoundedHistory:
    def test_history_is_capped_at_maxlen(self):
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 100, "seed": 1})
        for _ in range(1000):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert len(pipeline.history["S"]) == 100

    def test_capped_history_retains_the_most_recent_samples(self):
        """Eviction must be oldest-first so recent dynamics survive."""
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 10, "seed": 1})
        for _ in range(50):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert list(pipeline.history["S"])[-1] == pytest.approx(pipeline.S)

    def test_none_restores_unbounded_history(self):
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": None, "seed": 1})
        for _ in range(200):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert len(pipeline.history["S"]) == 200

    def test_history_is_numpy_convertible(self):
        """Downstream analysis calls np.asarray on these buffers."""
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 64, "seed": 1})
        for _ in range(100):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert np.asarray(pipeline.history["theta"]).shape == (64,)

    def test_invalid_maxlen_is_rejected(self):
        with pytest.raises(ValueError, match="history_maxlen must be >= 1"):
            APGIPipeline({**CONFIG, "history_maxlen": 0, "seed": 1})


class TestReproducePaperTarget:
    """The figure-reproduction target must actually run.

    Regression: the Makefile invoked `examples/figure_s1.py`, a file that has
    never existed in this repository's history (the generator is
    `examples/16_figures.py`, which WRITES outputs/figure_s1.png). So
    `make reproduce-paper` — the headline reproducibility entry point, and the
    target the run manifest is attached to — aborted on its first command.
    Nothing tested it, and neither did the review that added the manifest.
    """

    def _makefile_scripts(self) -> list[Path]:
        root = Path(__file__).resolve().parent.parent
        text = (root / "Makefile").read_text(encoding="utf-8")
        return [
            root / "examples" / line.split("$(EXAMPLES_DIR)/")[1].split()[0]
            for line in text.splitlines()
            if "$(EXAMPLES_DIR)/" in line
        ]

    def test_every_script_the_makefile_invokes_exists(self):
        missing = [str(p) for p in self._makefile_scripts() if not p.is_file()]
        assert not missing, f"Makefile invokes scripts that do not exist: {missing}"

    def test_makefile_references_at_least_one_figure_script(self):
        """Guards the guard: a Makefile with no scripts would pass vacuously."""
        assert self._makefile_scripts()


class TestHistoryBufferContract:
    """The bounded history is a deque, which does NOT support slicing.

    `deque[0]` works but `deque[0:2]` raises TypeError. Consumers currently
    only index (examples/11_maturity_assessment.py) and call np.asarray, both of
    which are fine — these tests pin that contract so a future slice does not
    fail only when history_maxlen happens to be set.
    """

    def test_integer_indexing_works(self):
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 50, "seed": 1})
        for _ in range(100):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert isinstance(pipeline.history["S"][0], float)
        assert isinstance(pipeline.history["S"][-1], float)

    def test_numpy_conversion_and_iteration_work(self):
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 50, "seed": 1})
        for _ in range(100):
            pipeline.step(x_e=0.5, x_i=0.3)
        assert np.asarray(pipeline.history["theta"]).shape == (50,)
        assert len(list(pipeline.history["theta"])) == 50

    def test_slicing_a_bounded_buffer_is_a_known_limitation(self):
        """Documents the sharp edge explicitly rather than leaving it latent."""
        pipeline = APGIPipeline({**CONFIG, "history_maxlen": 50, "seed": 1})
        for _ in range(60):
            pipeline.step(x_e=0.5, x_i=0.3)
        with pytest.raises(TypeError):
            _ = pipeline.history["S"][0:2]
        # The supported idiom for a window:
        assert len(list(pipeline.history["S"])[0:2]) == 2
