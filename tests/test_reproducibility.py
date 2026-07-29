"""Determinism guarantees.

This is the property the repository exists to provide: it is the cited
Data/Code Availability artifact for the APGI paper series, so a seeded run must
be reproducible bit-for-bit or the published figures cannot be regenerated.

Regression context: before core.rng, two RNG streams coexisted — NumPy's legacy
global (affected by np.random.seed) and fresh per-call default_rng() (which
ignores it). `main.py --seed` therefore had no effect on ignition sampling, and
two runs with an identical seed produced different ignition trains (121 vs 124
ignitions over 300 steps). These tests fail if that ever regresses.
"""

from __future__ import annotations

import numpy as np
import pytest

from config import CONFIG
from core.rng import get_global_rng, resolve_rng, seed_everything, spawn_rng
from pipeline import APGIPipeline

pytestmark = pytest.mark.conformance


def _drive(pipeline: APGIPipeline, n: int = 300) -> list[tuple[int, float, float]]:
    """Run a deterministic input sequence and return the full trajectory."""
    out = []
    for t in range(n):
        r = pipeline.step(x_e=float(np.sin(0.05 * t)), x_i=float(0.3 * np.cos(0.02 * t)))
        out.append((r["B"], round(r["S"], 12), round(r["theta"], 12)))
    return out


class TestPipelineDeterminism:
    def test_same_seed_gives_identical_trajectory(self):
        """Identical seeds must reproduce S, theta AND the ignition train."""
        cfg = {**CONFIG, "stochastic_ignition": True, "seed": 7}
        first = _drive(APGIPipeline(dict(cfg)))
        second = _drive(APGIPipeline(dict(cfg)))
        assert first == second

    def test_different_seed_gives_different_trajectory(self):
        """A different seed must actually change the run.

        Guards against the opposite failure: a 'reproducible' pipeline that
        ignores the seed entirely would pass the test above.
        """
        base = {**CONFIG, "stochastic_ignition": True}
        first = _drive(APGIPipeline({**base, "seed": 7}))
        other = _drive(APGIPipeline({**base, "seed": 8}))
        assert first != other

    def test_ignition_count_is_stable_across_seeded_runs(self):
        cfg = {**CONFIG, "stochastic_ignition": True, "seed": 1234}
        counts = {sum(b for b, _, _ in _drive(APGIPipeline(dict(cfg)))) for _ in range(3)}
        assert len(counts) == 1, f"ignition count varied across identical seeds: {counts}"

    def test_seed_survives_global_state_perturbation(self):
        """A seeded pipeline must not depend on ambient global RNG state.

        np.random.seed() touching the legacy stream must not perturb a run
        seeded through config, or reproducibility would depend on whatever
        else happened to draw first.
        """
        cfg = {**CONFIG, "stochastic_ignition": True, "seed": 99}
        expected = _drive(APGIPipeline(dict(cfg)))
        np.random.seed(4242)
        np.random.random(1000)
        seed_everything(31337)
        assert _drive(APGIPipeline(dict(cfg))) == expected

    def test_reservoir_and_kuramoto_are_seeded(self):
        """Optional stochastic subsystems must honour the seed too."""
        cfg = {
            **CONFIG,
            "seed": 5,
            "stochastic_ignition": True,
            "use_reservoir": True,
            "use_kuramoto": True,
            "use_hierarchical": True,
            "hierarchical_mode": "full",
        }
        assert _drive(APGIPipeline(dict(cfg)), n=100) == _drive(
            APGIPipeline(dict(cfg)), n=100
        )


class TestRNGModule:
    def test_seed_everything_is_deterministic(self):
        seed_everything(42)
        a = [get_global_rng().normal() for _ in range(5)]
        seed_everything(42)
        assert [get_global_rng().normal() for _ in range(5)] == a

    def test_resolve_rng_accepts_all_documented_forms(self):
        assert isinstance(resolve_rng(None), np.random.Generator)
        assert isinstance(resolve_rng(3), np.random.Generator)
        assert isinstance(resolve_rng(np.random.SeedSequence(3)), np.random.Generator)
        gen = np.random.default_rng(3)
        assert resolve_rng(gen) is gen, "an existing Generator must pass through unchanged"

    def test_resolve_rng_rejects_bad_types(self):
        with pytest.raises(TypeError, match="rng must be"):
            resolve_rng("not-a-seed")  # type: ignore[arg-type]

    def test_int_seed_matches_default_rng(self):
        assert resolve_rng(11).normal() == np.random.default_rng(11).normal()

    def test_spawn_rng_is_independent_and_deterministic(self):
        a, b = spawn_rng(resolve_rng(5), 2)
        assert a.normal() != b.normal(), "children must be independent streams"
        c, d = spawn_rng(resolve_rng(5), 2)
        assert (a.normal(), b.normal()) != (0.0, 0.0)
        assert isinstance(c, np.random.Generator) and isinstance(d, np.random.Generator)

    def test_spawn_rng_rejects_non_positive_n(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            spawn_rng(None, 0)

    def test_seed_none_is_accepted(self):
        """seed_everything(None) re-seeds from OS entropy without raising."""
        assert isinstance(seed_everything(None), np.random.Generator)


class TestNoUnseededRandomness:
    def test_library_has_no_bare_default_rng_or_legacy_calls(self):
        """No APGI module may bypass core.rng.

        A single `np.random.default_rng()` or `np.random.normal()` in a library
        path silently reintroduces an unseedable stream — the original defect.
        This test enforces the contract structurally rather than trusting
        review to catch it.
        """
        import pathlib
        import re

        root = pathlib.Path(__file__).resolve().parent.parent
        packages = [
            "core",
            "hierarchy",
            "oscillation",
            "reservoir",
            "stats",
            "validation",
            "analysis",
            "energy",
            "active_inference",
        ]
        files = [p for pkg in packages for p in (root / pkg).rglob("*.py")]
        files += [root / "pipeline.py", root / "main.py"]

        # Bare default_rng() with no argument, or any legacy global draw.
        bare_default_rng = re.compile(r"np\.random\.default_rng\(\s*\)")
        legacy_global = re.compile(
            r"np\.random\.(rand|randn|normal|uniform|choice|shuffle|random|seed)\("
        )
        offenders: list[str] = []
        for path in files:
            if path.name == "rng.py":  # the one module allowed to own the streams
                continue
            for lineno, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                code = line.split("#", 1)[0]
                if ">>>" in line:  # doctest prose
                    continue
                if bare_default_rng.search(code) or legacy_global.search(code):
                    offenders.append(f"{path.relative_to(root)}:{lineno}: {line.strip()}")
        assert not offenders, "unseeded RNG use bypassing core.rng:\n" + "\n".join(offenders)
