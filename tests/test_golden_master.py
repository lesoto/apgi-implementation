"""Golden-master regression on published-figure trajectories.

The paper figures are produced by driving the pipeline with a fixed seed. A
refactor that silently shifts a trajectory would change those figures without
failing any behavioural test — the numbers would still be finite, ordered and
plausible.

These tests pin the trajectory digest for each shipped environment config. A
deliberate change to the dynamics WILL fail them; that is the point. When the
change is intended, regenerate with:

    python -m tests.test_golden_master --update

and state the numerical change in the commit message.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from core.config_io import load_config
from pipeline import APGIPipeline

GOLDEN_PATH = Path(__file__).parent / "golden" / "trajectories.json"

#: Digest precision. Tight enough to catch a real change in the dynamics, loose
#: enough to survive last-bit differences in libm across platforms and BLAS
#: builds — the trajectories are rounded before hashing.
ROUND_DP = 9

CASES = {
    "prod": {"config": "configs/prod.toml", "steps": 200},
    "staging": {"config": "configs/staging.toml", "steps": 100},
    "dev": {"config": "configs/dev.toml", "steps": 200},
}


def _trajectory(config_path: str, steps: int) -> list[list[float]]:
    """Drive the pipeline deterministically and return the rounded trajectory."""
    config = load_config(config_path, use_env=False)
    pipeline = APGIPipeline(config)
    rows = []
    for t in range(steps):
        r = pipeline.step(
            x_e=float(np.sin(0.05 * t) + 0.5 * np.sin(0.01 * t)),
            x_i=float(0.5 + 0.3 * np.cos(0.02 * t)),
        )
        rows.append(
            [
                round(float(r["S"]), ROUND_DP),
                round(float(r["theta"]), ROUND_DP),
                float(r["B"]),
            ]
        )
    return rows


def _digest(rows: list[list[float]]) -> str:
    return hashlib.sha256(
        json.dumps(rows, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _load_golden() -> dict[str, dict[str, object]]:
    if not GOLDEN_PATH.is_file():
        pytest.skip(f"golden file missing: run `python -m tests.test_golden_master --update`")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectory_matches_golden_master(name: str):
    golden = _load_golden()
    assert name in golden, f"no golden entry for {name!r}; regenerate with --update"
    case = CASES[name]
    rows = _trajectory(str(case["config"]), int(case["steps"]))
    expected = golden[name]
    assert _digest(rows) == expected["digest"], (
        f"trajectory for {name!r} changed.\n"
        f"  ignitions now={sum(r[2] for r in rows):.0f} "
        f"was={expected['ignition_count']}\n"
        f"  final S now={rows[-1][0]} was={expected['final_S']}\n"
        f"  final theta now={rows[-1][1]} was={expected['final_theta']}\n"
        "If this change is intended, regenerate the golden file with "
        "`python -m tests.test_golden_master --update` and state the numerical "
        "change in the commit message."
    )


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectory_is_self_consistent_across_repeats(name: str):
    """Guards the golden file itself: a nondeterministic run makes it useless."""
    case = CASES[name]
    first = _trajectory(str(case["config"]), int(case["steps"]))
    second = _trajectory(str(case["config"]), int(case["steps"]))
    assert _digest(first) == _digest(second)


def _regenerate() -> None:
    """Write the golden file from the current implementation."""
    payload = {}
    for name, case in sorted(CASES.items()):
        rows = _trajectory(str(case["config"]), int(case["steps"]))
        payload[name] = {
            "config": case["config"],
            "steps": case["steps"],
            "digest": _digest(rows),
            "ignition_count": sum(r[2] for r in rows),
            "final_S": rows[-1][0],
            "final_theta": rows[-1][1],
        }
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH}")
    for name, entry in payload.items():
        print(f"  {name}: {entry['ignition_count']:.0f} ignitions, digest {entry['digest'][:12]}…")


if __name__ == "__main__":
    import sys

    if "--update" in sys.argv:
        _regenerate()
    else:
        print(__doc__)
        print("Pass --update to regenerate the golden file.")
