"""Centralised random-number generation for APGI.

Every stochastic component in APGI draws from a single, explicitly seedable
generator so that a run is bit-for-bit reproducible from its seed alone.

Rationale
---------
Prior to this module the codebase mixed two mutually invisible RNG streams:

  * the legacy global stream (``np.random.normal``), affected by
    ``np.random.seed()``; and
  * fresh per-call generators (``np.random.default_rng()``), which ignore
    ``np.random.seed()`` entirely.

``main.py --seed`` therefore had no effect on ignition sampling, and two runs
with an identical seed produced different ignition trains. Because this
repository is the cited Data/Code Availability artifact for the APGI paper
series, non-reproducible stochastic runs defeat its stated purpose.

Contract
--------
* Every stochastic API accepts ``rng``: ``None``, an ``int`` seed, a
  ``SeedSequence``, or a ``numpy.random.Generator``.
* ``rng=None`` resolves to the process-global APGI generator, which
  :func:`seed_everything` sets deterministically.
* No APGI code path calls ``np.random.*`` (legacy global) or bare
  ``np.random.default_rng()`` directly.

Examples
--------
>>> from core.rng import seed_everything, resolve_rng
>>> seed_everything(42)  # doctest: +ELLIPSIS
Generator(PCG64) at ...
>>> a = resolve_rng(None).normal()
>>> seed_everything(42)  # doctest: +ELLIPSIS
Generator(PCG64) at ...
>>> b = resolve_rng(None).normal()
>>> a == b
True
"""

from __future__ import annotations

from typing import TypeAlias

import numpy as np

__all__ = [
    "RNGLike",
    "resolve_rng",
    "seed_everything",
    "get_global_rng",
    "spawn_rng",
]

#: Anything accepted where a generator is expected.
RNGLike: TypeAlias = "None | int | np.random.SeedSequence | np.random.Generator"

# Process-global generator. Unseeded by default (OS entropy), so behaviour is
# unchanged for callers that never seed; seed_everything() makes it deterministic.
_GLOBAL_RNG: np.random.Generator = np.random.default_rng()
_GLOBAL_SEED: int | None = None


def seed_everything(seed: int | None) -> np.random.Generator:
    """Seed the process-global APGI generator (and NumPy's legacy global stream).

    This is the single entry point for reproducibility. Seeding the legacy
    stream too means any third-party or not-yet-migrated code that still calls
    ``np.random.*`` also becomes deterministic.

    Args:
        seed: Integer seed. ``None`` re-seeds from OS entropy (non-deterministic).

    Returns:
        The newly seeded process-global generator.
    """
    global _GLOBAL_RNG, _GLOBAL_SEED
    _GLOBAL_SEED = seed
    _GLOBAL_RNG = np.random.default_rng(seed)
    if seed is not None:
        # Keep the legacy global stream in lockstep so that any residual
        # np.random.* call site is deterministic too. Derived (not equal) to
        # avoid correlating the two streams.
        legacy_seed = int(np.random.SeedSequence(seed).generate_state(1)[0])
        np.random.seed(legacy_seed)
    return _GLOBAL_RNG


def get_global_rng() -> np.random.Generator:
    """Return the process-global APGI generator."""
    return _GLOBAL_RNG


def get_global_seed() -> int | None:
    """Return the seed last passed to :func:`seed_everything`, if any.

    Used by the run manifest to record how a run can be reproduced.
    """
    return _GLOBAL_SEED


def resolve_rng(rng: RNGLike = None) -> np.random.Generator:
    """Coerce ``rng`` into a ``numpy.random.Generator``.

    Args:
        rng: ``None`` (use the process-global generator), an ``int`` seed, a
            ``SeedSequence``, or an existing ``Generator`` (returned as-is).

    Returns:
        A generator instance.

    Raises:
        TypeError: If ``rng`` is not one of the accepted types.
    """
    if rng is None:
        return _GLOBAL_RNG
    if isinstance(rng, np.random.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return np.random.default_rng(int(rng))
    if isinstance(rng, np.random.SeedSequence):
        return np.random.default_rng(rng)
    raise TypeError(
        f"rng must be None, int, SeedSequence, or numpy.random.Generator, got {type(rng).__name__}"
    )


def spawn_rng(rng: RNGLike = None, n: int = 1) -> list[np.random.Generator]:
    """Derive ``n`` independent child generators from ``rng``.

    Use when a component needs its own stream that must not consume draws from
    the parent (e.g. per-level hierarchy noise), while remaining deterministic
    under the parent's seed.

    Args:
        rng: Parent generator specification.
        n: Number of children to spawn.

    Returns:
        List of ``n`` independent generators.

    Raises:
        ValueError: If ``n`` < 1.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    parent = resolve_rng(rng)
    # Prefer the PUBLIC API (numpy >= 1.25). The private
    # `bit_generator._seed_seq` fallback is retained only for older numpy
    # within the supported range; relying on it alone would make child-stream
    # determinism depend on a private attribute across a major-version bump.
    spawn = getattr(parent, "spawn", None)
    if callable(spawn):
        children: list[np.random.Generator] = list(spawn(n))
        return children
    # Older numpy: SeedSequence.spawn exists on the concrete class but not on
    # the ISeedSequence protocol that bit_generator.seed_seq is typed as.
    seed_seq = parent.bit_generator.seed_seq
    return [np.random.default_rng(s) for s in seed_seq.spawn(n)]  # type: ignore[attr-defined]
