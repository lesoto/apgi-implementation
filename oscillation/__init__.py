"""Oscillatory phase coupling module for APGI.

Implements phase dynamics ϕ_l(t) = ω_l t + ϕ_0 with cross-level coupling.
"""

from __future__ import annotations

from oscillation.phase import (
    compute_phase,
    phase_coupling_kuramoto,
    update_phase_euler,
)
from oscillation.threshold_modulation import (
    compute_modulation_factor,
    modulate_threshold_by_phase,
)

__all__ = [
    "compute_phase",
    "phase_coupling_kuramoto",
    "update_phase_euler",
    "compute_modulation_factor",
    "modulate_threshold_by_phase",
]
