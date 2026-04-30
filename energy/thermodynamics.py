from __future__ import annotations

import numpy as np

K_B = 1.380649e-23  # Boltzmann constant (J/K)
ATP_ENERGY = 5.2e-21  # Energy per ATP molecule at ~300K (~50 kJ/mol)
TYPICAL_TEMP = 310.0  # Body temperature in Kelvin (~37°C)


def estimate_bits_erased(S: float, eps_stab: float = 1e-6) -> float:
    """Estimate number of bits erased during ignition per §11.

    N_erase ≈ log₂(S(t) / ε_stab)

    Args:
        S: Signal level at time of ignition
        eps_stab: Numerical stability constant (default 10⁻⁶ per spec)

    Returns:
        Estimated bits erased
    """
    if S <= 0:
        return 0.0
    return float(np.log2(max(S, eps_stab) / eps_stab))


def metabolic_cost_landauer(
    N_erase: float,
    kappa_meta: float,
    T_env: float = TYPICAL_TEMP,
) -> float:
    """Compute metabolic cost mapped to Landauer's bound per §11.

    C(t) ≥ κ_meta · N_erase(t) · k_B · T_env · ln(2)

    Args:
        N_erase: Bits erased
        kappa_meta: Metabolic conversion coefficient
        T_env: Ambient temperature in Kelvin

    Returns:
        Mapped metabolic cost
    """
    e_min = N_erase * K_B * T_env * np.log(2)
    return float(kappa_meta * e_min)


def metabolic_cost(kappa: float, bits: float) -> float:
    """Compute metabolic cost in ATP molecules.

    Formula: C_metabolic = κ · (bits erased per ignition)

    Args:
        kappa: ATP cost per bit (typically 10-1000 ATP/bit)
        bits: Information processed in bits

    Returns:
        Metabolic cost in ATP molecules
    """

    return float(kappa * bits)


def landauer_limit(T: float = TYPICAL_TEMP) -> float:
    """Minimum energy to erase one bit (Landauer limit).

    E_min = k_B · T · ln(2)  joules per bit

    Args:
        T: Temperature in Kelvin (default: body temp 310K)

    Returns:
        Minimum energy in joules per bit
    """

    return float(K_B * T * np.log(2))


def landauer_cost_in_atp(bits: float, T: float = TYPICAL_TEMP) -> float:
    """Convert Landauer limit to ATP equivalents.

    Args:
        bits: Number of bits processed
        T: Temperature in Kelvin

    Returns:
        Minimum ATP molecules required (theoretical lower bound)
    """

    energy_joules = landauer_limit(T) * bits
    return float(energy_joules / ATP_ENERGY)


def estimate_information_content(
    z_e: float,
    z_i: float,
    pi_e: float,
    pi_i: float,
    bits_per_unit: float = 1.0,
) -> float:
    """Estimate information content of prediction errors in bits.

    Uses precision-weighted surprise as proxy for information content.
    Higher precision + larger error = more information.

    Args:
        z_e: Exteroceptive error
        z_i: Interoceptive error
        pi_e: Exteroceptive precision
        pi_i: Interoceptive precision
        bits_per_unit: Scaling to convert to bits

    Returns:
        Estimated bits of information
    """

    # Precision-weighted information
    info_e = pi_e * (z_e**2)
    info_i = pi_i * (z_i**2)

    # Convert to bits (information = -log2(probability))
    # Approximate: higher surprise = more bits
    total_info = (info_e + info_i) * bits_per_unit

    # Cap at reasonable maximum (prevents infinite bits)
    return float(min(total_info, 100.0))


def check_thermodynamic_feasibility(
    bits: float,
    atp_cost: float,
    efficiency: float = 0.1,
    T: float = TYPICAL_TEMP,
) -> dict:
    """Check if metabolic cost is thermodynamically feasible.

    The Landauer limit is a hard lower bound. Real systems operate
    at 10-1000x this limit due to inefficiency.

    Args:
        bits: Information processed in bits
        atp_cost: Actual ATP cost
        efficiency: Assumed thermodynamic efficiency (0.01 to 1.0)
        T: Temperature in Kelvin

    Returns:
        Dictionary with feasibility analysis
    """

    # Theoretical minimum
    min_atp = landauer_cost_in_atp(bits, T)

    # Practical minimum (accounting for efficiency)
    practical_min = min_atp / efficiency

    # Check feasibility
    is_feasible = atp_cost >= practical_min
    margin_factor = atp_cost / practical_min if practical_min > 0 else float("inf")

    return {
        "bits_processed": bits,
        "atp_cost": atp_cost,
        "landauer_minimum_atp": min_atp,
        "practical_minimum_atp": practical_min,
        "efficiency_assumed": efficiency,
        "is_feasible": is_feasible,
        "margin_factor": margin_factor,
        "landauer_violation": atp_cost < min_atp,
        "temperature": T,
    }


class ThermodynamicTracker:
    """Track thermodynamic costs and validate against physical limits."""

    def __init__(
        self,
        kappa: float = 100.0,
        efficiency: float = 0.1,
        temperature: float = TYPICAL_TEMP,
    ):
        """Initialize thermodynamic tracker.

        Args:
            kappa: ATP cost per bit (default: 100 ATP/bit, plausible range 10-1000)
            efficiency: Thermodynamic efficiency (default: 10%)
            temperature: Operating temperature in Kelvin
        """

        self.kappa = kappa
        self.efficiency = efficiency
        self.temperature = temperature

        # Accumulators
        self.total_bits = 0.0
        self.total_atp = 0.0
        self.total_ignitions = 0
        self.history: list[dict] = []

    def record_ignition(
        self,
        z_e: float,
        z_i: float,
        pi_e: float,
        pi_i: float,
        bits_per_unit: float = 1.0,
    ) -> dict:
        """Record thermodynamic cost of one ignition event.

        Args:
            z_e: Exteroceptive error
            z_i: Interoceptive error
            pi_e: Exteroceptive precision
            pi_i: Interoceptive precision
            bits_per_unit: Bits per surprise unit

        Returns:
            Cost breakdown for this ignition
        """

        # Estimate information processed
        bits = estimate_information_content(z_e, z_i, pi_e, pi_i, bits_per_unit)

        # Compute metabolic cost
        atp_cost = metabolic_cost(self.kappa, bits)

        # Check feasibility
        feasibility = check_thermodynamic_feasibility(
            bits, atp_cost, self.efficiency, self.temperature
        )

        # Record
        self.total_bits += bits
        self.total_atp += atp_cost
        self.total_ignitions += 1

        record = {
            "bits": bits,
            "atp_cost": atp_cost,
            "feasibility": feasibility,
        }
        self.history.append(record)

        return record

    def get_summary(self) -> dict:
        """Get cumulative thermodynamic summary."""

        avg_bits = self.total_bits / max(self.total_ignitions, 1)
        avg_atp = self.total_atp / max(self.total_ignitions, 1)

        return {
            "total_ignitions": self.total_ignitions,
            "total_bits_processed": self.total_bits,
            "total_atp_cost": self.total_atp,
            "average_bits_per_ignition": avg_bits,
            "average_atp_per_ignition": avg_atp,
            "kappa": self.kappa,
            "efficiency": self.efficiency,
            "temperature": self.temperature,
            "landauer_minimum_total": landauer_cost_in_atp(self.total_bits, self.temperature),
        }

    def validate_total(self) -> dict:
        """Validate cumulative costs against thermodynamic limits."""

        summary = self.get_summary()
        min_atp = summary["landauer_minimum_total"]
        practical_min = min_atp / self.efficiency

        return {
            "is_physically_possible": summary["total_atp_cost"] >= min_atp,
            "is_biologically_plausible": summary["total_atp_cost"] >= practical_min,
            "efficiency_ratio": (
                min_atp / summary["total_atp_cost"] if summary["total_atp_cost"] > 0 else 0
            ),
            **summary,
        }
