#!/usr/bin/env python3
"""
APGI BOLD-Calibrated Thermodynamic Analysis Example

This example demonstrates thermodynamic analysis using BOLD fMRI calibration
to convert from arbitrary units to Joules per bit erased (κ_meta).

Spec Reference: §11 Thermodynamic Constraints with BOLD calibration
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from core.thermodynamics import compute_information_bits  # noqa: E402
from core.thermodynamics import compute_landauer_cost
from energy.bold_calibration import BOLDCalibrator  # noqa: E402
from energy.bold_calibration import validate_energy_against_landauer
from pipeline import APGIPipeline  # noqa: E402


def main():
    """Run BOLD-calibrated thermodynamic analysis example."""

    print("=" * 60)
    print("APGI BOLD-Calibrated Thermodynamic Analysis Example")
    print("=" * 60)

    # Part 1: Demonstrate BOLD calibration
    print("\n1. BOLD fMRI Calibration Demonstration")
    print("-" * 40)

    # Create BOLD calibrator
    calibrator = BOLDCalibrator()

    # Simulate some trials with BOLD measurements
    trials = [
        {"baseline_bold": 1.0, "ignition_bold": 2.5, "bits": 6.6, "duration": 1.0},
        {"baseline_bold": 1.2, "ignition_bold": 3.0, "bits": 8.0, "duration": 1.0},
        {"baseline_bold": 0.8, "ignition_bold": 2.0, "bits": 5.0, "duration": 1.0},
    ]

    print(f"Calibrating κ_meta from {len(trials)} simulated fMRI trials...")
    for i, trial in enumerate(trials):
        kappa = calibrator.calibrate_from_trial(
            trial["baseline_bold"],
            trial["ignition_bold"],
            trial["bits"],
            trial["duration"],
        )
        print(f"  Trial {i + 1}: κ_meta = {kappa:.2e} J/bit")

    summary = calibrator.get_calibration_summary()
    print("\nCalibration Summary:")
    print(f"  κ_meta mean: {summary['kappa_mean']:.2e} J/bit")
    print(f"  κ_meta std: {summary['kappa_std']:.2e} J/bit")
    print(f"  Landauer minimum: {summary['landauer_energy_per_bit_j']:.2e} J/bit")
    print(
        f"  Efficiency ratio: {summary['kappa_mean'] / summary['landauer_energy_per_bit_j']:.1f}x"
    )

    # Part 2: Compare calibrated vs uncalibrated energy calculations
    print("\n2. Calibrated vs Uncalibrated Energy Calculations")
    print("-" * 40)

    S_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    eps = 0.01

    print("Signal | Bits | Landauer (dimensionless κ) | Landauer (calibrated κ)")
    print("-" * 80)

    for S in S_values:
        bits = compute_information_bits(S, eps)

        # Uncalibrated (dimensionless κ_meta = 1.0)
        cost_uncalibrated = compute_landauer_cost(
            S, eps, kappa_meta=1.0, kappa_units="dimensionless"
        )

        # Calibrated (using mean κ from BOLD calibration)
        cost_calibrated = compute_landauer_cost(
            S, eps, kappa_meta=summary["kappa_mean"], kappa_units="joules_per_bit"
        )

        print(
            f"{S:6.1f} | {bits:4.1f} | {cost_uncalibrated:10.2e} J | {cost_calibrated:10.2e} J"
        )

    # Part 3: Validate energy against Landauer's principle
    print("\n3. Energy Validation Against Landauer's Principle")
    print("-" * 40)

    # Simulate measured energy from BOLD
    bold_change = 2.5  # 2.5% BOLD signal change
    bits_erased = 6.6

    # Estimate energy from BOLD
    from energy.bold_calibration import bold_signal_to_energy  # noqa: E402

    measured_energy = bold_signal_to_energy(bold_change)

    # Validate
    validation = validate_energy_against_landauer(measured_energy, bits_erased)

    print(f"BOLD signal change: {bold_change:.1f}%")
    print(f"Bits erased: {bits_erased:.1f}")
    print(f"Measured energy: {validation['measured_energy_j']:.2e} J")
    print(f"Landauer minimum: {validation['landauer_minimum_j']:.2e} J")
    print(f"Ratio (E/E_min): {validation['ratio']:.1f}")
    print(f"Validation: {validation['message']}")

    # Part 4: APGI pipeline with BOLD calibration
    print("\n4. APGI Pipeline with BOLD-Calibrated Thermodynamics")
    print("-" * 40)

    # Create configuration with BOLD calibration
    config = CONFIG.copy()
    config["use_thermodynamic_cost"] = True
    config["use_bold_calibration"] = True
    config["kappa_meta"] = summary["kappa_mean"]  # Use calibrated κ
    config["kappa_units"] = "joules_per_bit"

    print("Configuration with BOLD calibration:")
    print(f"  κ_meta: {config['kappa_meta']:.2e} J/bit")
    print(f"  κ_units: {config['kappa_units']}")
    print(f"  BOLD conversion factor: {config['bold_conversion_factor']:.2e} J/%/cm³")
    print(f"  Ignition spike factor: {config['bold_ignition_spike_factor']:.3f}")

    # Initialize pipeline
    pipeline = APGIPipeline(config)

    # Run a few steps
    n_steps = 100
    print(f"\nRunning {n_steps} steps with BOLD-calibrated thermodynamics...")

    thermodynamic_data = {
        "S": [],
        "landauer_cost": [],
        "info_bits": [],
        "metabolic_cost": [],
        "efficiency": [],
    }

    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)

        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)

        # Record data
        S = result["S"]
        thermodynamic_data["S"].append(S)

        if "C_landauer" in result:
            thermodynamic_data["landauer_cost"].append(result["C_landauer"])
            thermodynamic_data["info_bits"].append(result.get("bits_erased", 0.0))
            thermodynamic_data["metabolic_cost"].append(result["C"])

            if result["C_landauer"] > 0:
                efficiency = result["C"] / result["C_landauer"]
                thermodynamic_data["efficiency"].append(efficiency)
            else:
                thermodynamic_data["efficiency"].append(0.0)

    # Analyze results
    if thermodynamic_data["landauer_cost"]:
        avg_cost = np.mean(thermodynamic_data["landauer_cost"])
        avg_bits = np.mean(thermodynamic_data["info_bits"])
        avg_efficiency = np.mean([e for e in thermodynamic_data["efficiency"] if e > 0])

        print("\nResults:")
        print(f"  Average Landauer cost: {avg_cost:.2e} J")
        print(f"  Average bits erased: {avg_bits:.1f}")
        print(f"  Average efficiency (C/E_min): {avg_efficiency:.1f}")

        # Validate overall
        total_energy = np.sum(thermodynamic_data["metabolic_cost"])
        total_bits = np.sum(thermodynamic_data["info_bits"])

        # Convert total metabolic cost to Joules (assuming scaling factor)
        scale_factor = 1e20  # Same scaling used in threshold.py
        total_energy_j = total_energy / scale_factor

        overall_validation = validate_energy_against_landauer(
            total_energy_j, total_bits
        )

        print("\nOverall Validation:")
        print(f"  Total energy: {total_energy_j:.2e} J")
        print(f"  Total bits: {total_bits:.1f}")
        print(f"  Validation: {overall_validation['message']}")

    print("\n" + "=" * 60)
    print("Summary:")
    print("  ✓ BOLD calibration converts AU to Joules per bit erased")
    print("  ✓ κ_meta now has physical units (J/bit) instead of dimensionless")
    print("  ✓ 5-10% energy spike during ignition events is modeled")
    print("  ✓ Direct comparison with Landauer's physical minimum enabled")
    print("=" * 60)


if __name__ == "__main__":
    main()
