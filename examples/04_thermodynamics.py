#!/usr/bin/env python3
"""
APGI Thermodynamic Analysis Example

This example demonstrates thermodynamic analysis using Landauer's principle
to ground metabolic cost in information theory.

Spec Reference: §11 Thermodynamic Constraints
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402
from config import CONFIG  # noqa: E402
from core.thermodynamics import (  # noqa: E402
    compute_landauer_cost,
    compute_information_bits,
    estimate_temperature_from_cost,
)


def main():
    """Run thermodynamic analysis example."""

    print("=" * 60)
    print("APGI Thermodynamic Analysis Example")
    print("=" * 60)

    # Create configuration with thermodynamics
    print("\n1. Creating configuration with thermodynamics...")
    config = CONFIG.copy()
    config["use_thermodynamics"] = True
    print("   ✓ Configuration created")
    print(f"     - Boltzmann constant: {config['k_boltzmann']:.2e} J/K")
    print(
        f"     - Environment temperature: {config['T_env']:.1f} K ({config['T_env'] - 273:.1f}°C)"
    )
    print(f"     - Metabolic efficiency: {config['kappa_meta']:.2f}")

    # Initialize pipeline
    print("\n2. Initializing APGI pipeline...")
    pipeline = APGIPipeline(config)
    print("   ✓ Pipeline initialized")

    # Run simulation
    print("\n3. Running simulation for thermodynamic analysis...")
    n_steps = 500

    thermodynamic_data = {
        "S": [],
        "landauer_cost": [],
        "info_bits": [],
        "metabolic_cost": [],
        "efficiency": [],
        "temperature": [],
    }

    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)

        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)

        # Record signal
        S = result["S"]
        thermodynamic_data["S"].append(S)

        # Compute Landauer cost
        landauer = compute_landauer_cost(
            S=S,
            eps=config["eps"],
            k_b=config["k_boltzmann"],
            T_env=config["T_env"],
            kappa_meta=config["kappa_meta"],
        )
        thermodynamic_data["landauer_cost"].append(landauer)

        # Compute information bits
        bits = compute_information_bits(S, config["eps"])
        thermodynamic_data["info_bits"].append(bits)

        # Record metabolic cost
        thermodynamic_data["metabolic_cost"].append(result["C"])

        # Compute efficiency
        if result["C"] > 0:
            efficiency = landauer / result["C"]
            thermodynamic_data["efficiency"].append(efficiency)
        else:
            thermodynamic_data["efficiency"].append(0)

        # Compute implied temperature
        if S > config["eps"]:
            temp = estimate_temperature_from_cost(
                C_metabolic=result["C"],
                S=S,
                eps=config["eps"],
                k_b=config["k_boltzmann"],
                kappa_meta=config["kappa_meta"],
            )
            thermodynamic_data["temperature"].append(temp)
        else:
            thermodynamic_data["temperature"].append(config["T_env"])

    print(f"   ✓ Simulation completed ({n_steps} steps)")

    # Display Landauer's principle
    print("\n4. Landauer's Principle (Spec §11):")
    print("   Formula: E_min = κ_meta · N_erase · k_B · T_env · ln(2)")
    print("   where N_erase ≈ log₂(S / ε)")

    # Display thermodynamic results
    print("\n5. Thermodynamic Analysis Results:")

    S_array = np.array(thermodynamic_data["S"])
    landauer_array = np.array(thermodynamic_data["landauer_cost"])
    bits_array = np.array(thermodynamic_data["info_bits"])
    cost_array = np.array(thermodynamic_data["metabolic_cost"])
    efficiency_array = np.array(thermodynamic_data["efficiency"])
    temp_array = np.array(thermodynamic_data["temperature"])

    print("   Signal (S):")
    print(f"     Mean:  {np.mean(S_array):.4f}")
    print(f"     Std:   {np.std(S_array):.4f}")
    print(f"     Range: [{np.min(S_array):.4f}, {np.max(S_array):.4f}]")

    print("   Information bits (N_erase):")
    print(f"     Mean:  {np.mean(bits_array):.4f}")
    print(f"     Std:   {np.std(bits_array):.4f}")
    print(f"     Range: [{np.min(bits_array):.4f}, {np.max(bits_array):.4f}]")

    print("   Landauer cost (E_min):")
    print(f"     Mean:  {np.mean(landauer_array):.4e} J")
    print(f"     Std:   {np.std(landauer_array):.4e} J")
    print(f"     Range: [{np.min(landauer_array):.4e}, {np.max(landauer_array):.4e}] J")

    print("   Metabolic cost (C):")
    print(f"     Mean:  {np.mean(cost_array):.4f}")
    print(f"     Std:   {np.std(cost_array):.4f}")
    print(f"     Range: [{np.min(cost_array):.4f}, {np.max(cost_array):.4f}]")

    print("   Metabolic efficiency (C / E_min):")
    print(f"     Mean:  {np.mean(efficiency_array):.4f}")
    print(f"     Std:   {np.std(efficiency_array):.4f}")
    print(
        f"     Range: [{np.min(efficiency_array):.4f}, {np.max(efficiency_array):.4f}]"
    )

    print("   Implied temperature:")
    print(
        f"     Mean:  {np.mean(temp_array):.1f} K ({np.mean(temp_array) - 273:.1f}°C)"
    )
    print(f"     Std:   {np.std(temp_array):.1f} K")
    print(f"     Range: [{np.min(temp_array):.1f}, {np.max(temp_array):.1f}] K")

    # Validate thermodynamic constraint
    print("\n6. Thermodynamic Constraint Validation:")
    print("   Constraint: C(t) ≥ κ_meta · N_erase(t) · k_B · T_env · ln(2)")

    # Check constraint satisfaction
    constraint_satisfied = np.all(cost_array >= landauer_array)
    violations = np.sum(cost_array < landauer_array)

    print(f"   Constraint satisfied: {constraint_satisfied}")
    print(f"   Violations: {violations} / {n_steps}")
    print(f"   Satisfaction rate: {(1 - violations / n_steps) * 100:.1f}%")

    if constraint_satisfied:
        print("   ✓ Thermodynamic constraint satisfied")
    else:
        print("   ⚠ Some constraint violations detected")
        print("   (This may indicate suboptimal metabolic efficiency)")

    # Display energy statistics
    print("\n7. Energy Statistics:")
    total_landauer = np.sum(landauer_array)
    total_metabolic = np.sum(cost_array)

    print(f"   Total Landauer cost: {total_landauer:.4e} J")
    print(f"   Total metabolic cost: {total_metabolic:.4f}")
    print(f"   Average efficiency: {total_metabolic / total_landauer:.4f}")

    # Display information-theoretic analysis
    print("\n8. Information-Theoretic Analysis:")
    total_bits = np.sum(bits_array)
    avg_bits_per_step = total_bits / n_steps

    print(f"   Total information bits: {total_bits:.2f}")
    print(f"   Average bits per step: {avg_bits_per_step:.4f}")
    print(f"   Information rate: {avg_bits_per_step / config['dt']:.4f} bits/ms")

    # Display cost-benefit analysis
    print("\n9. Cost-Benefit Analysis:")

    # Find high-signal periods
    high_signal_mask = S_array > np.mean(S_array)
    low_signal_mask = S_array <= np.mean(S_array)

    if np.any(high_signal_mask):
        print("   High-signal periods (S > mean):")
        print(
            f"     Average Landauer cost: {np.mean(landauer_array[high_signal_mask]):.4e} J"
        )
        print(
            f"     Average metabolic cost: {np.mean(cost_array[high_signal_mask]):.4f}"
        )
        print(
            f"     Average efficiency: {np.mean(efficiency_array[high_signal_mask]):.4f}"
        )

    if np.any(low_signal_mask):
        print("   Low-signal periods (S ≤ mean):")
        print(
            f"     Average Landauer cost: {np.mean(landauer_array[low_signal_mask]):.4e} J"
        )
        print(
            f"     Average metabolic cost: {np.mean(cost_array[low_signal_mask]):.4f}"
        )
        print(
            f"     Average efficiency: {np.mean(efficiency_array[low_signal_mask]):.4f}"
        )

    print("\n" + "=" * 60)
    print("Thermodynamic analysis example completed successfully!")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("  1. See docs/API_REFERENCE.md for thermodynamic functions")
    print("  2. See APGI-Specs.md §11 for thermodynamic theory")
    print("  3. Adjust kappa_meta to explore efficiency tradeoffs")


if __name__ == "__main__":
    main()
