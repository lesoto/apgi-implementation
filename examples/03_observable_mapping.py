#!/usr/bin/env python3
"""
APGI Observable Mapping Example

This example demonstrates observable mapping, which connects internal
APGI variables to neural and behavioral observables.

Spec Reference: §14 Observable Mapping
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402
from config import CONFIG  # noqa: E402


def main():
    """Run observable mapping example."""

    print("=" * 60)
    print("APGI Observable Mapping Example")
    print("=" * 60)

    # Create configuration with observable mapping
    print("\n1. Creating configuration with observable mapping...")
    config = CONFIG.copy()
    config["use_observable_mapping"] = True
    print("   ✓ Configuration created")

    # Initialize pipeline
    print("\n2. Initializing APGI pipeline...")
    pipeline = APGIPipeline(config)
    print("   ✓ Pipeline initialized")

    # Run simulation
    print("\n3. Running simulation for observable extraction...")
    n_steps = 500

    # Storage for observables
    neural_observables = {"gamma_power": [], "erp_amplitude": [], "ignition_rate": []}

    behavioral_observables = {
        "rt_variability": [],
        "response_criterion": [],
        "decision_rate": [],
    }

    internal_variables = {"S": [], "theta": [], "B": [], "delta": []}

    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)

        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)

        # Record internal variables
        internal_variables["S"].append(result["S"])
        internal_variables["theta"].append(result["theta"])
        internal_variables["B"].append(result["B"])
        if "prediction_margin" in result:
            internal_variables["delta"].append(result["prediction_margin"])
        else:
            # Fallback if prediction_validator is not active
            internal_variables["delta"].append(result["S"] - result["theta"])

        # Record neural observables
        if "neural_gamma_power" in result:
            neural_observables["gamma_power"].append(result["neural_gamma_power"])
        if "neural_erp_amplitude" in result:
            neural_observables["erp_amplitude"].append(result["neural_erp_amplitude"])
        if "neural_ignition_rate" in result:
            neural_observables["ignition_rate"].append(result["neural_ignition_rate"])

        # Record behavioral observables
        if "behavioral_rt_variability" in result:
            behavioral_observables["rt_variability"].append(
                result["behavioral_rt_variability"]
            )
        if "behavioral_response_criterion" in result:
            behavioral_observables["response_criterion"].append(
                result["behavioral_response_criterion"]
            )
        if "behavioral_decision_rate" in result:
            behavioral_observables["decision_rate"].append(
                result["behavioral_decision_rate"]
            )

    print(f"   ✓ Simulation completed ({n_steps} steps)")

    # Display observable mapping table
    print("\n4. Observable Mapping (Spec §14):")
    print("   " + "-" * 56)
    print("   Internal Variable | Neural Observable | Behavioral Observable")
    print("   " + "-" * 56)
    print("   S(t)              | Gamma power       | Perceptual sensitivity")
    print("   θ(t)              | P300/N200 ERP     | RT variability")
    print("   B(t)              | Global ignition   | Overt decision")
    print("   " + "-" * 56)

    # Display neural observables
    print("\n5. Neural Observables:")
    if neural_observables["gamma_power"]:
        gamma = np.array(neural_observables["gamma_power"])
        print("   Gamma-band power (30-100 Hz):")
        print(f"     Mean:  {np.mean(gamma):.4f}")
        print(f"     Std:   {np.std(gamma):.4f}")
        print(f"     Min:   {np.min(gamma):.4f}")
        print(f"     Max:   {np.max(gamma):.4f}")

    if neural_observables["erp_amplitude"]:
        erp = np.array(neural_observables["erp_amplitude"])
        print("   P300/N200 ERP amplitude:")
        print(f"     Mean:  {np.mean(erp):.4f}")
        print(f"     Std:   {np.std(erp):.4f}")
        print(f"     Min:   {np.min(erp):.4f}")
        print(f"     Max:   {np.max(erp):.4f}")

    if neural_observables["ignition_rate"]:
        ign = np.array(neural_observables["ignition_rate"])
        print("   Global ignition (gamma synchrony):")
        print(f"     Mean:  {np.mean(ign):.4f}")
        print(f"     Std:   {np.std(ign):.4f}")
        print(f"     Min:   {np.min(ign):.4f}")
        print(f"     Max:   {np.max(ign):.4f}")

    # Display behavioral observables
    print("\n6. Behavioral Observables:")
    if behavioral_observables["rt_variability"]:
        rt = np.array(behavioral_observables["rt_variability"])
        print("   RT variability:")
        print(f"     Mean:  {np.mean(rt):.4f}")
        print(f"     Std:   {np.std(rt):.4f}")
        print(f"     Min:   {np.min(rt):.4f}")
        print(f"     Max:   {np.max(rt):.4f}")

    if behavioral_observables["response_criterion"]:
        crit = np.array(behavioral_observables["response_criterion"])
        print("   Response criterion (conservatism):")
        print(f"     Mean:  {np.mean(crit):.4f}")
        print(f"     Std:   {np.std(crit):.4f}")
        print(f"     Min:   {np.min(crit):.4f}")
        print(f"     Max:   {np.max(crit):.4f}")

    if behavioral_observables["decision_rate"]:
        dec = np.array(behavioral_observables["decision_rate"])
        print("   Decision rate:")
        print(f"     Mean:  {np.mean(dec):.4f}")
        print(f"     Std:   {np.std(dec):.4f}")
        print(f"     Min:   {np.min(dec):.4f}")
        print(f"     Max:   {np.max(dec):.4f}")

    # Display internal variables
    print("\n7. Internal Variables:")
    S = np.array(internal_variables["S"])
    theta = np.array(internal_variables["theta"])
    B = np.array(internal_variables["B"])
    delta = np.array(internal_variables["delta"])

    print("   Signal (S):")
    print(f"     Mean:  {np.mean(S):.4f}")
    print(f"     Std:   {np.std(S):.4f}")

    print("   Threshold (θ):")
    print(f"     Mean:  {np.mean(theta):.4f}")
    print(f"     Std:   {np.std(theta):.4f}")

    print("   Ignition margin (Δ = S - θ):")
    print(f"     Mean:  {np.mean(delta):.4f}")
    print(f"     Std:   {np.std(delta):.4f}")

    print(f"   Ignition events: {sum(B)} / {n_steps}")
    print(f"   Ignition rate:   {sum(B) / n_steps * 100:.1f}%")

    # Validate key testable prediction
    print("\n8. Key Testable Prediction (Spec §14):")
    print("   Prediction: Margin Δ(t) = S(t) - θ(t) outperforms S(t) alone")

    # Compute correlations with ignition
    if len(B) > 1:
        # Correlation of signal with ignition
        corr_S = np.corrcoef(S, B)[0, 1]

        # Correlation of margin with ignition
        corr_delta = np.corrcoef(delta, B)[0, 1]

        print(f"   Correlation(S, B):     {corr_S:.4f}")
        print(f"   Correlation(Δ, B):     {corr_delta:.4f}")
        print(f"   Improvement:           {corr_delta - corr_S:.4f}")

        if corr_delta > corr_S:
            print("   ✓ Prediction validated: Margin outperforms signal")
        else:
            print("   ✗ Prediction not validated in this simulation")

    print("\n" + "=" * 60)
    print("Observable mapping example completed successfully!")
    print("=" * 60)

    # Print next steps
    print("\nNext steps:")
    print("  1. Try 04_thermodynamics.py for thermodynamic analysis")
    print("  2. See docs/API_REFERENCE.md for observable extraction API")
    print("  3. See APGI-Specs.md §14 for observable mapping theory")


if __name__ == "__main__":
    main()
