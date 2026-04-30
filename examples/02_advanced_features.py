#!/usr/bin/env python3
"""
APGI Advanced Features Example

This example demonstrates advanced features including:
- Kuramoto oscillators (§9)
- Observable mapping (§14)
- Stability analysis (§7)
- Hierarchical dynamics (§8)

Spec Reference: §9, §14, §7, §8
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402


def main() -> None:
    """Run advanced APGI example."""

    print("=" * 60)
    print("APGI Advanced Features Example")
    print("=" * 60)

    # Create configuration with advanced features
    print("\n1. Creating configuration with advanced features...")
    config = CONFIG.copy()
    config["use_kuramoto"] = True
    config["use_observable_mapping"] = True
    config["use_stability_analysis"] = True
    config["use_hierarchy"] = True
    config["n_levels"] = 3
    print("   ✓ Configuration created")
    print(f"     - Kuramoto oscillators: {config['use_kuramoto']}")
    print(f"     - Observable mapping: {config['use_observable_mapping']}")
    print(f"     - Stability analysis: {config['use_stability_analysis']}")
    print(f"     - Hierarchical system: {config['use_hierarchy']}")
    print(f"     - Hierarchical levels: {config['n_levels']}")

    # Initialize pipeline
    print("\n2. Initializing APGI pipeline with advanced features...")
    pipeline = APGIPipeline(config)
    print("   ✓ Pipeline initialized")

    # Run simulation
    print("\n3. Running simulation with advanced features...")
    n_steps = 200
    history: dict[str, list[float] | list[np.ndarray]] = {
        "S": [],
        "theta": [],
        "B": [],
        "kuramoto_phases": [],
        "kuramoto_sync": [],
        "neural_gamma": [],
        "behavioral_rt": [],
        "stability_eigs": [],
    }

    for t in range(n_steps):
        # Create input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 100)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 100)

        # Add neuromodulation
        g_ach = 1.0 + 0.2 * np.sin(2 * np.pi * t / 50)
        g_ne = 0.5 * np.cos(2 * np.pi * t / 50)

        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)
        # Apply manual neuromodulation to config before step if needed,
        # but the spec suggests these are inputs to the pipeline.
        # Since pipeline.step doesn't take them, we update config.
        pipeline.config["g_ach"] = g_ach
        pipeline.config["g_ne"] = g_ne

        # Record history
        history["S"].append(result["S"])
        history["theta"].append(result["theta"])
        history["B"].append(result["B"])

        # Kuramoto oscillators
        if "kuramoto_phases" in result:
            history["kuramoto_phases"].append(result["kuramoto_phases"])
            history["kuramoto_sync"].append(result["kuramoto_synchronization"])

        # Observable mapping
        if "neural_gamma_power" in result:
            history["neural_gamma"].append(result["neural_gamma_power"])
        if "behavioral_rt_variability" in result:
            history["behavioral_rt"].append(result["behavioral_rt_variability"])

        # Stability analysis
        if "stability_eigenvalues" in result:
            history["stability_eigs"].append(result["stability_eigenvalues"])

    print(f"   ✓ Simulation completed ({n_steps} steps)")

    # Display Kuramoto results
    if history["kuramoto_phases"]:
        print("\n4. Kuramoto Oscillators Results:")
        phases = np.array(history["kuramoto_phases"])
        sync = np.array(history["kuramoto_sync"])
        print(f"   Phase range: [{np.min(phases):.4f}, {np.max(phases):.4f}]")
        print(f"   Synchronization order (mean): {np.mean(sync):.4f}")
        print(f"   Synchronization order (max):  {np.max(sync):.4f}")
        print(f"   Synchronization order (min):  {np.min(sync):.4f}")

    # Display observable mapping results
    if history["neural_gamma"]:
        print("\n5. Observable Mapping Results:")
        gamma = np.array(history["neural_gamma"])
        rt_var = np.array(history["behavioral_rt"])
        print(f"   Neural gamma power (mean): {np.mean(gamma):.4f}")
        print(f"   Neural gamma power (std):  {np.std(gamma):.4f}")
        print(f"   Behavioral RT variability (mean): {np.mean(rt_var):.4f}")
        print(f"   Behavioral RT variability (std):  {np.std(rt_var):.4f}")

    # Display stability analysis results
    if history["stability_eigs"]:
        print("\n6. Stability Analysis Results:")
        eigs = np.array(history["stability_eigs"])
        eig_mags = np.abs(eigs)
        print(f"   Eigenvalue magnitudes (mean): {np.mean(eig_mags):.4f}")
        print(f"   Eigenvalue magnitudes (max):  {np.max(eig_mags):.4f}")
        print(f"   Eigenvalue magnitudes (min):  {np.min(eig_mags):.4f}")

        # Check stability
        stable = np.all(eig_mags < 1.0)
        print(f"   System stable: {stable}")

    # Display hierarchical results
    print("\n7. Hierarchical System Results:")
    print(f"   Number of levels: {config['n_levels']}")
    print(f"   Timescale ratio (k): {config['timescale_k']}")

    # Calculate timescales
    tau_0 = float(config["tau_s"])  # type: ignore[arg-type]
    k = float(config["timescale_k"])  # type: ignore[arg-type]
    n_levels: int = config["n_levels"]  # type: ignore[assignment]
    for level in range(n_levels):
        tau_l = tau_0 * (k**level)
        print(f"   Level {level}: τ = {tau_l:.2f} ms")

    # Display overall statistics
    print("\n8. Overall Statistics:")
    print("   Signal (S):")
    print(f"     Mean:  {np.mean(history['S']):.4f}")
    print(f"     Std:   {np.std(history['S']):.4f}")
    print("   Threshold (θ):")
    print(f"     Mean:  {np.mean(history['theta']):.4f}")
    print(f"     Std:   {np.std(history['theta']):.4f}")
    print(f"   Ignition events: {sum(history['B'])} / {n_steps}")
    print(f"   Ignition rate:   {sum(history['B']) / n_steps * 100:.1f}%")

    print("\n" + "=" * 60)
    print("Advanced features example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
