#!/usr/bin/env python3
"""
APGI Basic Usage Example

This example demonstrates the basic usage of the APGI pipeline.

Spec Reference: §13 Execution Pipeline
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from config import CONFIG  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402


def main():
    """Run basic APGI example."""

    print("=" * 60)
    print("APGI Basic Usage Example")
    print("=" * 60)

    # Initialize pipeline with default configuration
    print("\n1. Initializing APGI pipeline...")
    pipeline = APGIPipeline(CONFIG)
    print("   ✓ Pipeline initialized")

    # Run a single step
    print("\n2. Running single step...")
    result = pipeline.step(
        x_e=0.5,  # Exteroceptive signal
        x_hat_e=0.3,  # Exteroceptive prediction
        x_i=0.2,  # Interoceptive signal
        x_hat_i=0.1,  # Interoceptive prediction
    )
    print("   ✓ Step executed")

    # Display results
    print("\n3. Results from single step:")
    print(f"   Signal (S):           {result['S']:.4f}")
    print(f"   Threshold (θ):        {result['theta']:.4f}")
    print(f"   Ignition (B):         {result['B']}")
    print(f"   Ignition prob (P):    {result['p_ignite']:.4f}")
    print(f"   Metabolic cost (C):   {result['C']:.4f}")
    print(f"   Information value (V):{result['V']:.4f}")

    # Run simulation with varying inputs
    print("\n4. Running simulation with varying inputs...")
    n_steps = 100
    history = {"S": [], "theta": [], "B": [], "C": [], "V": [], "p_ignite": []}

    for t in range(n_steps):
        # Create sinusoidal input signals
        x_e = 0.5 + 0.2 * np.sin(2 * np.pi * t / 50)
        x_i = 0.2 + 0.1 * np.cos(2 * np.pi * t / 50)

        # Execute step
        result = pipeline.step(x_e=x_e, x_hat_e=0.3, x_i=x_i, x_hat_i=0.1)

        # Record history
        for key in ["S", "theta", "B", "C", "V", "p_ignite"]:
            if key in result:
                history[key].append(result[key])

    print(f"   ✓ Simulation completed ({n_steps} steps)")

    # Display statistics
    print("\n5. Simulation statistics:")
    print("   Signal (S):")
    print(f"     Mean:  {np.mean(history['S']):.4f}")
    print(f"     Std:   {np.std(history['S']):.4f}")
    print(f"     Min:   {np.min(history['S']):.4f}")
    print(f"     Max:   {np.max(history['S']):.4f}")

    print("   Threshold (θ):")
    print(f"     Mean:  {np.mean(history['theta']):.4f}")
    print(f"     Std:   {np.std(history['theta']):.4f}")
    print(f"     Min:   {np.min(history['theta']):.4f}")
    print(f"     Max:   {np.max(history['theta']):.4f}")

    print(f"   Ignition events: {sum(history['B'])} / {n_steps}")
    print(f"   Ignition rate:   {sum(history['B']) / n_steps * 100:.1f}%")

    # Verify final state
    print("\n6. Final pipeline state:")
    result = pipeline.step(0.5, 0.3, 0.2, 0.1)
    print(f"   Signal: {result['S']:.4f}")
    print(f"   Threshold: {result['theta']:.4f}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
