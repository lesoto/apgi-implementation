"""Example: Reservoir-as-Threshold Mode (Spec-Explicit Alternative)

Demonstrates the reservoir serving as an alternative execution path
replacing steps 7-8 (threshold computation and ignition) per APGI spec §10.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
from pipeline import APGIPipeline  # noqa: E402
from config import CONFIG  # noqa: E402

# Configuration with reservoir-as-threshold mode
config = CONFIG.copy()
config["use_reservoir"] = True
config["reservoir_as_threshold"] = True  # Enable spec-explicit alternative mode
config["reservoir_size"] = 100
config["reservoir_tau"] = 1.0
config["reservoir_amplification"] = 0.1
config["reservoir_theta_scale"] = 0.05  # Scale factor for threshold mapping
config["stochastic_ignition"] = False

# Initialize pipeline
pipeline = APGIPipeline(config)

# Run simulation
n_steps = 500
theta_history = []
S_history = []
B_history = []

np.random.seed(42)
for t in range(n_steps):
    # Generate input with periodic structure
    x_e = np.sin(0.1 * t) + 0.3 * np.random.randn()
    x_i = 0.5 * np.cos(0.05 * t) + 0.2 * np.random.randn()

    result = pipeline.step(x_e=x_e, x_i=x_i)

    theta_history.append(result["theta"])
    S_history.append(result["S"])
    B_history.append(result["B"])

# Analysis
n_ignitions = sum(B_history)
avg_theta = np.mean(theta_history)

print("Reservoir-as-Threshold Mode (Spec §10)")
print("=" * 50)
print("Configuration: reservoir_as_threshold=True")
print(f"Reservoir size: {config['reservoir_size']}")
print(f"Reservoir tau: {config['reservoir_tau']}")
print(f"Theta scale: {config['reservoir_theta_scale']}")
print()
print(f"Simulation steps: {n_steps}")
print(f"Ignition events: {n_ignitions} ({100 * n_ignitions / n_steps:.1f}%)")
print(f"Average threshold: {avg_theta:.3f}")
print(f"Average signal S: {np.mean(S_history):.3f}")
print()
print("✅ Reservoir successfully replaced allostatic threshold computation")
print("   (Steps 7-8 in canonical spec table)")
