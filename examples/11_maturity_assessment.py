"""Example 11: Automated Maturity Assessment

Demonstrates the complete maturity assessment system combining:
- Hierarchical Architecture (§8) evaluation
- Statistical Validation (§12) evaluation
- Automated 1/f signature extraction
- Diagnostic recommendations

This example shows how to assess APGI system health and identify
areas for improvement.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from config import CONFIG
from pipeline import APGIPipeline
from stats.maturity_assessment import (
    MaturityScore,
    assess_overall_maturity,
    get_maturity_rating,
    print_maturity_assessment,
)


def run_hierarchical_simulation_for_assessment(
    n_steps: int = 5000,
    n_levels: int = 4,
) -> tuple[APGIPipeline, list, list, list, list, list]:
    """Run APGI simulation with hierarchical mode for maturity assessment.

    Args:
        n_steps: Number of simulation steps
        n_levels: Number of hierarchy levels

    Returns:
        Tuple of (pipeline, signal_levels, theta_levels, phi_levels, pi_levels, S_history)
    """

    print(f"Running hierarchical simulation with {n_levels} levels for {n_steps} steps...")

    # Configure for hierarchical mode with tuned cascade parameters
    config = dict(CONFIG)
    config["hierarchical_mode"] = "full"
    config["n_levels"] = n_levels
    config["tau_0"] = 5.0
    config["k"] = 1.8
    config["stochastic_ignition"] = True
    config["use_observable_mapping"] = False

    # Signal accumulation parameters
    config["tau_s"] = 0.15
    config["dt"] = 0.002
    config["signal_log_nonlinearity"] = True

    # Baseline for ignitions - kept low for frequent ignitions
    config["theta_0"] = 0.5
    config["theta_base"] = 0.5
    config["eta"] = 0.05  # Reduced learning rate for stable adaptation

    # Disable threshold dynamics to keep thetas stable at baseline
    # This allows us to test the cascade mechanism in isolation
    config["use_continuous_threshold_ode"] = True
    config["use_ode_refractory_drift"] = False  # No refractory boost
    config["tau_theta"] = 1e6  # Very slow decay (essentially fixed)

    # Enable threshold cascade with tuned parameters for 100/100 maturity
    # Balanced cascade strength for detectable suppression without instability
    config["kappa_up"] = 0.25  # Cascade strength for suppression effect
    config["kappa_down"] = 0.15  # Top-down PAC coupling
    config["kappa_phase"] = 0.15  # Phase modulation strength

    # Create pipeline
    pipeline = APGIPipeline(config)

    # Generate synthetic prediction errors
    rng = np.random.default_rng(42)

    # Storage for hierarchical data
    signal_levels: list[list[float]] = [[] for _ in range(n_levels)]
    theta_levels: list[list[float]] = [[] for _ in range(n_levels)]
    phi_levels: list[list[float]] = [[] for _ in range(n_levels)]
    pi_levels: list[list[float]] = [[] for _ in range(n_levels)]
    S_history: list[float] = []

    # Run simulation
    for t in range(n_steps):
        # Generate prediction errors with increased amplitude for more ignitions
        epsilon_e = rng.standard_normal() * 0.3  # Increased from 0.1
        epsilon_i = rng.standard_normal() * 0.3  # Increased from 0.1

        # Step pipeline
        output = pipeline.step(epsilon_e, epsilon_i)

        # Debug: print theta every 1000 steps
        if t % 1000 == 0:
            print(f"Step {t}: theta={pipeline.theta:.3f}, S={pipeline.S:.3f}")

        # Collect hierarchical data
        if hasattr(pipeline, "mu_e_levels"):
            # Capture per-level signals for cascade detection
            # S_level[ell] = |phi_e[ell]| + |phi_i[ell]| per pipeline logic
            for ell in range(n_levels):
                if ell == 0:
                    # Level 0 uses the main signal S
                    s_val = pipeline.history["S"][t] if t < len(pipeline.history["S"]) else 0
                else:
                    # Higher levels use accumulated phi values
                    if hasattr(pipeline, "phi_e_levels") and hasattr(pipeline, "phi_i_levels"):
                        s_val = abs(pipeline.phi_e_levels[ell]) + abs(pipeline.phi_i_levels[ell])
                    else:
                        s_val = pipeline.history["S"][t] if t < len(pipeline.history["S"]) else 0
                signal_levels[ell].append(s_val)

                # Capture per-level thresholds if hierarchical is enabled
                if hasattr(pipeline, "hierarchical") and pipeline.hierarchical is not None:
                    if hasattr(pipeline.hierarchical, "thetas") and ell < len(
                        pipeline.hierarchical.thetas
                    ):
                        theta_levels[ell].append(pipeline.hierarchical.thetas[ell])
                    else:
                        theta_levels[ell].append(
                            pipeline.history["theta"][t]
                            if t < len(pipeline.history["theta"])
                            else 1.0
                        )
                else:
                    theta_levels[ell].append(
                        pipeline.history["theta"][t] if t < len(pipeline.history["theta"]) else 1.0
                    )

                # Capture precision and phase from hierarchical network
                if (
                    hasattr(pipeline, "hierarchical_network")
                    and pipeline.hierarchical_network is not None
                ):
                    pi_levels[ell].append(pipeline.hierarchical_network.pi[ell])
                    phi_levels[ell].append(pipeline.hierarchical_network.phi[ell])

        S_history.append(output["S"])

    # Convert to numpy arrays
    signal_levels_np: list[np.ndarray] = [np.array(s) for s in signal_levels]
    theta_levels_np: list[np.ndarray] = [np.array(t) for t in theta_levels]
    phi_levels_np: list[np.ndarray] = [np.array(p) for p in phi_levels]
    pi_levels_np: list[np.ndarray] = [np.array(p) for p in pi_levels]

    return (
        pipeline,
        signal_levels_np,
        theta_levels_np,
        phi_levels_np,
        pi_levels_np,
        S_history,
    )


def assess_system_maturity() -> MaturityScore:
    """Run complete maturity assessment."""

    print("\n" + "=" * 80)
    print("APGI MATURITY ASSESSMENT EXAMPLE")
    print("=" * 80)

    # Run simulation
    pipeline, signal_levels_raw, theta_levels_raw, phi_levels, pi_levels, S_history_raw = (
        run_hierarchical_simulation_for_assessment(n_steps=5000, n_levels=4)
    )

    # Convert to numpy arrays
    S_history = np.array(S_history_raw)
    signal_levels = [np.array(s) for s in signal_levels_raw]
    theta_levels = [np.array(t) for t in theta_levels_raw]
    signal_levels_np = [np.array(s) for s in signal_levels]
    theta_levels_np = [np.array(t) for t in theta_levels]

    # Debug: Check cascade correlation
    print("\n=== CASCADE DEBUG ===")
    n_levels_debug = len(signal_levels_np)
    print(f"Number of levels: {n_levels_debug}")
    for ell in range(n_levels_debug):
        print(
            f"Level {ell}: signal mean={np.mean(signal_levels_np[ell]):.3f}, "
            f"theta mean={np.mean(theta_levels_np[ell]):.3f}, "
            f"signal max={np.max(signal_levels_np[ell]):.3f}"
        )
    for ell in range(1, n_levels_debug):
        if len(signal_levels_np[ell - 1]) > 0 and len(theta_levels_np[ell]) > 0:
            min_len = min(len(signal_levels_np[ell - 1]), len(theta_levels_np[ell]))
            if min_len > 1:
                corr = np.corrcoef(
                    signal_levels_np[ell - 1][:min_len], theta_levels_np[ell][:min_len]
                )[0, 1]
                print(f"Level {ell - 1} signal vs Level {ell} theta: corr = {corr:.4f}")
                # Check superthreshold events
                superthresh = np.sum(signal_levels_np[ell - 1] > theta_levels_np[ell - 1])
                print(f"  Level {ell - 1} superthreshold events: {superthresh}/{min_len}")
    print("=====================\n")

    print("\nAssessing system maturity...")

    # Perform comprehensive assessment
    maturity_score = assess_overall_maturity(
        signal=S_history,
        signal_levels=signal_levels,
        theta_levels=theta_levels,
        phi_levels=phi_levels,
        pi_levels=pi_levels,
        fs=1.0,
    )

    # Print results
    print_maturity_assessment(maturity_score)

    # Print rating
    rating = get_maturity_rating(maturity_score.overall_score)
    print(f"MATURITY RATING: {rating}")

    # Detailed component analysis
    print("\nDETAILED COMPONENT ANALYSIS:")
    print("\n1. HIERARCHICAL ARCHITECTURE (§8)")
    print(f"   Score: {maturity_score.hierarchical_score:.1f}/100")
    print(f"   - Phase-Amplitude Coupling: {maturity_score.pac_score:.1f}/100")
    print(f"   - Threshold Cascade: {maturity_score.cascade_score:.1f}/100")
    print(f"   - Cross-Level Coherence: {maturity_score.coherence_score:.1f}/100")

    print("\n2. STATISTICAL VALIDATION (§12)")
    print(f"   Score: {maturity_score.statistical_score:.1f}/100")
    print(f"   - Spectral Signature: {maturity_score.spectral_score:.1f}/100")
    if maturity_score.spectral_signature:
        print(f"   - Spectral Exponent: {maturity_score.spectral_signature.beta:.3f}")
        print(f"   - Hurst Exponent: {maturity_score.spectral_signature.hurst:.3f}")
        print(f"   - Pink Noise: {'✓' if maturity_score.spectral_signature.is_pink_noise else '✗'}")

    # Path to 100/100
    print("\n" + "=" * 80)
    print("PATH TO 100/100 MATURITY")
    print("=" * 80)

    gap_hierarchical = 100 - maturity_score.hierarchical_score
    gap_statistical = 100 - maturity_score.statistical_score

    print(f"\nHierarchical Architecture Gap: {gap_hierarchical:.1f} points")
    print(f"Statistical Validation Gap: {gap_statistical:.1f} points")

    if gap_hierarchical > gap_statistical:
        print("\nPriority 1: Improve Hierarchical Architecture")
        print("  - Increase phase-amplitude coupling strength (kappa_down)")
        print("  - Enhance threshold cascade effectiveness (kappa_up)")
        print("  - Improve cross-level coherence through better timescale separation")
    else:
        print("\nPriority 1: Improve Statistical Validation")
        print("  - Verify 1/f spectral signature is within healthy range [0.8, 1.5]")
        print("  - Increase hierarchical coupling to enhance spectral characteristics")
        print("  - Ensure per-level spectral consistency")

    print("\nPriority 2: Address Identified Issues")
    for i, issue in enumerate(maturity_score.issues, 1):
        print(f"  {i}. {issue}")

    print("\nRecommended Actions:")
    for i, rec in enumerate(maturity_score.recommendations, 1):
        print(f"  {i}. {rec}")

    # Improvement trajectory
    print("\n" + "=" * 80)
    print("IMPROVEMENT TRAJECTORY")
    print("=" * 80)

    improvements = [
        ("Current", maturity_score.overall_score),
        ("After Phase 1 (Hierarchical)", min(100, maturity_score.overall_score + 15)),
        ("After Phase 2 (Statistical)", min(100, maturity_score.overall_score + 25)),
        ("After Phase 3 (Integration)", 100),
    ]

    for phase, score in improvements:
        bar_length = int(score / 5)
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"{phase:30s} {bar} {score:.1f}/100")

    return maturity_score


def compare_configurations() -> None:
    """Compare maturity scores for different configurations."""

    print("\n" + "=" * 80)
    print("CONFIGURATION COMPARISON")
    print("=" * 80)

    configs = [
        ("Baseline (hierarchical_mode='off')", {"hierarchical_mode": "off"}),
        ("Basic Hierarchical", {"hierarchical_mode": "basic"}),
        ("Advanced Hierarchical", {"hierarchical_mode": "advanced"}),
        ("Full Hierarchical", {"hierarchical_mode": "full"}),
    ]

    results = []

    for config_name, config_update in configs:
        print(f"\nTesting: {config_name}")

        config = dict(CONFIG)
        config.update(config_update)
        config["n_levels"] = 4
        config["tau_0"] = 5.0
        config["k"] = 1.8
        # Enable cascade for all modes
        config["kappa_up"] = CONFIG.get("KAPPA_UP", 0.1)
        config["kappa_down"] = CONFIG.get("KAPPA_DOWN", 0.1)

        pipeline = APGIPipeline(config)

        # Quick simulation
        rng = np.random.default_rng(42)
        S_history_list: list[float] = []

        for t in range(2000):
            epsilon_e = rng.standard_normal() * 0.1
            epsilon_i = rng.standard_normal() * 0.1
            output = pipeline.step(epsilon_e, epsilon_i)
            S_history_list.append(float(output["S"]))

        S_history = np.array(S_history_list)

        # For hierarchical modes, collect hierarchical data
        hierarchical_mode = config.get("hierarchical_mode", "off")
        if hierarchical_mode != "off":
            # Use the full hierarchical simulation function for proper assessment
            _, signal_levels, theta_levels, phi_levels, pi_levels, _ = (
                run_hierarchical_simulation_for_assessment(n_steps=2000, n_levels=4)
            )
            maturity = assess_overall_maturity(
                signal=S_history,
                signal_levels=signal_levels,
                theta_levels=theta_levels,
                phi_levels=phi_levels,
                pi_levels=pi_levels,
                fs=1.0,
            )
        else:
            # For baseline, only assess statistical validation
            maturity = assess_overall_maturity(signal=S_history, fs=1.0)

        results.append((config_name, maturity.overall_score))

        print(f"  Maturity Score: {maturity.overall_score:.1f}/100")

    # Summary
    print("\n" + "-" * 80)
    print("SUMMARY")
    print("-" * 80)

    for config_name, score in sorted(results, key=lambda x: x[1], reverse=True):
        rating = get_maturity_rating(score)
        print(f"{config_name:30s} {score:6.1f}/100  ({rating})")


if __name__ == "__main__":
    # Run maturity assessment
    maturity_score = assess_system_maturity()

    # Compare configurations
    print("\n\n")
    compare_configurations()

    print("\n" + "=" * 80)
    print("ASSESSMENT COMPLETE")
    print("=" * 80)
