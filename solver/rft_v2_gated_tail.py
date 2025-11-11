#!/usr/bin/env python3
"""
RFT v2: Kernel + Acceleration-Gated Tail

Extends RFT v1 kernel with a tail term that activates where baryons are weak.
Designed to break the g_res ∝ g_b coupling that caused SPARC RED gate.

Physics:
    g_total = g_bar + g_kernel + g_tail

    g_tail(r) = A_0 (r_geo/r)^α × gate_r(r) × gate_g(g_bar)

    gate_r(r) = tanh[(r - r_min) / Δr]  (don't activate too early)
    gate_g(g_bar) = 1 - tanh[(g_bar - g_min) / Δg]  (only where baryons weak)

Key idea: Tail activates based on acceleration deficit, not baryonic density.

Author: RFT Cosmology Project
Date: 2025-11-10
"""

import numpy as np
from typing import Dict, Optional
from sparc_rft.case import GalaxyCase
from solver.rft_kernel import apply_kernel as apply_kernel_v1, _geometric_mean_radius


def apply_v2_gated_tail(
    case: GalaxyCase,
    kernel_config: Dict,
    tail_config: Dict,
    disable_modes: Optional[list] = None,
) -> Dict:
    """
    Apply RFT v2: kernel + acceleration-gated tail.

    Args:
        case: Galaxy case with r_kpc, v_obs_kms, v_baryon_*
        kernel_config: Dict with kernel parameters (same as v1)
        tail_config: Dict with tail parameters:
            - A_0: amplitude [m/s²]
            - alpha: power-law index
            - r_geo: geometric scale [kpc]
            - r_min: minimum radius for activation [kpc]
            - Delta_r: radial gate transition width [kpc]
            - g_min: minimum baryonic acceleration [m/s²]
            - Delta_g: acceleration gate transition width [m/s²]
        disable_modes: Optional list of modes to disable (e.g., ["kernel", "tail"])

    Returns:
        Dict with keys:
            - r_kpc: radii
            - v_pred_kms: predicted velocities
            - g_components: {"kernel": g_kernel, "tail": g_tail}
            - descriptors: {"r_geo": ..., "tail_params": ...}
    """
    disable_modes = disable_modes or []

    # 1. Get kernel contribution (v1)
    if "kernel" in disable_modes:
        # Newtonian only
        r = np.asarray(case.r_kpc, dtype=float)
        v_disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
        v_gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
        v_bulge = getattr(case, "v_baryon_bulge_kms", None)
        if v_bulge is None:
            v_bulge = np.zeros_like(r)
        else:
            v_bulge = np.asarray(v_bulge, dtype=float)

        v_b_sq = v_disk**2 + v_gas**2 + v_bulge**2
        v_b = np.sqrt(v_b_sq)
        g_b = v_b_sq / np.clip(r, 1e-6, None)  # km²/s² / kpc
        g_kernel = np.zeros_like(r)
        r_geo = _geometric_mean_radius(r)
    else:
        kernel_result = apply_kernel_v1(case, kernel_config, disable_modes=disable_modes)
        r = kernel_result["r_kpc"]

        # Handle bulge (may be None or list)
        v_bulge = getattr(case, "v_baryon_bulge_kms", None)
        if v_bulge is None:
            v_bulge = np.zeros_like(r)
        else:
            v_bulge = np.asarray(v_bulge, dtype=float)

        v_b = np.sqrt(
            np.asarray(case.v_baryon_disk_kms, dtype=float) ** 2
            + np.asarray(case.v_baryon_gas_kms, dtype=float) ** 2
            + v_bulge ** 2
        )
        g_b = v_b**2 / np.clip(r, 1e-6, None)
        g_kernel = kernel_result["g_components"]["kernel"]
        r_geo = kernel_result["descriptors"]["r_geo"]

    # 2. Compute tail contribution
    if "tail" in disable_modes:
        g_tail = np.zeros_like(r)
    else:
        g_tail = _compute_tail_acceleration(r, g_b, r_geo, tail_config)

    # 3. Total velocity
    # All accelerations now in km²/s²/kpc (consistent units)
    # g_b, g_kernel, g_tail all in km²/s²/kpc
    # v² = r × g → v [km/s] = sqrt(r [kpc] × g [km²/s²/kpc])

    v_b_sq = v_b**2
    v_pred_sq = v_b_sq + r * (g_kernel + g_tail)
    v_pred = np.sqrt(np.clip(v_pred_sq, 0.0, None))

    return {
        "r_kpc": r,
        "v_pred_kms": v_pred,
        "g_components": {
            "kernel": g_kernel,  # km²/s²/kpc
            "tail": g_tail,  # km²/s²/kpc
        },
        "descriptors": {
            "r_geo": r_geo,
            "tail_params": tail_config,
        },
    }


def _compute_tail_acceleration(
    r_kpc: np.ndarray, g_bar_kms2_kpc: np.ndarray, r_geo: float, tail_config: Dict
) -> np.ndarray:
    """
    Compute acceleration-gated tail (GPT rational gate form).

    Formula:
        g_tail = A0 * (r_geo/r)^α * [1 + (gb/g*)^γ]^(-1) * [1 - exp(-(r/r_turn)^p)]

    Args:
        r_kpc: Radii [kpc]
        g_bar_kms2_kpc: Baryonic acceleration [km²/s²/kpc]
        r_geo: Geometric scale radius [kpc]
        tail_config: Tail parameters:
            - A0_kms2_per_kpc: amplitude [km²/s²/kpc]
            - alpha: power-law index
            - g_star_kms2_per_kpc: acceleration scale [km²/s²/kpc]
            - gamma: acceleration gate sharpness
            - r_turn_kpc: onset radius [kpc]
            - p: onset sharpness

    Returns:
        g_tail: Tail acceleration [km²/s²/kpc]
    """
    A_0 = tail_config["A0_kms2_per_kpc"]  # km²/s²/kpc
    alpha = tail_config.get("alpha", 1.0)
    g_star = tail_config["g_star_kms2_per_kpc"]  # km²/s²/kpc
    gamma = tail_config.get("gamma", 1.0)
    r_turn = tail_config.get("r_turn_kpc", 2.0)  # kpc
    p = tail_config.get("p", 2.0)

    # Safe clips
    r_safe = np.clip(r_kpc, 1e-6, None)
    r_geo_safe = np.clip(r_geo, 1e-6, None)
    g_bar_safe = np.clip(g_bar_kms2_kpc, 1e-12, None)
    g_star = max(g_star, 1e-12)

    # Power-law base
    base = A_0 * (r_geo_safe / r_safe) ** alpha

    # Acceleration gate (rational form - smooth, no discontinuity)
    gate_accel = 1.0 / (1.0 + (g_bar_safe / g_star) ** gamma)

    # Onset gate (exponential - suppresses inner disk)
    gate_onset = 1.0 - np.exp(-(r_safe / r_turn) ** p) if r_turn > 0 else 1.0

    g_tail = base * gate_accel * gate_onset

    return g_tail


def validate_v2_config(kernel_config: Dict, tail_config: Dict) -> None:
    """Validate v2 configuration."""
    # Validate kernel config (reuse v1 validation if available)
    required_kernel = ["grid", "weights"]
    for key in required_kernel:
        if key not in kernel_config:
            raise ValueError(f"Missing required kernel parameter: {key}")

    # Validate tail config
    required_tail = ["A_0", "alpha", "r_geo", "r_min", "Delta_r", "g_min", "Delta_g"]
    for key in required_tail:
        if key not in tail_config:
            raise ValueError(f"Missing required tail parameter: {key}")

    # Range checks
    if tail_config["A_0"] <= 0:
        raise ValueError(f"A_0 must be positive, got {tail_config['A_0']}")
    if tail_config["alpha"] < 0:
        raise ValueError(f"alpha must be non-negative, got {tail_config['alpha']}")
    if tail_config["r_geo"] <= 0:
        raise ValueError(f"r_geo must be positive, got {tail_config['r_geo']}")
    if tail_config["r_min"] < 0:
        raise ValueError(f"r_min must be non-negative, got {tail_config['r_min']}")
    if tail_config["Delta_r"] <= 0:
        raise ValueError(f"Delta_r must be positive, got {tail_config['Delta_r']}")
    if tail_config["g_min"] < 0:
        raise ValueError(f"g_min must be non-negative, got {tail_config['g_min']}")
    if tail_config["Delta_g"] <= 0:
        raise ValueError(f"Delta_g must be positive, got {tail_config['Delta_g']}")


if __name__ == "__main__":
    print("RFT v2: Acceleration-Gated Tail Solver")
    print("=" * 70)
    print()
    print("This module implements RFT v2 = kernel + gated tail.")
    print("Use via cli/rft_v2_bench.py for batch processing.")
    print()
    print("Tail formula:")
    print("  g_tail(r) = A_0 (r_geo/r)^α × gate_r(r) × gate_g(g_bar)")
    print()
    print("Gates:")
    print("  gate_r = tanh[(r - r_min) / Δr]  (suppress inner disk)")
    print("  gate_g = 1 - tanh[(g_bar - g_min) / Δg]  (activate where g_bar weak)")
    print()
    print("Parameters:")
    print("  A_0: amplitude [m/s²]")
    print("  α: power-law index")
    print("  r_geo: geometric scale [kpc]")
    print("  r_min, Δr: radial gate")
    print("  g_min, Δg: acceleration gate")
    print("=" * 70)
