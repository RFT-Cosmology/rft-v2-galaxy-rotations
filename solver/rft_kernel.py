#!/usr/bin/env python3
"""
C13: Universal Kernel RFT Solver

Global, nonparametric convolution kernel K(Δρ) in log-radius space.
No per-galaxy tuning - one kernel fits all galaxies.

Physics:
    ρ = ln(r / r_geo)  (dimensionless log-radius)
    g_res(ρ) = ∫ K(ρ - ρ') · g_b(ρ') dρ'
    v_pred²(ρ) = v_b²(ρ) + r(ρ) · g_res(ρ)

Key properties:
    - K(Δρ) ≥ 0 (nonnegative)
    - K smooth via Tikhonov regularization
    - Learns both tail behavior and mid-scale structure
"""

import numpy as np
from typing import Dict, Optional
from sparc_rft.case import GalaxyCase


def _geometric_mean_radius(r: np.ndarray) -> float:
    """Compute r_geo = exp(mean(ln(r)))."""
    return float(np.exp(np.mean(np.log(np.clip(r, 1e-6, None)))))


def apply_kernel(
    case: GalaxyCase,
    kernel_config: Dict,
    disable_modes: Optional[list] = None,
) -> Dict:
    """
    Apply learned universal kernel to predict rotation curve.

    Args:
        case: Galaxy case with r_kpc, v_obs_kms, v_baryon_*
        kernel_config: Dict with keys:
            - grid: Δρ knot positions (e.g., [-3, -2.85, ..., 3])
            - weights: K(Δρ) values at knots (nonnegative)
            - r_scale: "r_geo" (default) or "median"
            - lambda: regularization (for documentation)
        disable_modes: Not used (for API compatibility)

    Returns:
        Dict with keys:
            - r_kpc: radii
            - v_pred_kms: predicted velocities
            - g_components: {"kernel": g_res}
            - descriptors: {"r_geo": ...}
    """
    if disable_modes and "kernel" in disable_modes:
        # Degenerate case - kernel disabled means Newtonian only
        r = np.asarray(case.r_kpc, dtype=float)
        v_disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
        v_gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
        v_bulge = getattr(case, "v_baryon_bulge_kms", None)
        if v_bulge is None:
            v_bulge = np.zeros_like(r)
        else:
            v_bulge = np.asarray(v_bulge, dtype=float)

        v_b = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)
        return {
            "r_kpc": r,
            "v_pred_kms": v_b,
            "g_components": {"kernel": np.zeros_like(r)},
            "descriptors": {"r_geo": _geometric_mean_radius(r)},
        }

    # Extract case data
    r = np.asarray(case.r_kpc, dtype=float)
    v_disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    v_gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
    v_bulge = getattr(case, "v_baryon_bulge_kms", None)
    if v_bulge is None:
        v_bulge = np.zeros_like(r)
    else:
        v_bulge = np.asarray(v_bulge, dtype=float)

    # Baryonic speed and acceleration
    v_b_sq = v_disk**2 + v_gas**2 + v_bulge**2
    v_b = np.sqrt(v_b_sq)
    g_b = v_b_sq / np.clip(r, 1e-6, None)  # km²/s² / kpc

    # Compute scale radius
    r_scale_method = kernel_config.get("r_scale", "r_geo")
    if r_scale_method == "r_geo":
        r_star = _geometric_mean_radius(r)
    elif r_scale_method == "median":
        r_star = float(np.median(r))
    else:
        raise ValueError(f"Unknown r_scale method: {r_scale_method}")

    # Transform to dimensionless log-radius
    rho = np.log(r / r_star)  # ρ = ln(r / r_geo)

    # Extract kernel
    grid = np.asarray(kernel_config["grid"], dtype=float)  # Δρ knots
    weights = np.asarray(kernel_config["weights"], dtype=float)  # K(Δρ)

    if len(grid) != len(weights):
        raise ValueError(
            f"Kernel grid ({len(grid)}) and weights ({len(weights)}) size mismatch"
        )

    # Convolve g_b in log-radius space
    g_res = _convolve_kernel_log(rho, g_b, grid, weights)

    # Predict velocity
    v_pred_sq = v_b_sq + r * g_res
    v_pred = np.sqrt(np.clip(v_pred_sq, 0.0, None))

    return {
        "r_kpc": r,
        "v_pred_kms": v_pred,
        "g_components": {"kernel": g_res},
        "descriptors": {
            "r_geo": r_star,
            "kernel_lambda": kernel_config.get("lambda", 0.0),
        },
    }


def _convolve_kernel_log(
    rho: np.ndarray, signal: np.ndarray, grid: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """
    Convolve signal in log-space using learned kernel.

    Args:
        rho: Log-radius positions (dimensionless)
        signal: g_b(ρ) values
        grid: Δρ knot positions
        weights: K(Δρ) values at knots

    Returns:
        Convolved signal g_res(ρ) = ∫ K(ρ - ρ') g_b(ρ') dρ'

    Implementation:
        Use linear interpolation of K(Δρ) between knots.
        For each output point ρ_i, compute:
            g_res[i] = Σ_j K(ρ_i - ρ_j) · g_b[j] · Δρ
        where Δρ is the spacing (approximate integral).
    """
    n = len(rho)
    g_res = np.zeros(n)

    # Compute spacing (assume roughly uniform in original radius)
    if n < 2:
        return g_res

    # For each output point
    for i in range(n):
        # Compute Δρ = ρ_i - ρ_j for all j
        delta_rho = rho[i] - rho  # shape: (n,)

        # Interpolate kernel weights at these Δρ positions
        # Use linear interpolation with extrapolation set to 0
        kernel_vals = np.interp(delta_rho, grid, weights, left=0.0, right=0.0)

        # Approximate integral: Σ K(Δρ) · g_b(ρ') · dρ'
        # Use midpoint spacing estimation
        if i == 0:
            d_rho = rho[1] - rho[0] if n > 1 else 1.0
        elif i == n - 1:
            d_rho = rho[i] - rho[i - 1]
        else:
            d_rho = 0.5 * (rho[i + 1] - rho[i - 1])

        # Weighted sum (discrete convolution)
        g_res[i] = float(np.sum(kernel_vals * signal) * d_rho / max(n, 1))

    return g_res


def validate_kernel_config(config: Dict) -> None:
    """Validate kernel configuration."""
    if "grid" not in config:
        raise ValueError("Kernel config missing 'grid'")
    if "weights" not in config:
        raise ValueError("Kernel config missing 'weights'")

    grid = np.asarray(config["grid"])
    weights = np.asarray(config["weights"])

    if len(grid) != len(weights):
        raise ValueError(
            f"Grid size ({len(grid)}) != weights size ({len(weights)})"
        )

    if np.any(weights < 0):
        raise ValueError("Kernel weights must be nonnegative")

    if not np.all(np.diff(grid) > 0):
        raise ValueError("Kernel grid must be strictly increasing")
