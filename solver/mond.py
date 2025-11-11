"""
MOND (Modified Newtonian Dynamics) solver for rotation curves.

Implements the simple interpolation function form of MOND:
  g_mond = g_N * μ(g_N/a0)

where:
  - g_N is Newtonian gravitational acceleration from baryons
  - a0 is the MOND acceleration scale (~1.2e-10 m/s² ≈ 3.7 km²/s²/kpc)
  - μ(x) is the interpolation function

Standard interpolation functions:
  - Simple: μ(x) = x / (1 + x)
  - Standard: μ(x) = x / sqrt(1 + x²)
"""

from typing import Dict, Any
import numpy as np


def apply_mond(
    case: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply MOND prescription to galaxy rotation curve.

    Parameters
    ----------
    case : GalaxyCase
        Galaxy data with baryonic components
    config : dict
        Configuration with:
          - a0_kms2_per_kpc: MOND acceleration scale (default: 3.7)
          - mu_form: 'simple' or 'standard' (default: 'standard')

    Returns
    -------
    dict
        Result with:
          - r_kpc: radii
          - v_pred_kms: predicted velocities
          - v_baryon_kms: baryonic velocities
          - g_baryon_kms2_per_kpc: baryonic acceleration
          - g_mond_kms2_per_kpc: MOND acceleration
    """
    # Extract config
    a0 = config.get("a0_kms2_per_kpc", 3.7)  # Standard MOND scale
    mu_form = config.get("mu_form", "standard")

    # Get radii
    r = np.array(case.r_kpc, dtype=float)

    # Compute baryonic velocity components
    v_disk = np.array(case.v_baryon_disk_kms, dtype=float)
    v_gas = np.array(case.v_baryon_gas_kms, dtype=float)

    # Handle optional bulge
    if case.v_baryon_bulge_kms is not None:
        v_bulge = np.array(case.v_baryon_bulge_kms, dtype=float)
    else:
        v_bulge = np.zeros_like(r)

    # Total baryonic velocity (quadrature sum)
    v_baryon = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)

    # Convert to acceleration: g_N = v²/r
    r_safe = np.clip(r, 1e-6, None)
    g_N = v_baryon**2 / r_safe  # km²/s²/kpc

    # MOND interpolation function
    x = g_N / a0
    if mu_form == "simple":
        # Simple: μ(x) = x/(1+x)
        mu = x / (1.0 + x)
    elif mu_form == "standard":
        # Standard: μ(x) = x/sqrt(1+x²)
        mu = x / np.sqrt(1.0 + x**2)
    else:
        raise ValueError(f"Unknown mu_form: {mu_form}")

    # MOND acceleration: g_mond = g_N * μ(g_N/a0)
    g_mond = g_N * mu

    # Convert back to velocity: v = sqrt(g * r)
    v_mond = np.sqrt(g_mond * r_safe)

    return {
        "r_kpc": r.tolist(),
        "v_pred_kms": v_mond.tolist(),
        "v_baryon_kms": v_baryon.tolist(),
        "g_baryon_kms2_per_kpc": g_N.tolist(),
        "g_mond_kms2_per_kpc": g_mond.tolist(),
        "mu": mu.tolist(),
        "config": config
    }
