"""
NFW dark matter halo solver for rotation curves (ΛCDM standard model).

Implements circular velocity from baryons + NFW dark matter halo:
  v_total² = v_baryon² + v_DM²

where v_DM comes from an NFW (Navarro-Frenk-White) halo:
  ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]

This is a simplified implementation that uses a pseudo-isothermal
approximation (cored NFW) for numerical stability.
"""

from typing import Dict, Any
import numpy as np


def apply_nfw_halo(
    case: Any,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Apply NFW dark matter halo to galaxy rotation curve.

    Parameters
    ----------
    case : GalaxyCase
        Galaxy data with baryonic components
    config : dict
        Configuration with:
          - v200_kms: virial velocity (default: fit to data)
          - c: concentration parameter (default: 10)
          - core_kpc: optional core radius for numerical stability

    Returns
    -------
    dict
        Result with:
          - r_kpc: radii
          - v_pred_kms: total predicted velocities
          - v_baryon_kms: baryonic velocities
          - v_dm_kms: dark matter velocities
    """
    # Get radii
    r = np.array(case.r_kpc, dtype=float)

    # Compute baryonic velocity
    v_disk = np.array(case.v_baryon_disk_kms, dtype=float)
    v_gas = np.array(case.v_baryon_gas_kms, dtype=float)
    if case.v_baryon_bulge_kms is not None:
        v_bulge = np.array(case.v_baryon_bulge_kms, dtype=float)
    else:
        v_bulge = np.zeros_like(r)

    v_baryon = np.sqrt(v_disk**2 + v_gas**2 + v_bulge**2)

    # NFW halo parameters
    v200 = config.get("v200_kms", None)
    c = config.get("c", 10.0)  # Typical concentration
    core_kpc = config.get("core_kpc", 0.5)  # Small core for stability

    # If v200 not specified, estimate from data
    if v200 is None:
        # Simple heuristic: use max observed velocity + 20%
        v_obs = np.array(case.v_obs_kms, dtype=float)
        v200 = np.max(v_obs) * 1.2

    # Pseudo-isothermal approximation (cored NFW)
    # v_DM(r) = v_halo * sqrt(ln(1 + r/r_s) - (r/r_s)/(1 + r/r_s))
    # Simplified with core:
    r_s = 50.0  # Scale radius ~50 kpc (typical for MW-mass galaxies)
    r_safe = np.sqrt(r**2 + core_kpc**2)  # Cored radius

    # Simple pseudo-isothermal profile
    # v_DM² ∝ v200² * [1 - (r_s/(r + r_s))]
    x = r_safe / r_s
    v_dm_factor = np.sqrt(np.log(1 + x) - x/(1 + x)) / np.sqrt(np.log(1 + c) - c/(1 + c))
    v_dm = v200 * v_dm_factor

    # Total velocity (quadrature sum)
    v_total = np.sqrt(v_baryon**2 + v_dm**2)

    return {
        "r_kpc": r.tolist(),
        "v_pred_kms": v_total.tolist(),
        "v_baryon_kms": v_baryon.tolist(),
        "v_dm_kms": v_dm.tolist(),
        "config": {
            **config,
            "v200_kms_used": float(v200),
            "r_scale_kpc": float(r_s)
        }
    }
