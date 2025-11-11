"""
MOND (Modified Newtonian Dynamics) rotation curve prediction.

Implements three MOND interpolation functions:
- standard: μ(x) = x/√(1+x²)  (Bekenstein & Milgrom 1984)
- simple: μ(x) = x/(1+x)      (simple interpolation)
- qumond: ν(y) = 1/2 + 1/2√(1+4/y)  (QUMOND, Milgrom 2010)

No per-galaxy tuning - uses global a₀ only.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Constants
KPC_IN_KM = 3.085677581491367e16 / 1e3  # kpc to km conversion
DEFAULT_A0_M_S2 = 1.2e-10  # Canonical MOND acceleration (m/s²)


def _compute_newtonian_accel(r_kpc: np.ndarray, v_b_kms: np.ndarray) -> np.ndarray:
    """
    Compute Newtonian acceleration from baryonic velocity.

    a_N = v²/r in km/s² units.

    Args:
        r_kpc: Radius in kpc
        v_b_kms: Baryonic velocity in km/s

    Returns:
        Newtonian acceleration in km/s²
    """
    r_km = np.clip(r_kpc, 1e-9, None) * KPC_IN_KM
    return (v_b_kms**2) / r_km


def _mu_standard(x: np.ndarray) -> np.ndarray:
    """Standard MOND interpolation: μ(x) = x/√(1+x²)."""
    return x / np.sqrt(1 + x * x)


def _mu_simple(x: np.ndarray) -> np.ndarray:
    """Simple MOND interpolation: μ(x) = x/(1+x)."""
    return x / (1 + x)


def _mu(x: np.ndarray, law: str) -> np.ndarray:
    """
    MOND interpolation function μ(x).

    Args:
        x: Dimensionless ratio a/a₀
        law: "standard" or "simple"

    Returns:
        μ(x) in range [0,1]
    """
    if law == "standard":
        return _mu_standard(x)
    elif law == "simple":
        return _mu_simple(x)
    else:
        raise ValueError(f"Unknown MOND law: {law}. Use 'standard' or 'simple'.")


def _solve_mond_standard(a_N: np.ndarray, a0: float, law: str, max_iter: int = 20) -> np.ndarray:
    """
    Solve MOND equation via fixed-point iteration.

    Standard/simple laws: a·μ(a/a₀) = a_N
    Iteratively solve: a = a_N / μ(a/a₀)

    Args:
        a_N: Newtonian acceleration (km/s²)
        a0: MOND acceleration scale (km/s²)
        law: "standard" or "simple"
        max_iter: Maximum iterations

    Returns:
        MOND acceleration a (km/s²)
    """
    a = np.copy(a_N)
    a0_safe = np.clip(a0, 1e-30, None)

    for _ in range(max_iter):
        mu_val = _mu(a / a0_safe, law)
        mu_val = np.clip(mu_val, 1e-12, None)  # Avoid division by zero
        a_new = a_N / mu_val

        # Check convergence
        if np.allclose(a_new, a, rtol=1e-6):
            break
        a = a_new

    return a


def _solve_mond_qumond(a_N: np.ndarray, a0: float) -> np.ndarray:
    """
    Solve MOND equation using QUMOND formulation.

    QUMOND: a = ν(a_N/a₀)·a_N
    where ν(y) = 1/2 + 1/2√(1+4/y)

    Args:
        a_N: Newtonian acceleration (km/s²)
        a0: MOND acceleration scale (km/s²)

    Returns:
        MOND acceleration a (km/s²)
    """
    y = a_N / np.clip(a0, 1e-30, None)
    y_safe = np.clip(y, 1e-12, None)
    nu = 0.5 + 0.5 * np.sqrt(1 + 4 / y_safe)
    return nu * a_N


def mond_predict(
    case,
    a0_m_s2: float = DEFAULT_A0_M_S2,
    law: str = "standard",
) -> Tuple[np.ndarray, Dict]:
    """
    Predict rotation curve using MOND.

    No per-galaxy parameters - uses only global a₀.

    Args:
        case: GalaxyCase with r_kpc, v_baryon_* arrays
        a0_m_s2: MOND acceleration scale in m/s² (default: 1.2e-10)
        law: MOND interpolation law - "standard", "simple", or "qumond"

    Returns:
        Tuple of:
        - v_pred_kms: Predicted velocity in km/s
        - params: Dictionary with {"a0_m_s2": a0_m_s2, "law": law}
    """
    # Convert a₀ to km/s²
    a0 = a0_m_s2 / 1e3

    # Convert case data to numpy arrays
    r_kpc = np.asarray(case.r_kpc, dtype=float)
    v_disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    v_gas = np.asarray(case.v_baryon_gas_kms, dtype=float)

    # Compute total baryonic velocity
    if case.v_baryon_bulge_kms is not None:
        v_bulge = np.asarray(case.v_baryon_bulge_kms, dtype=float)
    else:
        v_bulge = np.zeros_like(v_disk)

    v_baryon = np.sqrt(v_disk**2 + v_bulge**2 + v_gas**2)

    # Compute Newtonian acceleration
    a_N = _compute_newtonian_accel(r_kpc, v_baryon)

    # Solve for MOND acceleration
    if law == "qumond":
        a = _solve_mond_qumond(a_N, a0)
    else:
        a = _solve_mond_standard(a_N, a0, law)

    # Convert acceleration to velocity: v² = a·r
    r_km = r_kpc * KPC_IN_KM
    v_pred_kms = np.sqrt(np.clip(a * r_km, 0, None))

    params = {
        "a0_m_s2": a0_m_s2,
        "law": law,
    }

    return v_pred_kms, params
