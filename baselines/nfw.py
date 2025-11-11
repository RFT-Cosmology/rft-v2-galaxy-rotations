"""
NFW (Navarro-Frenk-White) dark matter halo rotation curve fitting.

Fits a 2-parameter NFW dark matter halo profile to the rotation curve:
- ρₛ: Characteristic density (M☉/kpc³)
- rₛ: Scale radius (kpc)

Uses a SciPy-free optimization strategy:
1. Coarse log-grid search over parameter space
2. Pattern-search refinement (Hooke-Jeeves style)

No external dependencies beyond NumPy.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

# Gravitational constant in kpc·km²·s⁻²·M☉⁻¹
G_KPC = 4.300917e-6

# Default parameter bounds and initial values
DEFAULT_RHO_S_BOUNDS = (1e6, 1e9)  # M☉/kpc³
DEFAULT_R_S_BOUNDS = (0.1, 100.0)  # kpc
DEFAULT_RHO_S_INIT = 1e8  # M☉/kpc³
DEFAULT_R_S_INIT = 10.0  # kpc


def _nfw_mass(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    """
    Compute NFW mass profile.

    M(r) = 4π ρₛ rₛ³ [ln(1+x) - x/(1+x)]
    where x = r/rₛ

    Args:
        r_kpc: Radius in kpc
        rho_s: Characteristic density (M☉/kpc³)
        r_s: Scale radius (kpc)

    Returns:
        Enclosed mass in M☉
    """
    r_s_safe = np.clip(r_s, 1e-6, None)
    x = np.clip(r_kpc, 1e-6, None) / r_s_safe

    # M(r) = 4π ρₛ rₛ³ [ln(1+x) - x/(1+x)]
    term = np.log(1 + x) - x / (1 + x)
    mass = 4 * np.pi * rho_s * (r_s_safe**3) * term

    return mass


def _v_dm_squared(r_kpc: np.ndarray, rho_s: float, r_s: float) -> np.ndarray:
    """
    Compute dark matter velocity squared.

    v²_dm = G·M(r)/r

    Args:
        r_kpc: Radius in kpc
        rho_s: Characteristic density (M☉/kpc³)
        r_s: Scale radius (kpc)

    Returns:
        Dark matter velocity squared (km²/s²)
    """
    r_safe = np.clip(r_kpc, 1e-6, None)
    mass = _nfw_mass(r_kpc, rho_s, r_s)
    v2_dm = G_KPC * mass / r_safe

    return np.clip(v2_dm, 0, None)


def _compute_loss(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    v_baryon_sq: np.ndarray,
    rho_s: float,
    r_s: float,
) -> float:
    """
    Compute loss function for NFW fit.

    Loss = mean((v_pred - v_obs)²)
    where v_pred² = v_baryon² + v_dm²

    Args:
        r_kpc: Radius array
        v_obs_kms: Observed velocity
        v_baryon_sq: Baryonic velocity squared
        rho_s: NFW density parameter
        r_s: NFW scale radius

    Returns:
        Mean squared error
    """
    v_dm_sq = _v_dm_squared(r_kpc, rho_s, r_s)
    v_pred_sq = v_baryon_sq + v_dm_sq
    v_pred = np.sqrt(np.clip(v_pred_sq, 0, None))

    residuals = v_pred - v_obs_kms
    loss = np.nanmean(residuals**2)

    return float(loss)


def _coarse_grid_search(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    v_baryon_sq: np.ndarray,
    rho_s_bounds: Tuple[float, float],
    r_s_bounds: Tuple[float, float],
    n_grid: int = 12,
) -> Tuple[float, float, float]:
    """
    Coarse grid search over log-space parameter grid.

    Args:
        r_kpc: Radius array
        v_obs_kms: Observed velocity
        v_baryon_sq: Baryonic velocity squared
        rho_s_bounds: (min, max) for ρₛ
        r_s_bounds: (min, max) for rₛ
        n_grid: Grid resolution per dimension

    Returns:
        (best_loss, best_rho_s, best_r_s)
    """
    rho_min, rho_max = rho_s_bounds
    r_min, r_max = r_s_bounds

    rho_grid = np.logspace(np.log10(rho_min), np.log10(rho_max), n_grid)
    r_grid = np.logspace(np.log10(r_min), np.log10(r_max), n_grid)

    best_loss = float('inf')
    best_rho = rho_grid[0]
    best_r = r_grid[0]

    for rho_s in rho_grid:
        for r_s in r_grid:
            loss = _compute_loss(r_kpc, v_obs_kms, v_baryon_sq, rho_s, r_s)
            if loss < best_loss:
                best_loss = loss
                best_rho = rho_s
                best_r = r_s

    return best_loss, best_rho, best_r


def _pattern_search_refine(
    r_kpc: np.ndarray,
    v_obs_kms: np.ndarray,
    v_baryon_sq: np.ndarray,
    rho_s_init: float,
    r_s_init: float,
    rho_s_bounds: Tuple[float, float],
    r_s_bounds: Tuple[float, float],
    max_iter: int = 200,
) -> Tuple[float, float, float]:
    """
    Pattern search (Hooke-Jeeves) refinement.

    Explores multiplicative steps: {0.8, 1.0, 1.25} for each parameter.

    Args:
        r_kpc: Radius array
        v_obs_kms: Observed velocity
        v_baryon_sq: Baryonic velocity squared
        rho_s_init: Initial ρₛ
        r_s_init: Initial rₛ
        rho_s_bounds: (min, max) for ρₛ
        r_s_bounds: (min, max) for rₛ
        max_iter: Maximum iterations

    Returns:
        (best_loss, best_rho_s, best_r_s)
    """
    rho_min, rho_max = rho_s_bounds
    r_min, r_max = r_s_bounds

    rho_s = rho_s_init
    r_s = r_s_init
    best_loss = _compute_loss(r_kpc, v_obs_kms, v_baryon_sq, rho_s, r_s)

    # Pattern search steps (multiplicative)
    steps = [
        (0.8, 1.0),   # Decrease rho
        (1.25, 1.0),  # Increase rho
        (1.0, 0.8),   # Decrease r
        (1.0, 1.25),  # Increase r
    ]

    for iteration in range(max_iter):
        improved = False

        for rho_mult, r_mult in steps:
            rho_try = np.clip(rho_s * rho_mult, rho_min, rho_max)
            r_try = np.clip(r_s * r_mult, r_min, r_max)

            loss = _compute_loss(r_kpc, v_obs_kms, v_baryon_sq, rho_try, r_try)

            if loss < best_loss:
                rho_s = rho_try
                r_s = r_try
                best_loss = loss
                improved = True
                break  # Take first improvement (greedy)

        if not improved:
            # No improvement found, converged
            break

    return best_loss, rho_s, r_s


def nfw_fit_predict(
    case,
    rho_s_bounds: Optional[Tuple[float, float]] = None,
    r_s_bounds: Optional[Tuple[float, float]] = None,
    rho_s_init: Optional[float] = None,
    r_s_init: Optional[float] = None,
    max_iter: int = 200,
) -> Tuple[np.ndarray, Dict]:
    """
    Fit NFW dark matter halo and predict rotation curve.

    Two-stage optimization:
    1. Coarse log-grid search (12×12 grid)
    2. Pattern search refinement

    Args:
        case: GalaxyCase with r_kpc, v_obs_kms, v_baryon_* arrays
        rho_s_bounds: (min, max) for ρₛ in M☉/kpc³ (default: 1e6 to 1e9)
        r_s_bounds: (min, max) for rₛ in kpc (default: 0.1 to 100.0)
        rho_s_init: Initial ρₛ (default: 1e8 M☉/kpc³)
        r_s_init: Initial rₛ (default: 10.0 kpc)
        max_iter: Maximum pattern search iterations (default: 200)

    Returns:
        Tuple of:
        - v_pred_kms: Predicted velocity in km/s
        - params: Dictionary with {"rho_s", "r_s", "loss"}
    """
    # Set defaults
    if rho_s_bounds is None:
        rho_s_bounds = DEFAULT_RHO_S_BOUNDS
    if r_s_bounds is None:
        r_s_bounds = DEFAULT_R_S_BOUNDS
    if rho_s_init is None:
        rho_s_init = DEFAULT_RHO_S_INIT
    if r_s_init is None:
        r_s_init = DEFAULT_R_S_INIT

    # Convert case data to numpy arrays
    r_kpc = np.asarray(case.r_kpc, dtype=float)
    v_obs_kms = np.asarray(case.v_obs_kms, dtype=float)
    v_disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    v_gas = np.asarray(case.v_baryon_gas_kms, dtype=float)

    # Compute total baryonic velocity squared
    if case.v_baryon_bulge_kms is not None:
        v_bulge = np.asarray(case.v_baryon_bulge_kms, dtype=float)
    else:
        v_bulge = np.zeros_like(v_disk)

    v_baryon_sq = v_disk**2 + v_bulge**2 + v_gas**2

    # Stage 1: Coarse grid search
    loss_coarse, rho_coarse, r_coarse = _coarse_grid_search(
        r_kpc,
        v_obs_kms,
        v_baryon_sq,
        rho_s_bounds,
        r_s_bounds,
    )

    # Stage 2: Pattern search refinement
    loss_final, rho_final, r_final = _pattern_search_refine(
        r_kpc,
        v_obs_kms,
        v_baryon_sq,
        rho_coarse,
        r_coarse,
        rho_s_bounds,
        r_s_bounds,
        max_iter,
    )

    # Compute final prediction
    v_dm_sq = _v_dm_squared(r_kpc, rho_final, r_final)
    v_pred_sq = v_baryon_sq + v_dm_sq
    v_pred_kms = np.sqrt(np.clip(v_pred_sq, 0, None))

    params = {
        "rho_s": float(rho_final),
        "r_s": float(r_final),
        "loss": float(loss_final),
    }

    return v_pred_kms, params
