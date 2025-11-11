from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def _ensure_sorted(r: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(r)
    r_sorted = r[order]
    y_sorted = y[order]
    return order, r_sorted, y_sorted


def _assert_same_len(name: str, x: np.ndarray, y: np.ndarray) -> None:
    """Assert that x and y have the same length for interpolation."""
    if len(x) != len(y):
        raise ValueError(
            f"{name}: interpolation array length mismatch - "
            f"len(x)={len(x)}, len(y)={len(y)}"
        )


def log_gaussian_convolve(
    r_kpc: np.ndarray,
    y: np.ndarray,
    sigma_ln_r: float,
    *,
    pad_ln_r: float = 0.25,
    samples_per_point: int = 4,
) -> np.ndarray:
    """Convolve ``y(r)`` along ln r with a Gaussian kernel.

    Parameters
    ----------
    r_kpc : array-like
        Radii in kpc (must be positive).
    y : array-like
        Quantity defined at each radius.
    sigma_ln_r : float
        Width of the Gaussian kernel in ln r.
    pad_ln_r : float, optional
        Extra padding (in ln r) applied to both ends before convolution.
    samples_per_point : int, optional
        Controls the resolution of the uniform ln r grid.
    """

    r_arr = np.asarray(r_kpc, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if r_arr.shape != y_arr.shape:
        raise ValueError("r_kpc and y must have matching shapes")
    if r_arr.ndim != 1:
        raise ValueError("Inputs must be 1-D arrays")
    if r_arr.size == 0:
        raise ValueError("Inputs must not be empty")
    if sigma_ln_r <= 0:
        raise ValueError("sigma_ln_r must be positive")
    if np.any(r_arr <= 0):
        raise ValueError("r_kpc values must be positive for log transform")

    if r_arr.size == 1:
        return y_arr.copy()

    order, r_sorted, y_sorted = _ensure_sorted(r_arr, y_arr)

    # Use log with small clip to avoid log(0)
    ln_r = np.log(np.clip(r_sorted, 1e-12, None))

    # De-duplicate ln_r by averaging duplicate y values
    # This prevents "fp and xp are not of the same length" errors in np.interp
    ln_r_unique, inverse_indices = np.unique(ln_r, return_inverse=True)
    y_unique = np.zeros_like(ln_r_unique, dtype=float)
    counts = np.zeros_like(ln_r_unique, dtype=float)

    # Accumulate y values for each unique ln_r
    np.add.at(y_unique, inverse_indices, y_sorted)
    np.add.at(counts, inverse_indices, 1.0)

    # Average duplicates
    y_unique = y_unique / np.clip(counts, 1.0, None)

    pad = float(pad_ln_r)
    ln_min = ln_r_unique.min() - pad
    ln_max = ln_r_unique.max() + pad
    num_points = max(int(len(ln_r_unique) * samples_per_point), 32)
    u_grid = np.linspace(ln_min, ln_max, num_points)

    # Assert arrays have same length before interpolation
    _assert_same_len("log_gaussian_convolve:first_interp", ln_r_unique, y_unique)
    y_interp = np.interp(u_grid, ln_r_unique, y_unique)

    du = u_grid[1] - u_grid[0]
    radius = max(int(math.ceil(4 * sigma_ln_r / du)), 1)

    # Cap radius to avoid kernel larger than input (causes conv length mismatch)
    max_radius = len(u_grid) // 2 - 1
    if radius > max_radius:
        radius = max(max_radius, 1)

    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * ((offsets * du) / sigma_ln_r) ** 2)
    kernel /= kernel.sum()

    conv = np.convolve(y_interp, kernel, mode="same")

    # Assert arrays have same length before second interpolation
    _assert_same_len("log_gaussian_convolve:second_interp", u_grid, conv)

    # Interpolate back to unique ln_r values
    result_unique = np.interp(ln_r_unique, u_grid, conv)

    # Expand back to full array (handling duplicates)
    result_sorted = result_unique[inverse_indices]

    inverse_order = np.empty_like(order)
    inverse_order[order] = np.arange(order.size)
    return result_sorted[inverse_order]


def log_kernel_lorentzian(drho: np.ndarray, sigma: float) -> np.ndarray:
    """
    Lorentzian (Cauchy) kernel in log-radius space.
    Heavy tails for long-range coupling.

    Args:
        drho: Log-radius offsets (Δρ)
        sigma: Width parameter

    Returns:
        Normalized kernel K(Δρ)
    """
    K = 1.0 / (1.0 + (drho / max(sigma, 1e-6))**2)
    s = K.sum()
    return K / (s if s > 0 else 1.0)


def log_kernel_mixed(
    drho: np.ndarray,
    w: float,
    sigma_g: float,
    sigma_l: float,
    asym: float = 0.0
) -> np.ndarray:
    """
    Mixed Gaussian + Lorentzian kernel (C11).

    Combines local coupling (Gaussian) with heavy-tail nonlocal effects (Lorentzian).
    This hard-codes the two key structures (local peak + tail) that universal kernels
    failed to learn under regularization.

    K(Δρ) = (1-w)·exp(-½(Δρ/σ_G)²) + w·1/(1+(Δρ/σ_L)²)

    Optionally adds asymmetry to favor outer radii:
    K(Δρ) ← K(Δρ)·[1 + a·sign(Δρ)]

    Args:
        drho: Log-radius offsets (Δρ = ρ - ρ')
        w: Lorentzian weight ∈ [0,1] (0=pure Gaussian, 1=pure Lorentzian)
        sigma_g: Gaussian width in log r
        sigma_l: Lorentzian width in log r
        asym: Asymmetry parameter (>0 favors positive Δρ, i.e., outer radii)

    Returns:
        Normalized kernel K(Δρ)

    Example:
        >>> drho = np.linspace(-3, 3, 61)
        >>> K = log_kernel_mixed(drho, w=0.5, sigma_g=0.45, sigma_l=0.60)
        >>> np.isclose(K.sum(), 1.0)  # Normalized
        True
    """
    # Gaussian component (local)
    G = np.exp(-0.5 * (drho / max(sigma_g, 1e-6))**2)

    # Lorentzian component (heavy tail)
    L = 1.0 / (1.0 + (drho / max(sigma_l, 1e-6))**2)

    # Mix
    K = (1.0 - w) * G + w * L

    # Optional asymmetry
    if asym != 0.0:
        K = K * (1.0 + asym * np.sign(drho))
        K = np.clip(K, 0.0, None)  # Ensure nonnegative

    # Normalize
    s = K.sum()
    return K / (s if s > 0 else 1.0)
