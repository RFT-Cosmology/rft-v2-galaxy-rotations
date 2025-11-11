from __future__ import annotations

import math
from typing import Iterable, Tuple

import numpy as np


def residuals(v_obs: Iterable[float], v_pred: Iterable[float]) -> np.ndarray:
    obs = np.asarray(v_obs, dtype=float)
    pred = np.asarray(v_pred, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError("Observed and predicted arrays must share the same shape")
    return pred - obs


def _poly_detrend(ln_r: np.ndarray, residual: np.ndarray, degree: int = 2) -> np.ndarray:
    if ln_r.size <= degree:
        return np.zeros_like(residual)
    coeffs = np.polyfit(ln_r, residual, deg=degree)
    trend = np.polyval(coeffs, ln_r)
    return residual - trend


def high_pass_residuals(
    r_kpc: Iterable[float],
    residual_values: Iterable[float],
    *,
    poly_degree: int = 2,
) -> np.ndarray:
    r = np.asarray(r_kpc, dtype=float)
    residual = np.asarray(residual_values, dtype=float)
    if r.shape != residual.shape:
        raise ValueError("Radius and residual arrays must match")
    mask = np.isfinite(r) & np.isfinite(residual)
    if not mask.any():
        return np.zeros_like(residual)
    ln_r = np.log(np.clip(r[mask], 1e-6, None))
    detrended = _poly_detrend(ln_r, residual[mask], degree=min(poly_degree, max(1, mask.sum() - 1)))
    result = np.zeros_like(residual)
    result[mask] = detrended
    return result


def oscillation_index(
    r_kpc: Iterable[float],
    v_obs: Iterable[float],
    v_pred: Iterable[float],
    *,
    poly_degree: int = 2,
) -> Tuple[float, np.ndarray]:
    obs = np.asarray(v_obs, dtype=float)
    res = residuals(v_obs, v_pred)
    hp = high_pass_residuals(r_kpc, res, poly_degree=poly_degree)
    mask = np.isfinite(hp) & np.isfinite(obs)
    if not mask.any():
        return float("nan"), hp
    rms_hp = math.sqrt(float(np.mean(hp[mask] ** 2)))
    denom = float(np.mean(np.abs(obs[mask])))
    if denom <= 1e-9:
        return float("nan"), hp
    return (rms_hp / denom) * 100.0, hp
