from __future__ import annotations

import math
from typing import Dict, Sequence

import numpy as np

DEFAULT_PASS_THRESHOLD = 10.0  # percent


def _to_numpy(name: str, values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    return arr


def compute_metrics(
    v_obs_kms: Sequence[float],
    v_pred_kms: Sequence[float],
    *,
    pass_threshold: float = DEFAULT_PASS_THRESHOLD,
) -> Dict[str, float]:
    obs = _to_numpy("v_obs_kms", v_obs_kms)
    pred = _to_numpy("v_pred_kms", v_pred_kms)
    if obs.size != pred.size:
        raise ValueError("Observed and predicted arrays must be the same length")

    mask = np.isfinite(obs) & np.isfinite(pred)
    if not mask.any():
        raise ValueError("No finite velocity pairs available for metrics")

    diff = pred[mask] - obs[mask]
    rms = float(math.sqrt(np.mean(diff ** 2)))
    mae = float(np.mean(np.abs(diff)))

    percent_mask = mask & (np.abs(obs) > 1e-9)
    if percent_mask.any():
        obs_ref = obs[percent_mask]
        rel = (pred[percent_mask] - obs_ref) / np.abs(obs_ref)
        rms_percent = float(math.sqrt(np.mean((rel * 100.0) ** 2)))
        mae_percent = float(np.mean(np.abs(rel) * 100.0))
    else:
        rms_percent = float("nan")
        mae_percent = float("nan")

    passed = bool(rms_percent <= pass_threshold) if math.isfinite(rms_percent) else False

    return {
        "n_points": int(mask.sum()),
        "rms_kms": rms,
        "mae_kms": mae,
        "rms_percent": rms_percent,
        "mae_percent": mae_percent,
        "pass": passed,
        "threshold_percent": float(pass_threshold),
    }
