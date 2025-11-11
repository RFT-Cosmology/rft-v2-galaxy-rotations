from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Set

import numpy as np

from sparc_rft.case import GalaxyCase, case_arrays, baryon_speed_squared
from sparc_rft.kernels import log_gaussian_convolve, log_kernel_mixed


def baryon_accel(
    r_kpc: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: Optional[np.ndarray],
    v_gas: np.ndarray,
) -> np.ndarray:
    total_v2 = v_disk ** 2 + v_gas ** 2
    if v_bulge is not None:
        total_v2 = total_v2 + v_bulge ** 2
    return total_v2 / np.clip(r_kpc, 1e-6, None)


def _geometric_mean_radius(r: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(np.clip(r, 1e-6, None)))))


def _log_weighted_sum(values: np.ndarray, radii: np.ndarray) -> float:
    ln_r = np.log(np.clip(radii, 1e-6, None))
    diffs = np.diff(ln_r)
    weights = np.empty_like(values)
    if values.size == 1:
        return float(values[0])
    weights[1:-1] = (diffs[:-1] + diffs[1:]) * 0.5
    weights[0] = diffs[0] * 0.5
    weights[-1] = diffs[-1] * 0.5
    if not np.all(np.isfinite(weights)):
        return float(np.sum(values) / max(len(values), 1))
    return float(np.sum(values * weights))


def descriptors(
    r_kpc: np.ndarray,
    v_disk: np.ndarray,
    v_bulge: Optional[np.ndarray],
    v_b: np.ndarray,
    v_gas: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    n_points = r_kpc.size
    if n_points == 0:
        raise ValueError("Cannot compute descriptors for empty arrays")

    r_geo = _geometric_mean_radius(r_kpc)

    disk_peak_idx = int(np.argmax(v_disk)) if np.any(v_disk > 0) else None
    if disk_peak_idx is None or v_disk[disk_peak_idx] <= 0:
        disk_peak_idx = int(np.argmax(v_b))
    r_peak = float(r_kpc[disk_peak_idx])

    if v_bulge is not None and np.any(v_bulge > 0):
        inner_count = max(1, int(np.ceil(0.2 * n_points)))
        total_v2 = np.clip(v_b[:inner_count] ** 2, 1e-6, None)
        frac = np.clip((v_bulge[:inner_count] ** 2) / total_v2, 0.0, 1.0)
        bulge_frac = float(np.median(frac))
    else:
        bulge_frac = 0.0

    outer_count = max(2, int(np.ceil(0.2 * n_points)))
    x = np.log(np.clip(r_kpc[-outer_count:], 1e-6, None))
    y = np.log(np.clip(v_b[-outer_count:] ** 2, 1e-9, None))
    if np.allclose(x, x[0]) or outer_count < 2:
        slope_out = 0.0
    else:
        slope_out = float(np.polyfit(x, y, 1)[0])

    # C9 descriptors: xi_outer (clamped outer slope)
    xi_outer = float(np.clip(-slope_out, 0.0, 5.0))  # S_max = 5.0

    # C9 descriptors: gas_frac_outer (gas contribution in outer 30%)
    if v_gas is not None and np.any(v_gas > 0):
        outer_30pct = max(1, int(np.ceil(0.3 * n_points)))
        v_b_outer_sq = np.clip(v_b[-outer_30pct:] ** 2, 1e-9, None)
        v_gas_outer_sq = v_gas[-outer_30pct:] ** 2
        gas_frac_outer = float(np.median(np.clip(v_gas_outer_sq / v_b_outer_sq, 0.0, 1.0)))
    else:
        gas_frac_outer = 0.0

    # C9 descriptors: r_knee (radius where v_b reaches 90% of max)
    v_b_max = np.max(v_b)
    if v_b_max > 0:
        threshold = 0.9 * v_b_max
        idx_above = np.where(v_b >= threshold)[0]
        if len(idx_above) > 0:
            r_knee = float(r_kpc[idx_above[0]])
        else:
            r_knee = float(r_kpc[-1])  # Fall back to last radius
    else:
        r_knee = float(r_geo)  # Fall back to geometric mean

    return {
        "r_geo": r_geo,
        "r_peak": r_peak,
        "bulge_frac": bulge_frac,
        "slope_out": slope_out,
        "xi_outer": xi_outer,
        "gas_frac_outer": gas_frac_outer,
        "r_knee": r_knee,
    }


def _dump_modes(
    dump_dir: Path,
    r: np.ndarray,
    gb: np.ndarray,
    g_components: Dict[str, np.ndarray],
    v_pred: np.ndarray,
    v_obs: np.ndarray,
) -> None:
    dump_dir.mkdir(parents=True, exist_ok=True)
    g_shelf = g_components.get("shelf", np.zeros_like(r))
    g_tail = g_components.get("tail", np.zeros_like(r))
    g_total = gb + g_components["flat"] + g_components["spiral"] + g_components["core"] + g_shelf + g_tail
    data = np.column_stack(
        [
            r,
            gb,
            g_components["flat"],
            g_components["spiral"],
            g_components["core"],
            g_shelf,
            g_tail,
            g_total,
            v_obs,
            v_pred,
        ]
    )
    header = "r_kpc,g_baryon,g_flat,g_spiral,g_core,g_shelf,g_tail,g_total,v_obs,v_pred"
    np.savetxt(dump_dir / "components.csv", data, delimiter=",", header=header, comments="")

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(r, np.sqrt(np.clip(gb * r, 0, None)), label="Baryon", color="#1f77b4")
    ax.plot(r, v_pred, label="RFT geom", color="#d62728")
    ax.scatter(r, v_obs, label="Observed", color="#000000", s=12)
    ax.set_xlabel("Radius [kpc]")
    ax.set_ylabel("Velocity [km/s]")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(dump_dir / "velocities.png", dpi=200)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for key, color in zip(["flat", "spiral", "core"], ["#ff7f0e", "#2ca02c", "#9467bd"]):
        ax2.plot(r, g_components[key], label=f"g_{key}", color=color)
    ax2.set_xlabel("Radius [kpc]")
    ax2.set_ylabel("Acceleration [km^2 s^-2 kpc^-1]")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.savefig(dump_dir / "modes.png", dpi=200)
    plt.close(fig2)


MODE_ALIASES = {
    "mode_flattening": "flat",
    "mode_spiral": "spiral",
    "mode_core": "core",
    "mode_tail": "tail",
}


def _normalize_disable_list(disable_modes: Optional[Iterable[str]]) -> Set[str]:
    if not disable_modes:
        return set()
    disabled = set()
    for label in disable_modes:
        if not label:
            continue
        key = MODE_ALIASES.get(label.lower())
        if key:
            disabled.add(key)
    return disabled


def _validate_config(params: Dict) -> None:
    """Validate config structure and fail fast on missing required keys."""
    # Validate mode_flattening
    if "mode_flattening" not in params:
        raise ValueError("Config missing required section: 'mode_flattening'")

    flat_cfg = params["mode_flattening"]
    # Check for C9 (descriptor-driven) or C8 (fixed) mode
    has_beta = "beta0" in flat_cfg
    has_aflat = "A_flat" in flat_cfg
    if not (has_beta or has_aflat):
        raise ValueError(
            "mode_flattening must have either 'beta0' (C9 descriptor mode) or 'A_flat' (C8 fixed mode)"
        )

    # Validate mode_spiral
    if "mode_spiral" not in params:
        raise ValueError("Config missing required section: 'mode_spiral'")
    spiral_cfg = params["mode_spiral"]

    # C11: Check for mixed kernel mode
    kernel_type = spiral_cfg.get("kernel", "gauss")
    if kernel_type == "mix":
        # Mixed kernel requires w, sigma_g, sigma_l
        for key in ("w", "sigma_g", "sigma_l"):
            if key not in spiral_cfg:
                raise ValueError(f"mode_spiral.{key} required for kernel='mix'")
        w = spiral_cfg["w"]
        if not (0.0 <= w <= 1.0):
            raise ValueError("mode_spiral.w must be in [0,1]")
        if spiral_cfg.get("asym", 0.0) < 0:
            raise ValueError("mode_spiral.asym must be >= 0")
        # Mixed kernel also needs alpha params for amplitude
        required_spiral = ["alpha0", "alpha1", "alpha2"]
    else:
        # Gaussian kernel requires sigma_ln_r
        required_spiral = ["sigma_ln_r", "alpha0", "alpha1", "alpha2"]

    for key in required_spiral:
        if key not in spiral_cfg:
            raise ValueError(f"mode_spiral missing required key: '{key}'")

    # Validate mode_core
    if "mode_core" not in params:
        raise ValueError("Config missing required section: 'mode_core'")
    core_cfg = params["mode_core"]
    required_core = ["A_core", "gamma_core"]
    for key in required_core:
        if key not in core_cfg:
            raise ValueError(f"mode_core missing required key: '{key}'")

    # Mode III "shelf" is optional - only validate if present
    if "mode_shelf" in params:
        shelf_cfg = params["mode_shelf"]
        if "A_shelf" not in shelf_cfg:
            raise ValueError("mode_shelf present but missing required key: 'A_shelf'")

    # C10/C10.2: Mode IV "tail" is optional - only validate if present and enabled
    if "mode_tail" in params:
        tail_cfg = params["mode_tail"]
        if tail_cfg.get("enabled", False):
            if "A0" not in tail_cfg and "A0_kms2_per_kpc" not in tail_cfg:
                raise ValueError("mode_tail enabled but missing required key: 'A0' or 'A0_kms2_per_kpc'")
            alpha = tail_cfg.get("alpha", 1.0)
            if alpha <= 0:
                raise ValueError("mode_tail.alpha must be > 0")
            if tail_cfg.get("r_scale", "r_geo") not in ("r_geo", "r0"):
                raise ValueError("mode_tail.r_scale must be 'r_geo' or 'r0'")
            if tail_cfg.get("r_scale") == "r0" and "r0_kpc" not in tail_cfg:
                raise ValueError("mode_tail.r0_kpc required when r_scale='r0'")
            if tail_cfg.get("r_turn_kpc", 0.0) < 0:
                raise ValueError("mode_tail.r_turn_kpc must be >= 0")
            if tail_cfg.get("p", 0.0) < 0:
                raise ValueError("mode_tail.p must be >= 0")


def rft_geom_predict(
    case: GalaxyCase,
    params: Dict,
    dump_dir: Optional[str] = None,
    *,
    disable_modes: Optional[Iterable[str]] = None,
) -> Dict[str, np.ndarray]:
    # Validate config at entry
    _validate_config(params)

    disabled = _normalize_disable_list(disable_modes)
    r, v_disk, v_bulge, v_gas = case_arrays(case)
    if r.size < params.get("regularization", {}).get("min_points", 6):
        raise ValueError("Insufficient radial samples for RFT geometry solver")

    gb = baryon_accel(r, v_disk, v_bulge, v_gas)
    total_speed = np.sqrt(np.clip(baryon_speed_squared(v_disk, v_gas, v_bulge), 0.0, None))
    desc = descriptors(r, v_disk, v_bulge, total_speed, v_gas)

    flat_cfg = params["mode_flattening"]
    r_geo = desc["r_geo"]
    # FIX: Use geometric mean of gb (log-mean) instead of log-weighted sum
    # This matches RFT spec and improves low-acceleration disk support
    log_mean = np.exp(np.mean(np.log(np.clip(gb, 1e-12, None))))
    r_safe = np.clip(r, 1e-6, None)
    ratio = r_geo / r_safe
    r_cut = 1.3 * max(r_geo, 1e-6)
    gate = 1.0 / (1.0 + (np.clip(r_cut, 1e-6, None) / r_safe) ** 4)
    gate = gate ** 2

    # C9: Descriptor-driven A_flat (backward compatible with fixed A_flat)
    A_flat_clamped = False
    if "beta0" in flat_cfg:
        # C9 mode: β-mapping
        A_flat_raw = (
            flat_cfg["beta0"]
            + flat_cfg.get("beta1", 0.0) * desc["xi_outer"]
            + flat_cfg.get("beta2", 0.0) * desc["gas_frac_outer"]
        )
        # Clamp A_flat to physical bounds
        A_min = flat_cfg.get("A_flat_min", 0.18)
        A_max = flat_cfg.get("A_flat_max", 0.50)
        A_flat = float(np.clip(A_flat_raw, A_min, A_max))
        if A_flat != A_flat_raw:
            A_flat_clamped = True
    else:
        # C8 mode: fixed A_flat
        A_flat = flat_cfg["A_flat"]

    g_flat = A_flat * log_mean * ratio * gate
    if "flat" in disabled:
        g_flat = np.zeros_like(g_flat)

    spiral_cfg = params["mode_spiral"]
    reg = params.get("regularization", {})

    # C11: Kernel dispatch - support both Gaussian and mixed kernels
    kernel_type = spiral_cfg.get("kernel", "gauss")
    if kernel_type == "mix":
        # Mixed kernel: Gaussian (local) + Lorentzian (heavy tail)
        # Build kernel at each point and convolve manually
        rho = np.log(np.clip(r, 1e-12, None) / max(r_geo, 1e-6))
        g_conv = np.zeros_like(r)

        # Extract mixed kernel params
        w = spiral_cfg["w"]
        sigma_g = spiral_cfg["sigma_g"]
        sigma_l = spiral_cfg["sigma_l"]
        asym = spiral_cfg.get("asym", 0.0)

        # Convolve at each point
        for i in range(len(r)):
            drho = rho[i] - rho
            K = log_kernel_mixed(drho, w, sigma_g, sigma_l, asym)
            # Trapezoidal integration in log-radius
            if len(rho) > 1:
                d_rho = np.diff(rho)
                weights = np.zeros_like(rho)
                weights[0] = d_rho[0] * 0.5 if len(d_rho) > 0 else 1.0
                weights[-1] = d_rho[-1] * 0.5 if len(d_rho) > 0 else 1.0
                if len(d_rho) > 1:
                    weights[1:-1] = (d_rho[:-1] + d_rho[1:]) * 0.5
                g_conv[i] = np.sum(K * gb * weights)
            else:
                g_conv[i] = gb[i]
    else:
        # Gaussian kernel (default, C8-C10 behavior)
        g_conv = log_gaussian_convolve(
            r,
            gb,
            spiral_cfg["sigma_ln_r"],
            pad_ln_r=reg.get("pad_ln_r", 0.25),
        )

    A_I = (
        spiral_cfg["alpha0"]
        + spiral_cfg["alpha1"] * max(0.0, -desc["slope_out"])
        + spiral_cfg["alpha2"] * desc["bulge_frac"]
    )
    g_spiral = A_I * g_conv
    if "spiral" in disabled:
        g_spiral = np.zeros_like(g_spiral)

    core_cfg = params["mode_core"]
    rc = core_cfg["gamma_core"] * max(desc["r_peak"], 1e-6) + core_cfg["eps_kpc"]
    g_core = core_cfg["A_core"] * np.exp(-((r / rc) ** 2)) * gb
    if "core" in disabled:
        g_core = np.zeros_like(g_core)

    # C9: Mode III "shelf" for long-scale outer plateaus (optional)
    if "mode_shelf" in params and params["mode_shelf"].get("A_shelf", 0.0) > 0:
        shelf_cfg = params["mode_shelf"]
        r_knee = desc["r_knee"]
        p = shelf_cfg.get("p", 1.5)
        # Shelf: A_shelf * (1 - exp(-(r/r_knee)^p)) * g_b
        shelf_factor = 1.0 - np.exp(-((r_safe / max(r_knee, 1e-6)) ** p))
        g_shelf = shelf_cfg["A_shelf"] * shelf_factor * gb
        if "shelf" in disabled:
            g_shelf = np.zeros_like(g_shelf)
    else:
        g_shelf = np.zeros_like(r)

    # C10/C10.2: Mode IV "tail" - scale-decoupled power-law acceleration (optional)
    # Physics: g_tail(r) = A0 * (r_star / r)^alpha
    # Result: v_tail² = A0 * r_star^alpha * r^(1-alpha)
    # - alpha = 1.0 → constant v (C10 baseline)
    # - alpha < 1.0 → rising v(r) for outer regions (C10.2)
    # - alpha > 1.0 → declining v(r) (unused)
    if "mode_tail" in params and params["mode_tail"].get("enabled", False):
        tail_cfg = params["mode_tail"]
        # Support both key names for backward compatibility
        A0 = tail_cfg.get("A0", tail_cfg.get("A0_kms2_per_kpc", 0.0))
        alpha = float(tail_cfg.get("alpha", 1.0))

        # Compute scale radius
        if tail_cfg.get("r_scale", "r_geo") == "r_geo":
            r_star = r_geo  # Use geometric mean radius
        else:
            r_star = float(tail_cfg.get("r0_kpc", np.median(r)))

        # Power-law tail: (r_star / r)^alpha
        base = (np.clip(r_star, 1e-6, None) / r_safe) ** alpha

        # Optional gentle onset gate for inner regions
        gate = 1.0
        r_turn = float(tail_cfg.get("r_turn_kpc", 0.0))
        p_gate = float(tail_cfg.get("p", 0.0))
        if r_turn > 0.0 and p_gate > 0.0:
            gate = 1.0 - np.exp(-(r_safe / max(r_turn, 1e-6)) ** p_gate)

        g_tail = A0 * base * gate

        if "mode_tail" in disabled or "tail" in disabled:
            g_tail = np.zeros_like(g_tail)
    else:
        g_tail = np.zeros_like(r)

    g_components = {"flat": g_flat, "spiral": g_spiral, "core": g_core, "shelf": g_shelf, "tail": g_tail}
    g_res = g_flat + g_spiral + g_core + g_shelf + g_tail
    v_pred_sq = r * (gb + g_res)
    v_pred = np.sqrt(np.clip(v_pred_sq, 0.0, None))

    if dump_dir:
        _dump_modes(Path(dump_dir), r, gb, g_components, v_pred, np.asarray(case.v_obs_kms, dtype=float))

    # Add C9-specific metadata
    desc_with_meta = dict(desc)
    desc_with_meta["A_flat_clamped"] = A_flat_clamped
    if "beta0" in flat_cfg:
        desc_with_meta["A_flat_value"] = A_flat

    return {
        "r_kpc": r,
        "v_pred_kms": v_pred,
        "g_components": g_components,
        "descriptors": desc_with_meta,
    }
