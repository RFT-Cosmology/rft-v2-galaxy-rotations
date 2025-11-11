#!/usr/bin/env python3
"""
Figure 5: Representative Rotation Curves

Shows six TEST galaxies covering:
  - Two RFT wins (RFT passes, NFW_global fails)
  - Two ties (both models pass)
  - Two near-misses (NFW_global passes, RFT fails)

Each panel includes observed velocities (with σ), RFT v2, NFW_global, and MOND
predictions plus fractional residuals. RFT configuration:
config/global_rc_v2_frozen.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from baselines.mond import mond_predict  # noqa: E402
from baselines.nfw import _v_dm_squared  # noqa: E402
from metrics.rc_metrics import compute_metrics  # noqa: E402
from solver.rft_v2_gated_tail import apply_v2_gated_tail  # noqa: E402
from sparc_rft.case import GalaxyCase, load_case  # noqa: E402

WINDOW_MIN = 1.0
WINDOW_MAX = 30.0
PREDICTIVE_THRESHOLD = 20.0
COLORS = {
    "obs": "#000000",
    "rft": "#0072B2",
    "nfw": "#E69F00",
    "mond": "#009E73",
}


@dataclass
class GalaxyCurves:
    name: str
    label: str
    radius: np.ndarray
    v_obs: np.ndarray
    sigma_v: np.ndarray
    v_rft: np.ndarray
    v_nfw: np.ndarray
    v_mond: np.ndarray
    res_rft: np.ndarray
    res_nfw: np.ndarray
    res_mond: np.ndarray
    metrics_rft: Dict[str, float]
    metrics_nfw: Dict[str, float]
    metrics_mond: Dict[str, float]


def load_per_galaxy_map(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text())
    return {entry["name"]: entry for entry in payload["per_galaxy"]}, payload


def resolve_case_path(name: str) -> Path:
    candidates = [
        PROJECT_ROOT / "cases" / f"{name}.json",
        PROJECT_ROOT / "cases" / "sparc_all" / f"{name}.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Case file not found for {name}")


def predict_nfw(case: GalaxyCase, rho_s: float, r_s: float) -> np.ndarray:
    r = np.asarray(case.r_kpc, dtype=float)
    disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
    if case.v_baryon_bulge_kms is not None:
        bulge = np.asarray(case.v_baryon_bulge_kms, dtype=float)
    else:
        bulge = np.zeros_like(disk)
    v_baryon_sq = disk**2 + gas**2 + bulge**2
    v_dm_sq = _v_dm_squared(r, rho_s, r_s)
    v_pred_sq = v_baryon_sq + v_dm_sq
    return np.sqrt(np.clip(v_pred_sq, 0.0, None))


def prepare_gallery_entry(
    name: str,
    label: str,
    config: dict,
    rho_s: float,
    r_s: float,
) -> GalaxyCurves:
    case = load_case(resolve_case_path(name))
    mask = (
        (np.asarray(case.r_kpc, dtype=float) >= WINDOW_MIN)
        & (np.asarray(case.r_kpc, dtype=float) <= WINDOW_MAX)
        & (np.asarray(case.sigma_v_kms, dtype=float) > 0)
    )

    r_window = np.asarray(case.r_kpc, dtype=float)[mask]
    v_obs = np.asarray(case.v_obs_kms, dtype=float)[mask]
    sigma = np.asarray(case.sigma_v_kms, dtype=float)[mask]

    rft_pred = apply_v2_gated_tail(case, config["kernel"], config["tail"])["v_pred_kms"]
    v_rft = np.asarray(rft_pred, dtype=float)[mask]
    v_nfw = predict_nfw(case, rho_s, r_s)[mask]
    v_mond = mond_predict(case, a0_m_s2=1.2e-10, law="standard")[0][mask]

    # Fractional residuals (%)
    safe_obs = np.clip(v_obs, 1e-3, None)
    res_rft = 100.0 * (v_rft - v_obs) / safe_obs
    res_nfw = 100.0 * (v_nfw - v_obs) / safe_obs
    res_mond = 100.0 * (v_mond - v_obs) / safe_obs

    metrics_rft = compute_metrics(v_obs, v_rft, pass_threshold=PREDICTIVE_THRESHOLD)
    metrics_nfw = compute_metrics(v_obs, v_nfw, pass_threshold=PREDICTIVE_THRESHOLD)
    metrics_mond = compute_metrics(v_obs, v_mond, pass_threshold=PREDICTIVE_THRESHOLD)

    return GalaxyCurves(
        name=name,
        label=label,
        radius=r_window,
        v_obs=v_obs,
        sigma_v=sigma,
        v_rft=v_rft,
        v_nfw=v_nfw,
        v_mond=v_mond,
        res_rft=res_rft,
        res_nfw=res_nfw,
        res_mond=res_mond,
        metrics_rft=metrics_rft,
        metrics_nfw=metrics_nfw,
        metrics_mond=metrics_mond,
    )


def select_gallery_examples(
    rft_per: Dict[str, dict],
    nfw_per: Dict[str, dict],
) -> List[Tuple[str, str]]:
    wins = sorted(
        [name for name, stats in rft_per.items() if stats["pass_20"] and not nfw_per[name]["pass_20"]],
        key=str.lower,
    )
    ties = sorted(
        [name for name, stats in rft_per.items() if stats["pass_20"] and nfw_per[name]["pass_20"]],
        key=str.lower,
    )
    near_miss = sorted(
        [name for name, stats in rft_per.items() if not stats["pass_20"] and nfw_per[name]["pass_20"]],
        key=str.lower,
    )

    if len(wins) < 2 or len(ties) < 2 or len(near_miss) < 2:
        raise RuntimeError("Insufficient galaxies in each category to build gallery")

    return [
        ("Win: RFT pass / NFW fail", wins[0]),
        ("Win: RFT pass / NFW fail", wins[1]),
        ("Both pass", ties[0]),
        ("Both pass", ties[1]),
        ("Near-miss: NFW pass", near_miss[0]),
        ("Near-miss: NFW pass", near_miss[1]),
    ]


def plot_gallery(entries: List[GalaxyCurves]) -> None:
    rows = 2
    cols = 3
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(
        rows * 2,
        cols,
        height_ratios=[3, 1] * rows,
        hspace=0.3,
        wspace=0.2,
    )

    legend_handles = None
    legend_labels = None

    for idx, entry in enumerate(entries):
        row = idx // cols
        col = idx % cols
        ax_vel = fig.add_subplot(gs[row * 2, col])
        ax_res = fig.add_subplot(gs[row * 2 + 1, col], sharex=ax_vel)

        # Velocity panel
        ax_vel.errorbar(
            entry.radius,
            entry.v_obs,
            yerr=entry.sigma_v,
            fmt="o",
            color=COLORS["obs"],
            markersize=4,
            capsize=2,
            label="Observed" if idx == 0 else None,
        )
        ax_vel.plot(entry.radius, entry.v_rft, color=COLORS["rft"], linewidth=2.2, label="RFT v2" if idx == 0 else None)
        ax_vel.plot(
            entry.radius,
            entry.v_nfw,
            color=COLORS["nfw"],
            linewidth=2.0,
            linestyle="--",
            label="NFW$_{\\text{global}}$" if idx == 0 else None,
        )
        ax_vel.plot(
            entry.radius,
            entry.v_mond,
            color=COLORS["mond"],
            linewidth=2.0,
            linestyle=":",
            label="MOND" if idx == 0 else None,
        )

        if idx == 0:
            legend_handles, legend_labels = ax_vel.get_legend_handles_labels()

        ax_vel.set_title(f"{entry.label}\n{entry.name}", fontsize=12, fontweight="bold")
        ax_vel.set_ylabel("Velocity [km/s]")
        ax_vel.grid(alpha=0.3, linestyle="--")

        text = (
            f"RFT RMS={entry.metrics_rft['rms_percent']:.1f}% "
            f"{'(pass)' if entry.metrics_rft['pass'] else '(fail)'}\n"
            f"NFW RMS={entry.metrics_nfw['rms_percent']:.1f}% "
            f"{'(pass)' if entry.metrics_nfw['pass'] else '(fail)'}\n"
            f"MOND RMS={entry.metrics_mond['rms_percent']:.1f}% "
            f"{'(pass)' if entry.metrics_mond['pass'] else '(fail)'}"
        )
        ax_vel.text(
            0.02,
            0.98,
            text,
            transform=ax_vel.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
        )

        if row == rows - 1:
            ax_res.set_xlabel("Radius [kpc]")
        ax_res.set_ylabel("Residual [%]")
        ax_res.axhline(0, color="gray", linewidth=1)
        ax_res.axhline(PREDICTIVE_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
        ax_res.axhline(-PREDICTIVE_THRESHOLD, color="gray", linestyle=":", linewidth=0.8)
        ax_res.plot(entry.radius, entry.res_rft, color=COLORS["rft"], linewidth=1.6)
        ax_res.plot(entry.radius, entry.res_nfw, color=COLORS["nfw"], linewidth=1.4, linestyle="--")
        ax_res.plot(entry.radius, entry.res_mond, color=COLORS["mond"], linewidth=1.4, linestyle=":")
        ax_res.grid(alpha=0.3, linestyle="--")
        ax_res.set_ylim(-80, 80)

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc="upper center", ncol=4, frameon=False, fontsize=11)

    fig.suptitle("Representative TEST Rotation Curves (Predictive k=0 Comparison)", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    out_pdf = PROJECT_ROOT / "paper" / "figs" / "fig5_rotation_gallery.pdf"
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"✅ Saved {out_pdf}")


def main() -> int:
    rft_per, _ = load_per_galaxy_map(PROJECT_ROOT / "results" / "v2.1_refine" / "v2.1_test_results.json")
    nfw_per, nfw_payload = load_per_galaxy_map(PROJECT_ROOT / "baselines" / "results" / "nfw_global_test_baseline.json")
    mond_per, _ = load_per_galaxy_map(PROJECT_ROOT / "baselines" / "results" / "mond_test_baseline.json")
    if set(rft_per.keys()) != set(nfw_per.keys()) or set(rft_per.keys()) != set(mond_per.keys()):
        raise RuntimeError("Mismatch in galaxy lists between solvers")

    rho_s = nfw_payload["global_params"]["rho_s_Msun_per_kpc3"]
    r_s = nfw_payload["global_params"]["r_s_kpc"]
    config = json.loads((PROJECT_ROOT / "config" / "global_rc_v2_frozen.json").read_text())

    gallery_plan = select_gallery_examples(rft_per, nfw_per)
    entries = [
        prepare_gallery_entry(name, label, config, rho_s, r_s)
        for label, name in gallery_plan
    ]

    plot_gallery(entries)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
