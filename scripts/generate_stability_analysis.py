#!/usr/bin/env python3
"""
Generate ±10% stability data for the frozen RFT v2 configuration.

This script perturbs each tail parameter by ±10%, evaluates the TEST cohort,
and stores the resulting pass@20 rates and median RMS values.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metrics.rc_metrics import compute_metrics
from solver.rft_v2_gated_tail import apply_v2_gated_tail
from sparc_rft.case import load_case
CONFIG_PATH = PROJECT_ROOT / "config/global_rc_v2_frozen.json"
MANIFEST_PATH = PROJECT_ROOT / "cases/SP99-TEST.manifest.txt"
OUTPUT_PATH = PROJECT_ROOT / "data/refs/v2_stability.json"
BASELINE_RESULTS_PATH = PROJECT_ROOT / "results/v2_frozen/test_results.json"

WINDOW_MIN = 1.0
WINDOW_MAX = 30.0
MIN_POINTS = 3

PERTURB_PARAMS = [
    "A0_kms2_per_kpc",
    "alpha",
    "g_star_kms2_per_kpc",
    "gamma",
    "r_turn_kpc",
    "p",
]


@dataclass
class CasePayload:
    name: str
    radii: np.ndarray
    v_obs: np.ndarray
    sigma: np.ndarray
    case_obj: object


def load_cases(manifest_path: Path) -> List[CasePayload]:
    cases: List[CasePayload] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            case_path = (manifest_path.parent / stripped).resolve()
            case = load_case(str(case_path))
            cases.append(
                CasePayload(
                    name=case.name,
                    radii=np.asarray(case.r_kpc, dtype=float),
                    v_obs=np.asarray(case.v_obs_kms, dtype=float),
                    sigma=np.asarray(case.sigma_v_kms, dtype=float),
                    case_obj=case,
                )
            )
    return cases


def evaluate_tail_config(
    cases: List[CasePayload],
    kernel_cfg: Dict,
    tail_cfg: Dict,
) -> Tuple[int, float, float]:
    pass_count = 0
    rms_values: List[float] = []
    evaluated_cases = 0

    for payload in cases:
        result = apply_v2_gated_tail(payload.case_obj, kernel_cfg, tail_cfg)
        radii = np.asarray(result["r_kpc"], dtype=float)
        v_pred = np.asarray(result["v_pred_kms"], dtype=float)
        mask = (
            (radii >= WINDOW_MIN)
            & (radii <= WINDOW_MAX)
            & (payload.sigma > 0)
        )
        if mask.sum() < MIN_POINTS:
            continue
        evaluated_cases += 1
        metrics = compute_metrics(
            v_obs_kms=payload.v_obs[mask],
            v_pred_kms=v_pred[mask],
        )
        rms_values.append(float(metrics["rms_percent"]))
        if metrics["rms_percent"] <= 20.0:
            pass_count += 1

    n = evaluated_cases or 1
    pass_rate = 100.0 * pass_count / n
    median_rms = float(np.median(rms_values)) if rms_values else 0.0
    return pass_count, pass_rate, median_rms


def build_tail_variant(base_tail: Dict, param: str, scale: float) -> Dict:
    variant = copy.deepcopy(base_tail)
    value = float(variant[param])
    variant[param] = max(value * scale, 1e-9)
    return variant


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    kernel_cfg = config["kernel"]
    base_tail = config["tail"]

    cases = load_cases(MANIFEST_PATH)

    baseline_pass_count, baseline_rate, baseline_median = evaluate_tail_config(
        cases, kernel_cfg, base_tail
    )

    try:
        frozen_summary = json.loads(BASELINE_RESULTS_PATH.read_text(encoding="utf-8"))
        baseline_rate = frozen_summary.get("pass_20_rate", baseline_rate)
        baseline_pass_count = frozen_summary.get("pass_20_count", baseline_pass_count)
        baseline_median = frozen_summary.get("rms_median", baseline_median)
    except FileNotFoundError:
        pass

    perturbations = []
    for param in PERTURB_PARAMS:
        for delta in (-0.1, 0.1):
            scale = 1.0 + delta
            tail_variant = build_tail_variant(base_tail, param, scale)
            pass_count, pass_rate, median_rms = evaluate_tail_config(
                cases, kernel_cfg, tail_variant
            )
            perturbations.append(
                {
                    "param": param,
                    "delta": f"{delta * 100:+.0f}%",
                    "pass_20_count": pass_count,
                    "pass_20_rate": pass_rate,
                    "median_rms_pct": median_rms,
                }
            )

    payload = {
        "baseline": {
            "pass_20_count": baseline_pass_count,
            "pass_20_rate": baseline_rate,
            "median_rms_pct": baseline_median,
            "n_galaxies": len(cases),
        },
        "perturbations": perturbations,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[stability] Wrote {OUTPUT_PATH} with {len(perturbations)} entries.")


if __name__ == "__main__":
    main()
