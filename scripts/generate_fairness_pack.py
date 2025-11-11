#!/usr/bin/env python3
"""
Generate the fairness pack JSON used by the galaxy rotation results page.

This script aggregates RFT v2, NFW, MOND (and Newtonian) performance on the
SPARC TEST cohort, computes head-to-head tallies, two-proportion z-tests,
Wilson confidence intervals, and the LSB/HSB split based on vmax < 120 km/s.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sparc_rft.case import load_case
COMPARISON_PATH = PROJECT_ROOT / "reports/comparisons/theory_comparison_test.json"
MANIFEST_PATH = PROJECT_ROOT / "cases/SP99-TEST.manifest.txt"
OUTPUT_PATH = PROJECT_ROOT / "data/refs/v2_fairness_pack.json"

THEORY_KEY_MAP = {
    "RFT_v2": "rft_v2",
    "NFW_halo": "nfw_halo",
    "MOND": "mond",
    "Newtonian": "newtonian",
}

DISPLAY_NAMES = {
    "rft_v2": "RFT v2",
    "nfw_halo": "NFW halo",
    "mond": "MOND",
    "newtonian": "Newtonian",
}

LSB_THRESHOLD_KMS = 120.0


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def two_proportion_ztest(k1: int, n1: int, k2: int, n2: int) -> Tuple[float, float]:
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    pooled = (k1 + k2) / (n1 + n2)
    denom = math.sqrt(pooled * (1 - pooled) * ((1 / n1) + (1 / n2)))
    if denom == 0:
        return 0.0, 1.0
    z = (p1 - p2) / denom
    p_value = 2 * (1 - _normal_cdf(abs(z)))
    return z, p_value


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p_hat = k / n
    denom = 1 + (z**2) / n
    center = (p_hat + (z**2) / (2 * n)) / denom
    half_width = z * math.sqrt((p_hat * (1 - p_hat) / n) + (z**2) / (4 * n**2)) / denom
    low = max(0.0, center - half_width)
    high = min(1.0, center + half_width)
    return low, high


def load_manifest_cases(manifest_path: Path) -> Dict[str, float]:
    vmax_lookup: Dict[str, float] = {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            case_path = (manifest_path.parent / stripped).resolve()
            case = load_case(str(case_path))
            observed = [float(v) for v in case.v_obs_kms if v is not None]
            vmax = max(observed) if observed else 0.0
            vmax_lookup[case.name] = vmax
    return vmax_lookup


def load_theory_rows(path: Path) -> Dict[str, dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    theories: Dict[str, dict] = {}
    for entry in payload.get("theories", []):
        mapped = THEORY_KEY_MAP.get(entry.get("theory"))
        if not mapped:
            continue
        cases = {row["name"]: row for row in entry.get("results", [])}
        entry["results_by_name"] = cases
        theories[mapped] = entry
    missing = [name for name in ("rft_v2", "nfw_halo", "mond") if name not in theories]
    if missing:
        raise SystemExit(f"Missing theories in {path}: {', '.join(missing)}")
    return theories


def build_head_to_head(rft_cases: Dict[str, dict], other_cases: Dict[str, dict]) -> dict:
    rft_wins: List[str] = []
    other_wins: List[str] = []
    ties: List[str] = []
    for name, rft_row in rft_cases.items():
        other_row = other_cases.get(name)
        if other_row is None:
            continue
        rft_pass = bool(rft_row.get("pass_20"))
        other_pass = bool(other_row.get("pass_20"))
        if rft_pass and not other_pass:
            rft_wins.append(name)
        elif other_pass and not rft_pass:
            other_wins.append(name)
        else:
            ties.append(name)
    return {
        "rft_wins": sorted(rft_wins),
        "competitor_wins": sorted(other_wins),
        "ties": sorted(ties),
        "rft_win_count": len(rft_wins),
        "competitor_win_count": len(other_wins),
        "tie_count": len(ties),
    }


def build_lsb_hsb_splits(
    vmax_lookup: Dict[str, float],
    theory_results: Dict[str, dict],
) -> dict:
    buckets = {"lsb": {"names": [], "n": 0}, "hsb": {"names": [], "n": 0}}
    for name, vmax in vmax_lookup.items():
        bucket = "lsb" if vmax < LSB_THRESHOLD_KMS else "hsb"
        buckets[bucket]["names"].append(name)
        buckets[bucket]["n"] += 1

    split_payload = {
        "threshold_kms": LSB_THRESHOLD_KMS,
        "lsb": {},
        "hsb": {},
    }

    for bucket_name, bucket_data in buckets.items():
        names = sorted(bucket_data["names"])
        n_bucket = bucket_data["n"]
        split_payload[bucket_name]["n"] = n_bucket
        split_payload[bucket_name]["names"] = names
        for key, theory in theory_results.items():
            case_rows = theory["results_by_name"]
            passes = sum(1 for nm in names if case_rows.get(nm, {}).get("pass_20"))
            rate = (passes / n_bucket * 100.0) if n_bucket else 0.0
            split_payload[bucket_name][key] = {
                "pass_count": passes,
                "rate": rate,
            }
    return split_payload


def main() -> None:
    theories = load_theory_rows(COMPARISON_PATH)
    vmax_lookup = load_manifest_cases(MANIFEST_PATH)

    rft_entry = theories["rft_v2"]
    nfw_entry = theories["nfw_halo"]
    mond_entry = theories["mond"]

    n_total = rft_entry["n_cases"]

    head_to_head = {
        "rft_vs_nfw": build_head_to_head(rft_entry["results_by_name"], nfw_entry["results_by_name"]),
        "rft_vs_mond": build_head_to_head(rft_entry["results_by_name"], mond_entry["results_by_name"]),
    }

    proportion_tests = {
        "rft_vs_nfw": dict(
            zip(
                ("z_score", "p_value"),
                two_proportion_ztest(
                    rft_entry["pass_20_count"],
                    rft_entry["n_cases"],
                    nfw_entry["pass_20_count"],
                    nfw_entry["n_cases"],
                ),
            )
        ),
        "rft_vs_mond": dict(
            zip(
                ("z_score", "p_value"),
                two_proportion_ztest(
                    rft_entry["pass_20_count"],
                    rft_entry["n_cases"],
                    mond_entry["pass_20_count"],
                    mond_entry["n_cases"],
                ),
            )
        ),
    }

    theory_totals = {}
    for key, entry in theories.items():
        ci_low, ci_high = wilson_ci(entry["pass_20_count"], entry["n_cases"])
        theory_totals[key] = {
            "display": DISPLAY_NAMES.get(key, key),
            "n": entry["n_cases"],
            "pass_20_count": entry["pass_20_count"],
            "pass_20_rate": entry["pass_20_rate"],
            "wilson_ci": [ci_low, ci_high],
        }

    split_payload = build_lsb_hsb_splits(vmax_lookup, theories)

    payload = {
        "metadata": {
            "cohort": "SP99-TEST",
            "n_galaxies": n_total,
        },
        "theories": theory_totals,
        "head_to_head": head_to_head,
        "proportion_tests": proportion_tests,
        "lsb_hsb_split": split_payload,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[fairness-pack] Wrote {OUTPUT_PATH} (n={n_total})")


if __name__ == "__main__":
    main()
