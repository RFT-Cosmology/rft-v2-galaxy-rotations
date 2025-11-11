#!/usr/bin/env python3
"""
Verify Headline Numbers

Prints the main results table for human verification.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def main():
    print("=" * 70)
    print("RFT v2 Galaxy Rotations - Headline Numbers")
    print("=" * 70)
    print()

    # Load comparison or individual results
    comparison_path = PROJECT_ROOT / "results/v2.1_refine/v2.1_test_results.json"
    nfw_path = PROJECT_ROOT / "baselines/results/nfw_global_test_baseline.json"
    mond_path = PROJECT_ROOT / "baselines/results/mond_test_baseline.json"

    with open(comparison_path) as f:
        rft = json.load(f)
    with open(nfw_path) as f:
        nfw = json.load(f)
    with open(mond_path) as f:
        mond = json.load(f)

    rft_count = rft.get("pass_20_count", 0)
    rft_total = rft.get("n_cases") or rft.get("n_galaxies", 34)
    rft_rate = 100.0 * rft_count / rft_total if rft_total > 0 else 0.0

    nfw_count = nfw.get("pass_20_count", 0)
    nfw_total = nfw.get("n_cases") or nfw.get("n_galaxies", 34)
    nfw_rate = 100.0 * nfw_count / nfw_total if nfw_total > 0 else 0.0

    mond_count = mond.get("pass_20_count", 0)
    mond_total = mond.get("n_cases") or mond.get("n_galaxies", 34)
    mond_rate = 100.0 * mond_count / mond_total if mond_total > 0 else 0.0

    print(f"{'Model':<20s} {'Pass@20%':<15s} {'Total Params':<15s}")
    print("-" * 70)
    print(f"{'RFT v2':<20s} {rft_count}/{rft_total} ({rft_rate:5.1f}%)  6 global (k=0)")
    print(f"{'NFW (global)':<20s} {nfw_count}/{nfw_total} ({nfw_rate:5.1f}%)  2 global (k=0)")
    print(f"{'MOND':<20s} {mond_count}/{mond_total} ({mond_rate:5.1f}%)  1 global (k=0)")
    print()

    # Load fairness if available
    fairness_path = PROJECT_ROOT / "app/static/data/v2_fairness_pack.json"
    if fairness_path.exists():
        with open(fairness_path) as f:
            fairness = json.load(f)

        rft_vs_nfw = fairness.get("proportion_tests", {}).get("rft_vs_nfw", {})
        print("Statistical Tests:")
        print(f"  RFT vs NFW:  z={rft_vs_nfw.get('z_score', 0):.2f}, p={rft_vs_nfw.get('p_value', 1):.4f}")

        rft_vs_mond = fairness.get("proportion_tests", {}).get("rft_vs_mond", {})
        print(f"  RFT vs MOND: z={rft_vs_mond.get('z_score', 0):.2f}, p={rft_vs_mond.get('p_value', 1):.4f}")
        print()

    print("=" * 70)
    print("✅ Headline numbers printed successfully")
    print("=" * 70)

if __name__ == "__main__":
    main()
