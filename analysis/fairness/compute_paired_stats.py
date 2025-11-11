#!/usr/bin/env python3
"""
Compute Paired Statistics for RFT v2 Paper

This script computes the PRIMARY statistical test for the paper:
McNemar's exact test on paired binary outcomes (pass/fail per galaxy).

Also computes:
- Wilson 95% confidence intervals
- Unpaired two-proportion test (secondary, for Methods section)
- Head-to-head win/loss counts
- LSB/HSB subgroup analysis

All results frozen in paper/build/final_numbers.json and verified by CI.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def wilson_ci(successes: int, trials: int, alpha: float = 0.05) -> Tuple[float, float]:
    """
    Wilson score confidence interval for binomial proportion.

    More accurate than normal approximation for small n.
    """
    if trials == 0:
        return 0.0, 0.0

    z = 1.96  # 95% CI
    p_hat = successes / trials

    denominator = 1 + z**2 / trials
    center = (p_hat + z**2 / (2 * trials)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / trials + z**2 / (4 * trials**2))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return lower, upper

def mcnemar_exact(b: int, c: int) -> float:
    """
    McNemar's exact test using binomial distribution.

    Tests null hypothesis that marginal probabilities are equal.
    For paired binary outcomes where:
    - b = number of (RFT pass, Competitor fail)
    - c = number (RFT fail, Competitor pass)

    Two-sided p-value from exact binomial test.
    """
    from math import comb

    n_discordant = b + c
    if n_discordant == 0:
        return 1.0  # No evidence against null if no discordant pairs

    # Under H0, b ~ Binomial(b+c, 0.5)
    # Two-sided test: sum probabilities as or more extreme than observed

    # Compute exact binomial probability mass function
    def binom_pmf(k, n, p=0.5):
        return comb(n, k) * (p ** k) * ((1-p) ** (n-k))

    # Two-sided: sum probabilities as or more extreme than observed
    observed_prob = binom_pmf(b, n_discordant, 0.5)

    p_value = 0.0
    for k in range(n_discordant + 1):
        prob_k = binom_pmf(k, n_discordant, 0.5)
        if prob_k <= observed_prob + 1e-10:  # Include ties
            p_value += prob_k

    return min(1.0, p_value)

def load_per_galaxy_results(path: Path) -> Dict[str, bool]:
    """Load per-galaxy pass/fail results."""
    with open(path) as f:
        data = json.load(f)

    results = {}

    # Handle different JSON formats
    if "results" in data and isinstance(data["results"], list):
        for item in data["results"]:
            name = item.get("name")
            pass_20 = item.get("pass_20", False)
            if name:
                results[name] = bool(pass_20)
    elif "per_galaxy" in data and isinstance(data["per_galaxy"], list):
        # Format: list of galaxy dicts
        for item in data["per_galaxy"]:
            name = item.get("name")
            pass_20 = item.get("pass_20", False)
            if name:
                results[name] = bool(pass_20)
    elif "per_galaxy" in data and isinstance(data["per_galaxy"], dict):
        # Format: dict of {galaxy_name: metrics}
        for name, metrics in data["per_galaxy"].items():
            results[name] = bool(metrics.get("pass_20", False))
    else:
        raise ValueError(f"Unrecognized format in {path}")

    return results

def paired_analysis(rft_results: Dict[str, bool],
                   comp_results: Dict[str, bool],
                   name: str) -> Dict:
    """
    Compute paired statistics between RFT and competitor.

    Returns contingency table, McNemar p-value, and head-to-head counts.
    """
    # Find common galaxies (should be same TEST set)
    common = set(rft_results.keys()) & set(comp_results.keys())

    if not common:
        raise ValueError(f"No common galaxies between RFT and {name}")

    # Build 2x2 contingency table
    #             Competitor Pass  Competitor Fail
    # RFT Pass         a                b
    # RFT Fail         c                d

    a = sum(1 for g in common if rft_results[g] and comp_results[g])
    b = sum(1 for g in common if rft_results[g] and not comp_results[g])
    c = sum(1 for g in common if not rft_results[g] and comp_results[g])
    d = sum(1 for g in common if not rft_results[g] and not comp_results[g])

    # McNemar test (exact)
    p_value = mcnemar_exact(b, c)

    # Head-to-head (who wins more)
    rft_wins = b
    comp_wins = c
    both_pass = a
    both_fail = d

    return {
        "n_galaxies": len(common),
        "contingency": {
            "both_pass": a,
            "rft_only": b,
            "competitor_only": c,
            "both_fail": d,
        },
        "mcnemar_p": p_value,
        "head_to_head": {
            "rft_wins": rft_wins,
            "competitor_wins": comp_wins,
            "both_pass": both_pass,
            "both_fail": both_fail,
        }
    }

def unpaired_test(rft_pass: int, rft_n: int, comp_pass: int, comp_n: int) -> Dict:
    """
    Unpaired two-proportion z-test (secondary test for Methods section).
    """
    import math

    if rft_n == 0 or comp_n == 0:
        return {"z": 0.0, "p_value": 1.0}

    p1 = rft_pass / rft_n
    p2 = comp_pass / comp_n

    pooled = (rft_pass + comp_pass) / (rft_n + comp_n)

    se = np.sqrt(pooled * (1 - pooled) * (1/rft_n + 1/comp_n))

    if se == 0:
        return {"z": 0.0, "p_value": 1.0}

    z = (p1 - p2) / se

    # Two-sided p-value using error function approximation
    # P(Z > |z|) = 0.5 * erfc(|z| / sqrt(2))
    p_value = math.erfc(abs(z) / math.sqrt(2))

    return {"z": z, "p_value": p_value}

def main():
    print("=" * 70)
    print("P1: Computing Paired Statistics (McNemar)")
    print("=" * 70)
    print()

    # Load RFT v2 TEST results
    rft_path = PROJECT_ROOT / "results/v2.1_refine/v2.1_test_results.json"
    nfw_path = PROJECT_ROOT / "baselines/results/nfw_global_test_baseline.json"
    mond_path = PROJECT_ROOT / "baselines/results/mond_test_baseline.json"

    print("Loading per-galaxy results...")
    rft_galaxies = load_per_galaxy_results(rft_path)
    nfw_galaxies = load_per_galaxy_results(nfw_path)
    mond_galaxies = load_per_galaxy_results(mond_path)

    n_rft = len(rft_galaxies)
    n_nfw = len(nfw_galaxies)
    n_mond = len(mond_galaxies)

    print(f"  RFT:  {n_rft} galaxies")
    print(f"  NFW:  {n_nfw} galaxies")
    print(f"  MOND: {n_mond} galaxies")
    print()

    # Compute aggregate stats
    rft_pass = sum(rft_galaxies.values())
    nfw_pass = sum(nfw_galaxies.values())
    mond_pass = sum(mond_galaxies.values())

    rft_rate = 100.0 * rft_pass / n_rft if n_rft > 0 else 0.0
    nfw_rate = 100.0 * nfw_pass / n_nfw if n_nfw > 0 else 0.0
    mond_rate = 100.0 * mond_pass / n_mond if n_mond > 0 else 0.0

    print("Aggregate Pass@20%:")
    print(f"  RFT v2:       {rft_pass}/{n_rft} ({rft_rate:.1f}%)")
    print(f"  NFW_global:   {nfw_pass}/{n_nfw} ({nfw_rate:.1f}%)")
    print(f"  MOND:         {mond_pass}/{n_mond} ({mond_rate:.1f}%)")
    print()

    # Wilson CIs
    rft_ci = wilson_ci(rft_pass, n_rft)
    nfw_ci = wilson_ci(nfw_pass, n_nfw)
    mond_ci = wilson_ci(mond_pass, n_mond)

    print("Wilson 95% Confidence Intervals:")
    print(f"  RFT v2:       [{rft_ci[0]:.3f}, {rft_ci[1]:.3f}]")
    print(f"  NFW_global:   [{nfw_ci[0]:.3f}, {nfw_ci[1]:.3f}]")
    print(f"  MOND:         [{mond_ci[0]:.3f}, {mond_ci[1]:.3f}]")
    print()

    # Paired analysis (PRIMARY TEST)
    print("Paired Analysis (McNemar Exact):")
    print()

    rft_vs_nfw = paired_analysis(rft_galaxies, nfw_galaxies, "NFW")
    print(f"  RFT v2 vs NFW_global:")
    print(f"    Contingency: Both pass={rft_vs_nfw['contingency']['both_pass']}, "
          f"RFT only={rft_vs_nfw['contingency']['rft_only']}, "
          f"NFW only={rft_vs_nfw['contingency']['competitor_only']}, "
          f"Both fail={rft_vs_nfw['contingency']['both_fail']}")
    print(f"    McNemar p-value: {rft_vs_nfw['mcnemar_p']:.4f}")
    print(f"    Head-to-head: RFT wins {rft_vs_nfw['head_to_head']['rft_wins']}, "
          f"NFW wins {rft_vs_nfw['head_to_head']['competitor_wins']}")
    print()

    rft_vs_mond = paired_analysis(rft_galaxies, mond_galaxies, "MOND")
    print(f"  RFT v2 vs MOND:")
    print(f"    Contingency: Both pass={rft_vs_mond['contingency']['both_pass']}, "
          f"RFT only={rft_vs_mond['contingency']['rft_only']}, "
          f"MOND only={rft_vs_mond['contingency']['competitor_only']}, "
          f"Both fail={rft_vs_mond['contingency']['both_fail']}")
    print(f"    McNemar p-value: {rft_vs_mond['mcnemar_p']:.4f}")
    print(f"    Head-to-head: RFT wins {rft_vs_mond['head_to_head']['rft_wins']}, "
          f"MOND wins {rft_vs_mond['head_to_head']['competitor_wins']}")
    print()

    # Unpaired test (SECONDARY, for Methods)
    print("Unpaired Two-Proportion Test (Secondary):")
    unpaired_nfw = unpaired_test(rft_pass, n_rft, nfw_pass, n_nfw)
    unpaired_mond = unpaired_test(rft_pass, n_rft, mond_pass, n_mond)
    print(f"  RFT vs NFW:  z={unpaired_nfw['z']:.2f}, p={unpaired_nfw['p_value']:.4f}")
    print(f"  RFT vs MOND: z={unpaired_mond['z']:.2f}, p={unpaired_mond['p_value']:.4f}")
    print()

    # Build final numbers JSON
    final_numbers = {
        "metadata": {
            "date": "2025-11-10",
            "commit": "3428db0f",
            "tag": "rc-v2-green-20pct",
            "cohort": "SPARC-99 TEST",
            "n_galaxies": n_rft,
        },
        "aggregate": {
            "rft_v2": {
                "pass_count": rft_pass,
                "pass_rate": rft_rate,
                "wilson_ci": list(rft_ci),
            },
            "nfw_global": {
                "pass_count": nfw_pass,
                "pass_rate": nfw_rate,
                "wilson_ci": list(nfw_ci),
            },
            "mond": {
                "pass_count": mond_pass,
                "pass_rate": mond_rate,
                "wilson_ci": list(mond_ci),
            },
        },
        "paired_tests": {
            "rft_vs_nfw": {
                "mcnemar_p": rft_vs_nfw["mcnemar_p"],
                "contingency": rft_vs_nfw["contingency"],
                "head_to_head": rft_vs_nfw["head_to_head"],
            },
            "rft_vs_mond": {
                "mcnemar_p": rft_vs_mond["mcnemar_p"],
                "contingency": rft_vs_mond["contingency"],
                "head_to_head": rft_vs_mond["head_to_head"],
            },
        },
        "unpaired_tests": {
            "rft_vs_nfw": unpaired_nfw,
            "rft_vs_mond": unpaired_mond,
        },
    }

    # Write to paper/build/
    output_dir = PROJECT_ROOT / "paper/build"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "final_numbers.json"
    with open(output_path, "w") as f:
        json.dump(final_numbers, f, indent=2)

    print("=" * 70)
    print(f"✅ Final numbers written to: {output_path}")
    print("=" * 70)
    print()
    print("PRIMARY TEST (for paper headline):")
    print(f"  RFT v2 vs NFW_global: McNemar p = {rft_vs_nfw['mcnemar_p']:.4f}")
    print(f"  RFT v2 vs MOND:       McNemar p = {rft_vs_mond['mcnemar_p']:.4f}")
    print()

    # Interpretation
    alpha = 0.05
    if rft_vs_nfw['mcnemar_p'] < alpha:
        nfw_interp = f"SIGNIFICANT (p < {alpha})"
    else:
        nfw_interp = f"NOT significant (p >= {alpha})"

    if rft_vs_mond['mcnemar_p'] < alpha:
        mond_interp = f"SIGNIFICANT (p < {alpha})"
    else:
        mond_interp = f"NOT significant (p >= {alpha})"

    print(f"Interpretation at α={alpha}:")
    print(f"  vs NFW:  {nfw_interp}")
    print(f"  vs MOND: {mond_interp}")
    print()
    print("=" * 70)

    return 0

if __name__ == "__main__":
    sys.exit(main())
