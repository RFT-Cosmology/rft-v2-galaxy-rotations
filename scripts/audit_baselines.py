#!/usr/bin/env python3
"""
Baseline Consistency Lock (Gate 0)

Refuses to proceed if baseline numbers drift from frozen publication values.
This prevents accidental misreporting when code or data changes.

Expected values (frozen 2025-11-10):
  - RFT v2 TEST: 20/34 pass@20% (58.8%)
  - NFW_global TEST: 18/34 pass@20% (52.9%) [k=0, fair comparison]
  - MOND TEST: 8/34 pass@20% (23.5%) [k=0, fair comparison]

Supplementary (not gated, for reference):
  - NFW_fitted: 28/34 (82.4%) [k=2 per galaxy, unfair but documented]
  - NFW_halo variant: 10/34 (29.4%) [different implementation]
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Frozen baseline paths
PATHS = {
    "rft": PROJECT_ROOT / "results/v2.1_refine/v2.1_test_results.json",
    "nfw_global": PROJECT_ROOT / "baselines/results/nfw_global_test_baseline.json",
    "mond": PROJECT_ROOT / "baselines/results/mond_test_baseline.json",
}

# Expected values (frozen publication)
EXPECTED = {
    "rft.pass_20": (20, 34, 58.8),
    "nfw_global.pass_20": (18, 34, 52.9),
    "mond.pass_20": (8, 34, 23.5),
}

def load_pass20(path):
    """Load pass@20% count and total from various JSON formats."""
    if not path.exists():
        raise FileNotFoundError(f"{path}")

    with open(path) as f:
        data = json.load(f)

    # Format 1: Direct counts
    if "pass_20_count" in data and ("n_cases" in data or "n_galaxies" in data):
        count = data["pass_20_count"]
        total = data.get("n_cases") or data.get("n_galaxies")
        return count, total

    # Format 2: Per-galaxy results
    if "results" in data and isinstance(data["results"], list):
        passes = sum(1 for r in data["results"] if r.get("pass_20"))
        return passes, len(data["results"])

    # Format 3: Theories array (from comparison)
    if "theories" in data:
        # This is a comparison file - skip it
        raise ValueError(f"Comparison file, not individual results: {path}")

    raise ValueError(f"Unrecognized format for {path}")

def main():
    print("=" * 70)
    print("Gate 0: Baseline Consistency Lock")
    print("=" * 70)
    print()

    errors = []
    counts = {}

    # Load all baselines
    for key, path in PATHS.items():
        try:
            counts[key] = load_pass20(path)
            count, total = counts[key]
            rate = 100.0 * count / total if total > 0 else 0.0
            print(f"  {key:15s} {count:2d}/{total} ({rate:5.1f}%)")
        except Exception as e:
            print(f"  {key:15s} ❌ ERROR: {e}")
            errors.append(f"{key}: {e}")

    print()

    # Verify against expected
    for tag, (exp_count, exp_total, exp_rate) in EXPECTED.items():
        key = tag.split(".")[0]
        if key not in counts:
            continue

        act_count, act_total = counts[key]

        if act_count != exp_count or act_total != exp_total:
            err = f"{tag}: expected {exp_count}/{exp_total}, got {act_count}/{act_total}"
            print(f"❌ {err}")
            errors.append(err)
        else:
            print(f"✅ {tag}: {act_count}/{act_total} (matches frozen)")

    print()
    print("=" * 70)

    if errors:
        print("❌ BASELINE LOCK FAILED")
        print()
        print("Baseline numbers have drifted from frozen publication values.")
        print("This indicates:")
        print("  • Code changed (revert or update freeze)")
        print("  • Data changed (revert or document)")
        print("  • JSON path wrong (fix PATHS)")
        print()
        print("Errors:")
        for err in errors:
            print(f"  • {err}")
        print()
        print("=" * 70)
        return 2

    print("✅ BASELINE LOCK VERIFIED")
    print()
    print("All frozen numbers match. Safe to proceed with:")
    print("  • Website deployment")
    print("  • arXiv submission")
    print("  • GitHub release")
    print("=" * 70)
    return 0

if __name__ == "__main__":
    sys.exit(main())
