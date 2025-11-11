#!/usr/bin/env python3
"""
Verify Final Numbers Hash (Gate P1)

Ensures the frozen final_numbers.json has not drifted from publication values.
This prevents accidental changes to p-values, contingency tables, or other
statistical results that will be reported in the paper.
"""

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# FROZEN HASH (commit 3428db0f, 2025-11-10)
EXPECTED_HASH = "d935dad7070d371578cdfacdaf6f6a62921ef5943ff8a0884e09c4b321c7bb1e"

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            sha256.update(chunk)
    return sha256.hexdigest()

def main():
    print("=" * 70)
    print("Gate P1: Verifying Final Numbers Hash")
    print("=" * 70)
    print()

    final_numbers_path = PROJECT_ROOT / "paper/build/final_numbers.json"

    if not final_numbers_path.exists():
        print(f"❌ FAIL: {final_numbers_path} does not exist")
        print()
        print("Run: python analysis/fairness/compute_paired_stats.py")
        return 2

    actual_hash = compute_sha256(final_numbers_path)

    print(f"Expected SHA256: {EXPECTED_HASH}")
    print(f"Actual SHA256:   {actual_hash}")
    print()

    if actual_hash == EXPECTED_HASH:
        print("✅ PASS: Final numbers hash matches frozen publication values")
        print()
        print("=" * 70)
        return 0
    else:
        print("❌ FAIL: Final numbers have drifted from frozen values!")
        print()
        print("This indicates one of the following:")
        print("  1. Input data files have changed")
        print("  2. Statistical computation has changed")
        print("  3. Intentional update (update EXPECTED_HASH if so)")
        print()
        print("If this is intentional, update EXPECTED_HASH in this script.")
        print("Otherwise, investigate the source of the drift.")
        print()
        print("=" * 70)
        return 2

if __name__ == "__main__":
    sys.exit(main())
