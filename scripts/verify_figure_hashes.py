#!/usr/bin/env python3
"""
Gate P2: Figure Hash Lock

Verifies that figure PDFs match their frozen SHA256 hashes.
This prevents accidental figure regeneration that could drift from the published results.
"""

import hashlib
import json
import sys
from pathlib import Path

# Expected figure hashes (frozen at tag rc-v2-green-20pct)
EXPECTED_HASHES = {
    "fig1_overview.pdf": None,  # Will be computed and stored
    "fig2_mcnemar.pdf": None,
    "fig3_lsb_hsb.pdf": None,
    "fig4_stability.pdf": None,
    "fig5_rotation_gallery.pdf": None,
    "fig6_ablations.pdf": None,
}

def compute_sha256(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    figs_dir = repo_root / "paper" / "figs"
    hash_file = repo_root / "paper" / "build" / "figure_hashes.json"

    print("=" * 70)
    print("Gate P2: Verifying Figure Hashes")
    print("=" * 70)
    print()

    # Load expected hashes if they exist
    if hash_file.exists():
        with open(hash_file) as f:
            expected = json.load(f)
        mode = "verify"
    else:
        # First run: compute and store hashes
        expected = {}
        mode = "freeze"

    current_hashes = {}
    all_pass = True

    # Check required figures
    for fig_name in EXPECTED_HASHES.keys():
        fig_path = figs_dir / fig_name

        if not fig_path.exists():
            print(f"❌ Missing: {fig_name}")
            all_pass = False
            continue

        current_hash = compute_sha256(fig_path)
        current_hashes[fig_name] = current_hash

        if mode == "verify":
            expected_hash = expected.get(fig_name)
            if expected_hash is None:
                print(f"⚠️  {fig_name}: Not in hash file (new figure?)")
                print(f"   SHA256: {current_hash}")
            elif current_hash == expected_hash:
                print(f"✅ {fig_name}: {current_hash[:16]}...")
            else:
                print(f"❌ {fig_name}: HASH MISMATCH")
                print(f"   Expected: {expected_hash}")
                print(f"   Got:      {current_hash}")
                all_pass = False
        else:
            print(f"🔒 {fig_name}: {current_hash[:16]}... (frozen)")

    print()

    if mode == "freeze":
        # Save hashes for future verification
        hash_file.parent.mkdir(parents=True, exist_ok=True)
        with open(hash_file, "w") as f:
            json.dump(current_hashes, f, indent=2, sort_keys=True)
        print(f"✅ Figure hashes frozen to {hash_file}")
        print()
        print("Add this file to git:")
        print(f"  git add {hash_file}")
        print()
        return 0

    if all_pass:
        print("=" * 70)
        print("✅ GATE P2 PASS: All figure hashes match frozen publication")
        print("=" * 70)
        return 0
    else:
        print("=" * 70)
        print("❌ GATE P2 FAIL: Figure hash mismatch detected")
        print("=" * 70)
        print()
        print("Figures have changed from the frozen publication version.")
        print("If this is intentional, update the hash file:")
        print(f"  rm {hash_file}")
        print("  python3 scripts/verify_figure_hashes.py")
        print()
        return 1

if __name__ == "__main__":
    sys.exit(main())
