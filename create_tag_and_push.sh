#!/bin/bash
# Script to create git tag and push to GitHub

cd /tmp/rft-v2-github

# Create the tag with proper multi-line message
git tag -a rc-v2-green-20pct -m "RFT v2 Repro Pack 1.0

Release rc-v2-green-20pct: First public release of RFT v2 galaxy rotation validation.

Performance:
- RFT v2: 58.8% pass@20% on TEST (20/34 galaxies)
- NFW (global): 52.9% (18/34)
- MOND: 23.5% (8/34)

Statistical Tests:
- McNemar vs NFW: p=0.69 (competitive, not significant)
- McNemar vs MOND: p=0.004 (significant improvement)

LSB Dominance:
- RFT v2: 66.7% on LSB galaxies (4/6)
- NFW/MOND: 0% on LSB galaxies (0/6)

Reproducibility:
- One-click RUNME.sh verification
- Frozen configs and results
- Baseline + hash locks
- MIT licensed

Commit: 3428db0f
Date: 2025-11-10"

echo "Tag created successfully!"
echo ""
echo "Now you need to:"
echo "1. Authenticate with GitHub CLI:"
echo "   gh auth login"
echo ""
echo "2. Add remote and push:"
echo "   git remote add origin https://github.com/rft-cosmology/rft-v2-galaxy-rotations.git"
echo "   git push -u origin main"
echo "   git push origin rc-v2-green-20pct"
