#!/usr/bin/env bash
set -euo pipefail

echo "=========================================================================="
echo "                  RFT v2 Galaxy Rotations - Verification"
echo "=========================================================================="
echo ""

# Install dependencies if needed
if ! python -c "import numpy, scipy, pandas" &>/dev/null; then
  echo "→ Installing dependencies..."
  python -m pip install -r requirements.txt -q
fi

# Gate 0: Baseline lock
echo "→ Running baseline lock (Gate 0)..."
python scripts/audit_baselines.py || {
  echo ""
  echo "❌ Baseline lock failed. Cannot proceed."
  exit 2
}

echo ""
echo "→ Generating fairness pack..."
python scripts/generate_fairness_pack.py \
  --rft results/v2.1_refine/v2.1_test_results.json \
  --nfw baselines/results/nfw_global_test_baseline.json \
  --mond baselines/results/mond_test_baseline.json \
  --out app/static/data/v2_fairness_pack.json

echo ""
echo "→ Generating stability analysis..."
python scripts/generate_stability_analysis.py \
  --frozen-config config/global_rc_v2_frozen.json \
  --out app/static/data/v2_stability.json

echo ""
echo "→ Verifying headline numbers..."
python scripts/verify_numbers.py

echo ""
echo "=========================================================================="
echo "✅ ALL CHECKS PASSED"
echo "=========================================================================="
echo ""
echo "Results:"
echo "  • Baseline lock verified (Gate 0)"
echo "  • Fairness pack generated"
echo "  • Stability analysis generated"
echo "  • Headline numbers match publication"
echo ""
echo "Generated files:"
echo "  • app/static/data/v2_fairness_pack.json"
echo "  • app/static/data/v2_stability.json"
echo ""
echo "=========================================================================="
