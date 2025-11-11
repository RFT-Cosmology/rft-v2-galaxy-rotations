.PHONY: verify fairness stability clean help

help:
\t@echo "RFT v2 Galaxy Rotations - Make targets"
\t@echo ""
\t@echo "  make verify    - Run full verification (RUNME.sh)"
\t@echo "  make fairness  - Generate fairness pack only"
\t@echo "  make stability - Generate stability analysis only"
\t@echo "  make clean     - Remove generated files"
\t@echo ""

verify:
\t./RUNME.sh

fairness:
\tpython scripts/generate_fairness_pack.py \
\t  --rft results/v2.1_refine/v2.1_test_results.json \
\t  --nfw baselines/results/nfw_global_test_baseline.json \
\t  --mond baselines/results/mond_test_baseline.json \
\t  --out app/static/data/v2_fairness_pack.json

stability:
\tpython scripts/generate_stability_analysis.py \
\t  --frozen-config config/global_rc_v2_frozen.json \
\t  --out app/static/data/v2_stability.json

clean:
\trm -f app/static/data/v2_fairness_pack.json
\trm -f app/static/data/v2_stability.json
\t@echo "Cleaned generated files"
