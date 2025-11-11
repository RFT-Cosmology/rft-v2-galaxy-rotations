#!/usr/bin/env python3
"""
Generate LaTeX tables for the RFT v2 galaxy rotation paper.

T1: Parameter budgets (predictive vs descriptive configurations)
T2: LSB vs HSB split (TEST cohort)
T3: Ablation study (causal contributions)
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def format_percent(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}\\%"


def format_count_and_rate(count: int, total: int, rate: float) -> str:
    return f"{count}/{total} ({format_percent(rate)})"


def generate_t1_param_budget(final_numbers: dict) -> str:
    """Parameter budgets table."""
    n_test = final_numbers["metadata"]["n_galaxies"]

    rows = [
        ("RFT v2", "0 (predictive)", "6 global", "Acceleration-gated tail (this work)"),
        ("NFW$_{\\text{global}}$", "0 (predictive)", "2 global", "Single halo $(\\rho_s, r_s)$"),
        ("MOND", "0 (predictive)", "1 global", "Canonical $a_0$"),
        ("NFW$_{\\text{fitted}}$", "2 per galaxy", f"{2 * n_test} total", "Reference descriptive fit"),
    ]

    latex = r"""\centering
\caption{Parameter budgets for models evaluated on the blind TEST cohort. Predictive comparisons constrain per-galaxy tuning to $k=0$.}
\label{tab:param_budget}
\begin{tabular}{lccc}
\toprule
\textbf{Model} & \textbf{Per-galaxy params} & \textbf{Global params} & \textbf{Notes} \\
\midrule
"""
    for model, per_gal, global_params, notes in rows:
        latex += f"{model} & {per_gal} & {global_params} & {notes} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
"""
    return latex


def generate_t2_lsb_hsb(fairness_pack: dict) -> str:
    """LSB vs HSB split."""
    lsb = fairness_pack["lsb_hsb_split"]["lsb"]
    hsb = fairness_pack["lsb_hsb_split"]["hsb"]
    threshold = fairness_pack["lsb_hsb_split"]["threshold_kms"]

    def row(label: str, cohort: dict) -> str:
        n = cohort["n"]
        rft_pass = format_count_and_rate(
            cohort["rft_v2"]["pass_count"], n, cohort["rft_v2"]["rate"]
        )
        nfw_pass = format_count_and_rate(
            cohort["nfw_halo"]["pass_count"], n, cohort["nfw_halo"]["rate"]
        )
        mond_pass = format_count_and_rate(
            cohort["mond"]["pass_count"], n, cohort["mond"]["rate"]
        )
        return f"{label} & {n} & {rft_pass} & {nfw_pass} & {mond_pass} \\\\"

    threshold_text = f"{threshold:.0f}"
    latex = (
        "\\centering\n"
        f"\\caption{{LSB vs HSB performance on TEST (threshold $v_{{\\max}} = {threshold_text}$ km/s). "
        "RFT v2 is the only k=0 model with non-zero LSB success.}}\n"
        "\\label{tab:lsbhsb}\n"
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "\\textbf{Cohort} & \\textbf{n} & \\textbf{RFT v2} & "
        "\\textbf{NFW$_{\\text{global}}$} & \\textbf{MOND} \\\\\n"
        "\\midrule\n"
        f"{row('LSB', lsb)}\n"
        f"{row('HSB', hsb)}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )
    return latex


def generate_t3_ablations(ablations: dict) -> str:
    """Ablation study table."""
    baseline = ablations["baseline"]
    variants = ablations["variants"]

    latex = r"""\centering
\caption{Ablation study on TEST ($n=34$). Removing the acceleration-gated tail collapses predictive accuracy.}
\label{tab:ablations}
\begin{tabular}{lcc}
\toprule
\textbf{Configuration} & \textbf{Pass@20\%} & \textbf{$\Delta$ vs baseline (pp)} \\
\midrule
"""
    latex += f"{baseline['label']} & {format_percent(baseline['pass_20_rate'])} & -- \\\\\n"
    for variant in variants:
        latex += (
            f"{variant['label']} & "
            f"{format_percent(variant['pass_20_rate'])} & "
            f"{variant['delta_pp']:+.1f} \\\\\n"
        )

    latex += r"""\bottomrule
\end{tabular}
"""
    return latex


def main() -> int:
    print("=" * 70)
    print("Generating LaTeX Tables (T1–T3)")
    print("=" * 70)

    final_numbers = load_json(PROJECT_ROOT / "paper" / "build" / "final_numbers.json")
    fairness_pack = load_json(PROJECT_ROOT / "app" / "static" / "data" / "v2_fairness_pack.json")
    ablations = load_json(PROJECT_ROOT / "app" / "static" / "data" / "v2_ablations.json")

    tables_dir = PROJECT_ROOT / "paper" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "t1_param_budget.tex": generate_t1_param_budget(final_numbers),
        "t2_lsb_hsb.tex": generate_t2_lsb_hsb(fairness_pack),
        "t3_ablations.tex": generate_t3_ablations(ablations),
    }

    for filename, latex in outputs.items():
        path = tables_dir / filename
        path.write_text(latex)
        print(f"✅ Wrote {path.relative_to(PROJECT_ROOT)}")

    print("=" * 70)
    print("✅ Tables ready")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
