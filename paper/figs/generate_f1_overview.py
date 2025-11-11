#!/usr/bin/env python3
"""
Figure 1: Overview Accuracy (Pass@20% with Wilson CIs)

Camera-ready bar chart showing predictive k=0 comparison on TEST cohort.
Source: paper/build/final_numbers.json (frozen, hash-locked)
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Colorblind-safe palette (Wong 2011)
COLORS = {
    'rft': '#0072B2',     # blue
    'nfw': '#E69F00',     # orange
    'mond': '#009E73',    # green
}

def load_final_numbers():
    """Load frozen final numbers."""
    path = PROJECT_ROOT / "paper/build/final_numbers.json"
    with open(path) as f:
        return json.load(f)

def main():
    data = load_final_numbers()

    # Extract aggregate stats
    rft = data['aggregate']['rft_v2']
    nfw = data['aggregate']['nfw_global']
    mond = data['aggregate']['mond']

    n = data['metadata']['n_galaxies']

    # Pass rates and counts
    models = ['RFT v2\n(k=0)', 'NFW global\n(k=0)', 'MOND\n(k=0)']
    pass_rates = [rft['pass_rate'], nfw['pass_rate'], mond['pass_rate']]
    pass_counts = [rft['pass_count'], nfw['pass_count'], mond['pass_count']]

    # Wilson CIs (convert to percentages)
    ci_rft = [100 * x for x in rft['wilson_ci']]
    ci_nfw = [100 * x for x in nfw['wilson_ci']]
    ci_mond = [100 * x for x in mond['wilson_ci']]

    # Error bars (asymmetric)
    errors_lower = [
        pass_rates[0] - ci_rft[0],
        pass_rates[1] - ci_nfw[0],
        pass_rates[2] - ci_mond[0],
    ]
    errors_upper = [
        ci_rft[1] - pass_rates[0],
        ci_nfw[1] - pass_rates[1],
        ci_mond[1] - pass_rates[2],
    ]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    x_pos = np.arange(len(models))
    colors = [COLORS['rft'], COLORS['nfw'], COLORS['mond']]

    bars = ax.bar(x_pos, pass_rates, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)

    # Error bars (Wilson CIs)
    ax.errorbar(x_pos, pass_rates,
                yerr=[errors_lower, errors_upper],
                fmt='none', ecolor='black', capsize=8, capthick=2, linewidth=2)

    # Add count labels on bars
    for i, (bar, count) in enumerate(zip(bars, pass_counts)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 3,
                f'{count}/{n}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting
    ax.set_ylabel('Pass@20% (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model (Predictive k=0 Comparison)', fontsize=14, fontweight='bold')
    ax.set_title('TEST Cohort Performance (n=34, blind)', fontsize=16, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, fontsize=12)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Legend for error bars
    ci_patch = mpatches.Patch(color='none', label='Error bars: Wilson 95% CI')
    ax.legend(handles=[ci_patch], loc='upper right', fontsize=11)

    # Metadata annotation
    commit = data['metadata']['commit']
    tag = data['metadata']['tag']
    ax.text(0.02, 0.02, f'Frozen: {tag} ({commit})',
            transform=ax.transAxes, fontsize=9, style='italic',
            verticalalignment='bottom', color='gray')

    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "paper/figs/fig1_overview.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure 1 saved to {output_path}")
    print(f"   RFT v2: {pass_rates[0]:.1f}% [{ci_rft[0]:.1f}%, {ci_rft[1]:.1f}%]")
    print(f"   NFW:    {pass_rates[1]:.1f}% [{ci_nfw[0]:.1f}%, {ci_nfw[1]:.1f}%]")
    print(f"   MOND:   {pass_rates[2]:.1f}% [{ci_mond[0]:.1f}%, {ci_mond[1]:.1f}%]")

    return 0

if __name__ == "__main__":
    sys.exit(main())
