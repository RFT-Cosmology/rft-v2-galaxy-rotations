#!/usr/bin/env python3
"""
Figure 2: Paired Head-to-Head (McNemar Test)

Panel A: 2x2 contingency matrix (RFT vs NFW_global)
Panel B: Discordant pairs bar showing RFT wins vs NFW wins

Emphasizes McNemar p=0.69 (NOT significant) as PRIMARY test.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Colorblind-safe palette
COLORS = {
    'rft_win': '#0072B2',     # blue
    'nfw_win': '#E69F00',     # orange
    'both_pass': '#56B4E9',   # light blue
    'both_fail': '#999999',   # gray
}

def load_final_numbers():
    """Load frozen final numbers."""
    path = PROJECT_ROOT / "paper/build/final_numbers.json"
    with open(path) as f:
        return json.load(f)

def main():
    data = load_final_numbers()

    # Extract paired test results
    rft_vs_nfw = data['paired_tests']['rft_vs_nfw']
    contingency = rft_vs_nfw['contingency']
    mcnemar_p = rft_vs_nfw['mcnemar_p']

    both_pass = contingency['both_pass']
    rft_only = contingency['rft_only']
    nfw_only = contingency['competitor_only']
    both_fail = contingency['both_fail']

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Panel A: Contingency Matrix =====
    contingency_matrix = np.array([[both_pass, rft_only],
                                     [nfw_only, both_fail]])

    im = ax1.imshow(contingency_matrix, cmap='Blues', alpha=0.6, vmin=0, vmax=20)

    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax1.text(j, i, contingency_matrix[i, j],
                           ha="center", va="center", color="black",
                           fontsize=24, fontweight='bold')

    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['NFW Pass', 'NFW Fail'], fontsize=13, fontweight='bold')
    ax1.set_yticklabels(['RFT Pass', 'RFT Fail'], fontsize=13, fontweight='bold')
    ax1.set_xlabel('NFW Global (k=0)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('RFT v2 (k=0)', fontsize=14, fontweight='bold')
    ax1.set_title('Panel A: Contingency Table (n=34)', fontsize=15, fontweight='bold')

    # Add grid
    ax1.set_xticks([0.5], minor=True)
    ax1.set_yticks([0.5], minor=True)
    ax1.grid(which='minor', color='black', linestyle='-', linewidth=2)

    # Add legend for cells
    legend_labels = [
        f'Both Pass: {both_pass}',
        f'RFT Only: {rft_only}',
        f'NFW Only: {nfw_only}',
        f'Both Fail: {both_fail}'
    ]
    ax1.text(0.02, 0.98, '\n'.join(legend_labels),
             transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # ===== Panel B: Discordant Pairs =====
    discordant_labels = ['RFT Wins\n(RFT pass, NFW fail)', 'NFW Wins\n(NFW pass, RFT fail)']
    discordant_counts = [rft_only, nfw_only]
    colors_bar = [COLORS['rft_win'], COLORS['nfw_win']]

    x_pos = np.arange(len(discordant_labels))
    bars = ax2.bar(x_pos, discordant_counts, color=colors_bar, alpha=0.85,
                   edgecolor='black', linewidth=1.5)

    # Add count labels
    for bar, count in zip(bars, discordant_counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{count}',
                ha='center', va='bottom', fontsize=16, fontweight='bold')

    ax2.set_ylabel('Count (Discordant Pairs)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Outcome', fontsize=14, fontweight='bold')
    ax2.set_title(f'Panel B: McNemar Exact Test\np = {mcnemar_p:.4f} (NOT significant)',
                  fontsize=15, fontweight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(discordant_labels, fontsize=12)
    ax2.set_ylim(0, max(discordant_counts) + 2)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')

    # Annotation for interpretation
    total_discordant = rft_only + nfw_only
    ax2.text(0.5, 0.95, f'Discordant pairs: {total_discordant}\nUnder H₀: 50/50 split expected',
             transform=ax2.transAxes, fontsize=11, ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # Metadata
    commit = data['metadata']['commit']
    ax2.text(0.02, 0.02, f'Frozen: {commit}',
             transform=ax2.transAxes, fontsize=9, style='italic',
             verticalalignment='bottom', color='gray')

    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "paper/figs/fig2_mcnemar.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure 2 saved to {output_path}")
    print(f"   Contingency: Both pass={both_pass}, RFT only={rft_only}, NFW only={nfw_only}, Both fail={both_fail}")
    print(f"   McNemar p = {mcnemar_p:.4f} (paired exact test)")
    print(f"   Interpretation: NOT significant (p > 0.05)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
