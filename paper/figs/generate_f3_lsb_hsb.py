#!/usr/bin/env python3
"""
Figure 3: LSB vs HSB Diagnostic

Shows RFT's acceleration-gated tail design advantage on low surface brightness
(LSB) galaxies where g_b << g*.

LSB defined as v_max < 120 km/s (threshold from fairness pack).
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Colorblind-safe palette
COLORS = {
    'rft': '#0072B2',     # blue
    'nfw': '#E69F00',     # orange
    'mond': '#009E73',    # green
}

def load_lsb_hsb_data():
    """Load LSB/HSB split from fairness pack."""
    path = PROJECT_ROOT / "app/static/data/v2_fairness_pack.json"
    with open(path) as f:
        data = json.load(f)
    return data['lsb_hsb_split']

def main():
    lsb_hsb = load_lsb_hsb_data()

    threshold = lsb_hsb['threshold_kms']
    lsb_data = lsb_hsb['lsb']
    hsb_data = lsb_hsb['hsb']

    # Extract pass rates
    lsb_rates = {
        'RFT v2': lsb_data['rft_v2']['rate'],
        'NFW global': lsb_data['nfw_halo']['rate'],
        'MOND': lsb_data['mond']['rate'],
    }

    hsb_rates = {
        'RFT v2': hsb_data['rft_v2']['rate'],
        'NFW global': hsb_data['nfw_halo']['rate'],
        'MOND': hsb_data['mond']['rate'],
    }

    lsb_counts = {
        'RFT v2': f"{lsb_data['rft_v2']['pass_count']}/{lsb_data['n']}",
        'NFW global': f"{lsb_data['nfw_halo']['pass_count']}/{lsb_data['n']}",
        'MOND': f"{lsb_data['mond']['pass_count']}/{lsb_data['n']}",
    }

    hsb_counts = {
        'RFT v2': f"{hsb_data['rft_v2']['pass_count']}/{hsb_data['n']}",
        'NFW global': f"{hsb_data['nfw_halo']['pass_count']}/{hsb_data['n']}",
        'MOND': f"{hsb_data['mond']['pass_count']}/{hsb_data['n']}",
    }

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    models = list(lsb_rates.keys())
    x = np.arange(len(models))
    width = 0.35

    # Group bars
    lsb_bars = ax.bar(x - width/2, list(lsb_rates.values()), width,
                      label=f'LSB (v$_{{max}}$ < {threshold:.0f} km/s)',
                      color=[COLORS['rft'], COLORS['nfw'], COLORS['mond']],
                      alpha=0.85, edgecolor='black', linewidth=1.5)

    hsb_bars = ax.bar(x + width/2, list(hsb_rates.values()), width,
                      label=f'HSB (v$_{{max}}$ ≥ {threshold:.0f} km/s)',
                      color=[COLORS['rft'], COLORS['nfw'], COLORS['mond']],
                      alpha=0.5, edgecolor='black', linewidth=1.5, hatch='//')

    # Add count labels
    for i, (lsb_bar, hsb_bar) in enumerate(zip(lsb_bars, hsb_bars)):
        model = models[i]

        # LSB label
        height_lsb = lsb_bar.get_height()
        ax.text(lsb_bar.get_x() + lsb_bar.get_width()/2., height_lsb + 2,
                lsb_counts[model],
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        # HSB label
        height_hsb = hsb_bar.get_height()
        ax.text(hsb_bar.get_x() + hsb_bar.get_width()/2., height_hsb + 2,
                hsb_counts[model],
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Formatting
    ax.set_ylabel('Pass@20% (%)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Model (k=0)', fontsize=14, fontweight='bold')
    ax.set_title('LSB vs HSB Performance (TEST n=34)', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=13)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight LSB dominance
    ax.axhline(50, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(0.02, 0.95,
            f'LSB n={lsb_data["n"]}, HSB n={hsb_data["n"]}\n'
            f'RFT LSB advantage: {lsb_rates["RFT v2"]:.1f}% vs {lsb_rates["NFW global"]:.1f}% (NFW), {lsb_rates["MOND"]:.1f}% (MOND)',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "paper/figs/fig3_lsb_hsb.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure 3 saved to {output_path}")
    print(f"   LSB (n={lsb_data['n']}): RFT {lsb_rates['RFT v2']:.1f}%, NFW {lsb_rates['NFW global']:.1f}%, MOND {lsb_rates['MOND']:.1f}%")
    print(f"   HSB (n={hsb_data['n']}): RFT {hsb_rates['RFT v2']:.1f}%, NFW {hsb_rates['NFW global']:.1f}%, MOND {hsb_rates['MOND']:.1f}%")
    print(f"   LSB dominance validates acceleration-gating mechanism")

    return 0

if __name__ == "__main__":
    sys.exit(main())
