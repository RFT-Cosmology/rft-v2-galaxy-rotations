#!/usr/bin/env python3
"""
Figure 6: Ablations (Causal Analysis)

Shows effect of removing each component to establish causality.
Tail removal causes largest drop (−35.3 pp), confirming it's the causal driver.
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_ablations_data():
    """Load ablations analysis."""
    path = PROJECT_ROOT / "app/static/data/v2_ablations.json"
    with open(path) as f:
        return json.load(f)

def main():
    data = load_ablations_data()

    baseline = data['baseline']
    variants = data['variants']

    baseline_rate = baseline['pass_20_rate']

    # Extract variant data
    labels = [baseline['label']] + [v['label'] for v in variants]
    rates = [baseline_rate] + [v['pass_20_rate'] for v in variants]
    deltas = [0.0] + [v['delta_pp'] for v in variants]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    y_pos = np.arange(len(labels))

    # Color by delta (red = worse, green = baseline)
    colors = ['green'] + ['red' if d < -5 else 'orange' for d in deltas[1:]]

    bars = ax.barh(y_pos, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)

    # Add delta labels
    for i, (bar, rate, delta) in enumerate(zip(bars, rates, deltas)):
        # Rate label
        ax.text(rate + 1.5, bar.get_y() + bar.get_height()/2.,
                f'{rate:.1f}%',
                va='center', fontsize=11, fontweight='bold')

        # Delta label (skip baseline)
        if i > 0:
            ax.text(2, bar.get_y() + bar.get_height()/2.,
                    f'Δ {delta:+.1f} pp',
                    va='center', fontsize=10, style='italic', color='darkred')

    # Formatting
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Pass@20% (%)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=14, fontweight='bold')
    ax.set_title('Ablation Study: Component Causality (TEST n=34)',
                 fontsize=16, fontweight='bold')
    ax.set_xlim(0, 75)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Baseline reference line
    ax.axvline(baseline_rate, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Baseline: {baseline_rate:.1f}%')

    ax.legend(loc='lower right', fontsize=11)

    # Interpretation box
    tail_delta = deltas[1]  # First variant is "no tail"
    ax.text(0.98, 0.98,
            f'Causal hierarchy:\n'
            f'• Tail removal: {tail_delta:.1f} pp (largest drop)\n'
            f'• Gates shape where/when boost applies\n'
            f'• Confirms tail is primary causal driver',
            transform=ax.transAxes, fontsize=10, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "paper/figs/fig6_ablations.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure 6 saved to {output_path}")
    print(f"   Baseline: {baseline_rate:.1f}%")
    for v in variants:
        print(f"   {v['label']}: {v['pass_20_rate']:.1f}% (Δ {v['delta_pp']:+.1f} pp)")

    return 0

if __name__ == "__main__":
    sys.exit(main())
