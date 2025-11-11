#!/usr/bin/env python3
"""
Figure 8: Mechanism Visualization

Shows how acceleration gate and radial onset shape the tail boost.

Panel A: Acceleration gate vs g_b/g*
Panel B: Radial onset vs r/r_turn
"""

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def load_frozen_config():
    """Load frozen RFT v2 parameters."""
    path = PROJECT_ROOT / "config/global_rc_v2_frozen.json"
    with open(path) as f:
        return json.load(f)['tail']

def acceleration_gate(g_b_over_g_star, gamma):
    """Acceleration gate: [1 + (g_b/g*)^γ]^(-1)"""
    return 1.0 / (1.0 + g_b_over_g_star ** gamma)

def radial_onset(r_over_r_turn, p):
    """Radial onset: 1 - exp(-(r/r_turn)^p)"""
    return 1.0 - np.exp(-(r_over_r_turn ** p))

def main():
    config = load_frozen_config()

    gamma = config['gamma']
    g_star = config['g_star_kms2_per_kpc']
    r_turn = config['r_turn_kpc']
    p = config['p']

    # Create figure with 2 panels
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ===== Panel A: Acceleration Gate =====
    g_b_over_g_star = np.logspace(-2, 1, 200)  # 0.01 to 10
    gate_values = acceleration_gate(g_b_over_g_star, gamma)

    ax1.semilogx(g_b_over_g_star, gate_values, 'b-', linewidth=3, label=f'γ = {gamma}')
    ax1.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label='g_b = g* (threshold)')
    ax1.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Mark typical LSB and HSB regions
    ax1.axvspan(0.01, 0.3, alpha=0.2, color='blue', label='Typical LSB (g_b << g*)')
    ax1.axvspan(2, 10, alpha=0.2, color='orange', label='Typical HSB (g_b >> g*)')

    ax1.set_xlabel('g$_b$ / g* (Baryonic Acceleration Ratio)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Gate Value [0, 1]', fontsize=13, fontweight='bold')
    ax1.set_title(f'Panel A: Acceleration Gate (g* = {g_star:.0f} km² s⁻² kpc⁻¹)',
                  fontsize=14, fontweight='bold')
    ax1.set_xlim(0.01, 10)
    ax1.set_ylim(0, 1.1)
    ax1.grid(alpha=0.3)
    ax1.legend(loc='upper right', fontsize=10)

    # Annotation
    ax1.text(0.05, 0.95,
             'LSB: g_b << g* → gate ≈ 1 (full tail boost)\n'
             'HSB: g_b >> g* → gate ≈ 0 (suppressed)',
             transform=ax1.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    # ===== Panel B: Radial Onset =====
    r_over_r_turn = np.linspace(0, 5, 200)
    onset_values = radial_onset(r_over_r_turn, p)

    ax2.plot(r_over_r_turn, onset_values, 'g-', linewidth=3, label=f'p = {p}')
    ax2.axvline(1.0, color='red', linestyle='--', linewidth=2, alpha=0.7,
                label=f'r = r_turn = {r_turn} kpc')
    ax2.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)

    # Mark regions
    ax2.axvspan(0, 0.5, alpha=0.2, color='red', label='Inner disk (suppressed)')
    ax2.axvspan(2, 5, alpha=0.2, color='green', label='Outer disk (active)')

    ax2.set_xlabel('r / r$_{turn}$ (Radial Distance Ratio)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Onset Value [0, 1]', fontsize=13, fontweight='bold')
    ax2.set_title(f'Panel B: Radial Onset (r_turn = {r_turn} kpc)',
                  fontsize=14, fontweight='bold')
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1.1)
    ax2.grid(alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)

    # Annotation
    ax2.text(0.05, 0.95,
             f'Inner: r << r_turn → onset ≈ 0 (no tail)\n'
             f'Outer: r >> r_turn → onset ≈ 1 (full tail)',
             transform=ax2.transAxes, fontsize=10, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

    plt.tight_layout()

    # Save
    output_path = PROJECT_ROOT / "paper/figs/fig8_mechanism.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.png'), dpi=300, bbox_inches='tight')

    print(f"✅ Figure 8 saved to {output_path}")
    print(f"   Acceleration gate: γ = {gamma}, g* = {g_star} km² s⁻² kpc⁻¹")
    print(f"   Radial onset: p = {p}, r_turn = {r_turn} kpc")
    print(f"   LSB advantage: gate ≈ 1 when g_b << g*, onset ≈ 1 at outer radii")

    return 0

if __name__ == "__main__":
    sys.exit(main())
