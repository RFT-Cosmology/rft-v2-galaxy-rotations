#!/usr/bin/env python3
"""
RFT projected weak lensing solver.

Computes excess surface density ΔΣ(R) from RFT nonlocal kernel in projected geometry.

Theory:
-------
1. RFT gives g_RFT(r) in 3D via nonlocal kernel K(Δln r):
   g_RFT(r) = ∫ K(ln r - ln r') g_bar(r') d ln r'

2. Surface density via projection:
   Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r² - R²)

3. Excess surface density:
   ΔΣ(R) = Σ̄(<R) - Σ(R)
   where Σ̄(<R) = (2/R²) ∫_0^R Σ(R') R' dR'

4. For zero per-cluster parameters:
   - Global kernel K(Δρ) learned from TRAIN stack
   - No M200, c, or scale radius per cluster
   - Prediction is ΔΣ_RFT(R | K_global, baryonic profile)

Author: RFT Cosmology Project
Date: 2025-11-10
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass

# Import RFT kernel functions
sys.path.insert(0, str(Path(__file__).parent.parent))
from sparc_rft.kernels import log_kernel_mixed


@dataclass
class LensingPrediction:
    """Container for RFT lensing predictions."""
    R_kpc: np.ndarray
    DeltaSigma_pred: np.ndarray  # Msun/pc^2
    Sigma_pred: np.ndarray  # Msun/pc^2 (for diagnostics)
    g_3d: np.ndarray  # 3D acceleration profile (for diagnostics)
    r_3d: np.ndarray  # 3D radii


def rft_acceleration_3d(
    r_kpc: np.ndarray,
    rho_bar_r: np.ndarray,
    kernel_params: Dict
) -> np.ndarray:
    """
    Compute RFT acceleration g_RFT(r) in 3D via nonlocal convolution.

    Pipeline:
    1. Compute Newtonian g_bar(r) from ρ_bar(r)
    2. Apply kernel convolution: g_RFT(r) = ∫ K(ln r - ln r') g_bar(r') d ln r'

    Parameters
    ----------
    r_kpc : np.ndarray
        3D radii [kpc]
    rho_bar_r : np.ndarray
        Baryonic density profile ρ_bar(r) [Msun/kpc^3]
    kernel_params : Dict
        Kernel configuration:
        - type: "mix" (only mixed kernel supported for now)
        - w: Lorentzian weight [0,1]
        - sigma_g: Gaussian width in log r
        - sigma_l: Lorentzian width in log r
        - asym: Asymmetry parameter (default 0)

    Returns
    -------
    g_RFT : np.ndarray
        RFT acceleration [km/s]^2 / kpc
    """
    G_kpc_Msun_kms = 4.302e-9  # G in kpc (km/s)^2 / Msun

    # Step 1: Compute Newtonian acceleration g_bar(r)
    # g_bar(r) = G M(<r) / r^2
    g_bar = np.zeros_like(r_kpc)
    for i, r in enumerate(r_kpc):
        r_inner = r_kpc[r_kpc <= r]
        rho_inner = rho_bar_r[r_kpc <= r]

        if len(r_inner) > 1:
            M_enc = 4 * np.pi * np.trapz(rho_inner * r_inner**2, r_inner)
            g_bar[i] = G_kpc_Msun_kms * M_enc / max(r**2, 1e-6)
        else:
            g_bar[i] = 0.0

    # Step 2: Apply kernel convolution in log-radius space
    # g_RFT(ρ_i) = ∫ K(ρ_i - ρ_j) g_bar(ρ_j) d ρ_j
    # where ρ = ln(r)

    kernel_type = kernel_params.get("type", "mix")

    if kernel_type != "mix":
        raise NotImplementedError(f"Kernel type '{kernel_type}' not yet supported for lensing")

    # Extract kernel parameters
    w = kernel_params.get("w", 0.5)
    sigma_g = kernel_params.get("sigma_g", 0.45)
    sigma_l = kernel_params.get("sigma_l", 0.60)
    asym = kernel_params.get("asym", 0.0)

    # Compute log-radius array
    rho = np.log(np.clip(r_kpc, 1e-12, None))

    # Convolve g_bar with kernel
    g_rft = np.zeros_like(r_kpc)
    for i in range(len(r_kpc)):
        # Compute Δρ = ρ_i - ρ_j for all j
        drho = rho[i] - rho

        # Compute kernel K(Δρ)
        K = log_kernel_mixed(drho, w, sigma_g, sigma_l, asym)

        # Trapezoidal integration: ∫ K(Δρ) g_bar(ρ) dρ
        if len(rho) > 1:
            d_rho = np.diff(rho)
            weights = np.zeros_like(rho)
            weights[0] = d_rho[0] * 0.5 if len(d_rho) > 0 else 1.0
            weights[-1] = d_rho[-1] * 0.5 if len(d_rho) > 0 else 1.0
            if len(d_rho) > 1:
                weights[1:-1] = (d_rho[:-1] + d_rho[1:]) * 0.5

            g_rft[i] = np.sum(K * g_bar * weights)
        else:
            g_rft[i] = g_bar[i]

    return g_rft


def project_density_to_surface_density(
    r_3d: np.ndarray,
    rho_3d: np.ndarray,
    R_proj: np.ndarray
) -> np.ndarray:
    """
    Project 3D density ρ(r) to surface density Σ(R).

    Σ(R) = 2 ∫_R^∞ ρ(r) r dr / √(r² - R²)

    Uses adaptive grid refinement near r=R for stability.

    Parameters
    ----------
    r_3d : np.ndarray
        3D radii [kpc]
    rho_3d : np.ndarray
        3D density [Msun/kpc^3]
    R_proj : np.ndarray
        Projected radii [kpc]

    Returns
    -------
    Sigma : np.ndarray
        Surface density [Msun/kpc^2]
    """
    Sigma = np.zeros_like(R_proj)

    for i, R in enumerate(R_proj):
        # Integrate from R to r_max
        # Use points where r > R (avoid singularity)
        mask = r_3d > R * 1.001  # Tiny buffer to avoid numerical issues
        r_far = r_3d[mask]
        rho_far = rho_3d[mask]

        if len(r_far) < 3:
            Sigma[i] = 0.0
            continue

        # Create refined grid near r=R for better accuracy
        # Use log-spacing to capture near-R behavior
        r_near_max = min(R * 1.5, r_far[0])
        r_near = R * np.logspace(np.log10(1.001), np.log10(r_near_max / R), 20)
        rho_near = np.interp(r_near, r_3d, rho_3d)

        # Combine near + far grids
        r_int = np.concatenate([r_near, r_far[r_far > r_near_max]])
        rho_int = np.concatenate([rho_near, rho_far[r_far > r_near_max]])

        # Integrand: ρ(r) r / √(r² - R²)
        integrand = rho_int * r_int / np.sqrt(r_int**2 - R**2)

        # Use log-spacing integration for better tail behavior
        Sigma[i] = 2 * np.trapz(integrand, r_int)

    return Sigma


def compute_mean_surface_density(R_kpc: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Compute mean surface density within R: Σ̄(<R).

    Σ̄(<R) = (2/R²) ∫_0^R Σ(R') R' dR'

    Parameters
    ----------
    R_kpc : np.ndarray
        Projected radii [kpc]
    Sigma : np.ndarray
        Surface density at R [Msun/kpc^2]

    Returns
    -------
    Sigma_mean : np.ndarray
        Mean surface density within R [Msun/kpc^2]
    """
    Sigma_mean = np.zeros_like(R_kpc)

    for i, R in enumerate(R_kpc):
        R_inner = R_kpc[R_kpc <= R]
        Sigma_inner = Sigma[R_kpc <= R]

        if len(R_inner) > 1:
            integrand = Sigma_inner * R_inner
            Sigma_mean[i] = (2.0 / max(R**2, 1e-6)) * np.trapz(integrand, R_inner)
        else:
            Sigma_mean[i] = Sigma_inner[0] if len(Sigma_inner) > 0 else 0.0

    return Sigma_mean


def rft_delta_sigma(
    R_proj_kpc: np.ndarray,
    rho_bar_3d: np.ndarray,
    r_3d_kpc: np.ndarray,
    kernel_params: Dict,
    return_diagnostics: bool = False
) -> np.ndarray:
    """
    Compute RFT excess surface density ΔΣ(R).

    Pipeline:
    1. Compute g_RFT(r) via nonlocal kernel convolution
    2. Invert to ρ_RFT(r) via Poisson equation
    3. Project to Σ(R)
    4. Compute ΔΣ(R) = Σ̄(<R) - Σ(R)

    Parameters
    ----------
    R_proj_kpc : np.ndarray
        Projected radii for output [kpc]
    rho_bar_3d : np.ndarray
        Baryonic density profile [Msun/kpc^3]
    r_3d_kpc : np.ndarray
        3D radii for baryonic profile [kpc]
    kernel_params : Dict
        RFT kernel configuration
    return_diagnostics : bool
        If True, return LensingPrediction with diagnostics

    Returns
    -------
    DeltaSigma : np.ndarray or LensingPrediction
        Excess surface density [Msun/pc^2] (or full prediction object)
    """
    # Step 1: Compute g_RFT(r) in 3D
    g_rft = rft_acceleration_3d(r_3d_kpc, rho_bar_3d, kernel_params)

    # Step 2: Invert to ρ_RFT(r) via Poisson equation
    # ∇²Φ = 4πG ρ  →  (1/r²) d/dr(r² dΦ/dr) = 4πG ρ
    # g(r) = dΦ/dr  →  ρ(r) = (1/4πG r²) d/dr(r² g(r))
    G_kpc_Msun_kms = 4.302e-9

    rho_rft = np.zeros_like(r_3d_kpc)
    for i in range(len(r_3d_kpc)):
        if i == 0:
            # Forward difference at origin
            if len(r_3d_kpc) > 1:
                dr = r_3d_kpc[1] - r_3d_kpc[0]
                d_r2g = (r_3d_kpc[1]**2 * g_rft[1] - r_3d_kpc[0]**2 * g_rft[0]) / dr
                rho_rft[i] = d_r2g / (4 * np.pi * G_kpc_Msun_kms * max(r_3d_kpc[i]**2, 1e-6))
        elif i == len(r_3d_kpc) - 1:
            # Backward difference at edge
            dr = r_3d_kpc[i] - r_3d_kpc[i-1]
            d_r2g = (r_3d_kpc[i]**2 * g_rft[i] - r_3d_kpc[i-1]**2 * g_rft[i-1]) / dr
            rho_rft[i] = d_r2g / (4 * np.pi * G_kpc_Msun_kms * max(r_3d_kpc[i]**2, 1e-6))
        else:
            # Central difference
            dr_fwd = r_3d_kpc[i+1] - r_3d_kpc[i]
            dr_bwd = r_3d_kpc[i] - r_3d_kpc[i-1]
            d_r2g = ((r_3d_kpc[i+1]**2 * g_rft[i+1] - r_3d_kpc[i]**2 * g_rft[i]) / dr_fwd +
                     (r_3d_kpc[i]**2 * g_rft[i] - r_3d_kpc[i-1]**2 * g_rft[i-1]) / dr_bwd) * 0.5
            rho_rft[i] = d_r2g / (4 * np.pi * G_kpc_Msun_kms * max(r_3d_kpc[i]**2, 1e-6))

    # Step 3: Project to Σ(R)
    Sigma = project_density_to_surface_density(r_3d_kpc, rho_rft, R_proj_kpc)

    # Step 4: Compute mean Σ̄(<R)
    Sigma_mean = compute_mean_surface_density(R_proj_kpc, Sigma)

    # Step 5: ΔΣ = Σ̄ - Σ
    DeltaSigma_kpc2 = Sigma_mean - Sigma

    # Convert Msun/kpc^2 → Msun/pc^2
    DeltaSigma_pc2 = DeltaSigma_kpc2 / 1e6  # 1 kpc^2 = 10^6 pc^2

    if return_diagnostics:
        return LensingPrediction(
            R_kpc=R_proj_kpc,
            DeltaSigma_pred=DeltaSigma_pc2,
            Sigma_pred=Sigma / 1e6,  # convert to Msun/pc^2
            g_3d=g_rft,
            r_3d=r_3d_kpc
        )
    else:
        return DeltaSigma_pc2


def solve_cluster(
    cluster_profile,
    global_config: Dict
) -> LensingPrediction:
    """
    Solve single cluster with RFT projected solver.

    Parameters
    ----------
    cluster_profile : ClusterProfile
        Cluster data from lensing_loader
    global_config : Dict
        Global RFT configuration (kernel params, no per-cluster tuning)

    Returns
    -------
    prediction : LensingPrediction
        RFT prediction for ΔΣ(R)
    """
    # TODO: Extract baryonic profile from cluster data
    # For now, use NFW placeholder for baryonic component
    # In real implementation, this would come from gas + stars

    R_obs = cluster_profile.R_kpc

    # Build 3D grid extending beyond max observed R
    r_max_3d = max(R_obs) * 3.0
    r_3d = np.logspace(np.log10(min(R_obs) * 0.1), np.log10(r_max_3d), 100)

    # Placeholder baryonic density (NFW)
    # ρ_NFW(r) = ρ_s / ((r/r_s) * (1 + r/r_s)^2)
    r_s = 200.0  # kpc (placeholder)
    rho_s = 1e7  # Msun/kpc^3 (placeholder)
    rho_bar = rho_s / ((r_3d / r_s) * (1 + r_3d / r_s)**2)

    # Get kernel params from config
    kernel_params = global_config.get("kernel", {
        "kernel": "mix",
        "w": 0.5,
        "sigma_g": 0.45,
        "sigma_l": 0.60
    })

    # Compute ΔΣ(R)
    prediction = rft_delta_sigma(
        R_obs,
        rho_bar,
        r_3d,
        kernel_params,
        return_diagnostics=True
    )

    return prediction


if __name__ == "__main__":
    # Simple test with mock cluster
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from ingest.lensing_loader import ClusterProfile
    from pathlib import Path

    # Mock cluster
    mock = ClusterProfile(
        name="MockCluster",
        redshift=0.3,
        source_redshift=0.8,
        H0=70.0,
        Om0=0.3,
        Ode0=0.7,
        R_kpc=np.array([100, 200, 400, 800, 1600]),
        DeltaSigma=np.array([15.0, 10.0, 6.0, 3.0, 1.5]),  # Msun/pc^2
        DeltaSigma_err=np.array([1.5, 1.0, 0.6, 0.3, 0.2]),
        Sigma_crit=3000.0,
        metadata={"survey": "Mock"}
    )

    config = {
        "kernel": {
            "kernel": "mix",
            "w": 0.5,
            "sigma_g": 0.45,
            "sigma_l": 0.60
        }
    }

    pred = solve_cluster(mock, config)
    print(f"RFT Prediction for {mock.name}:")
    for i, R in enumerate(pred.R_kpc):
        print(f"  R = {R:5.0f} kpc: ΔΣ_RFT = {pred.DeltaSigma_pred[i]:6.2f} Msun/pc^2 "
              f"(obs: {mock.DeltaSigma[i]:6.2f} ± {mock.DeltaSigma_err[i]:4.2f})")
