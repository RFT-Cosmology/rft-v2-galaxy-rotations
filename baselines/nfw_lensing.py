#!/usr/bin/env python3
"""
NFW weak lensing baseline.

Fits projected NFW profile ΔΣ_NFW(R; M200, c) to cluster lensing data.
Provides BIC-fair comparison to RFT (2 params/cluster vs 0 params/cluster).

Author: RFT Cosmology Project
Date: 2025-11-10
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class NFWFit:
    """Container for NFW fit results."""
    M200: float  # Virial mass [10^14 Msun]
    c: float  # Concentration
    r200: float  # Virial radius [kpc]
    r_s: float  # Scale radius [kpc]
    rho_s: float  # Characteristic density [Msun/kpc^3]
    chi2: float
    dof: int
    n_params: int  # Always 2 for NFW
    BIC: float


def concentration_duffy08(M200_1e14, z):
    """
    Concentration-mass relation from Duffy et al. (2008).

    c200 = A * (M200 / M_pivot)^B * (1+z)^C

    For relaxed clusters (full sample):
    A = 5.71, B = -0.084, C = -0.47
    M_pivot = 2e12 Msun/h

    Parameters
    ----------
    M200_1e14 : float
        Virial mass [10^14 Msun]
    z : float
        Redshift

    Returns
    -------
    c200 : float
        Concentration
    """
    h = 0.7  # Assumed Hubble parameter
    A, B, C = 5.71, -0.084, -0.47
    M_pivot_1e14 = 0.02 / h  # 2e12 Msun/h in units of 10^14 Msun

    c200 = A * (M200_1e14 / M_pivot_1e14)**B * (1 + z)**C
    return c200


def r200_from_M200(M200_1e14, z, Om0=0.3, Ode0=0.7):
    """
    Compute virial radius r200 from M200.

    M200 = (4π/3) * 200 * ρ_crit(z) * r200^3

    Parameters
    ----------
    M200_1e14 : float
        Virial mass [10^14 Msun]
    z : float
        Redshift
    Om0 : float
        Matter density
    Ode0 : float
        Dark energy density

    Returns
    -------
    r200 : float
        Virial radius [kpc]
    """
    # Critical density at z
    # ρ_crit(z) = ρ_crit(0) * E(z)^2
    # ρ_crit(0) = 3 H0^2 / 8π G ≈ 2.775e11 h^2 Msun/Mpc^3

    h = 0.7
    rho_crit_0 = 2.775e11 * h**2  # Msun/Mpc^3
    E_z_sq = Om0 * (1 + z)**3 + Ode0
    rho_crit_z = rho_crit_0 * E_z_sq

    # M200 = (4π/3) * 200 * ρ_crit(z) * r200^3
    # r200 = [3 M200 / (800π ρ_crit)]^(1/3)

    M200_Msun = M200_1e14 * 1e14
    r200_Mpc = (3 * M200_Msun / (800 * np.pi * rho_crit_z))**(1/3)
    r200_kpc = r200_Mpc * 1e3

    return r200_kpc


def nfw_params_from_M200_c(M200_1e14, c, z, Om0=0.3, Ode0=0.7):
    """
    Compute NFW parameters (r_s, ρ_s) from (M200, c).

    Parameters
    ----------
    M200_1e14 : float
        Virial mass [10^14 Msun]
    c : float
        Concentration
    z : float
        Redshift
    Om0, Ode0 : float
        Cosmology

    Returns
    -------
    r_s : float
        Scale radius [kpc]
    rho_s : float
        Characteristic density [Msun/kpc^3]
    """
    r200 = r200_from_M200(M200_1e14, z, Om0, Ode0)
    r_s = r200 / c

    # Characteristic density
    # M200 = M_NFW(<r200) = 4π ρ_s r_s^3 [ln(1+c) - c/(1+c)]
    # Solve for ρ_s

    def f_c(c):
        return np.log(1 + c) - c / (1 + c)

    M200_Msun = M200_1e14 * 1e14
    rho_s_Msun_kpc3 = M200_Msun / (4 * np.pi * r_s**3 * f_c(c))

    return r_s, rho_s_Msun_kpc3


def nfw_delta_sigma_analytical(R_kpc, r_s_kpc, rho_s):
    """
    Analytical NFW ΔΣ(R) from Wright & Brainerd (2000).

    ΔΣ(R) = Σ̄(<R) - Σ(R)

    Parameters
    ----------
    R_kpc : array
        Projected radii [kpc]
    r_s_kpc : float
        Scale radius [kpc]
    rho_s : float
        Characteristic density [Msun/kpc^3]

    Returns
    -------
    DeltaSigma : array
        Excess surface density [Msun/pc^2]
    """
    x = R_kpc / r_s_kpc

    # Σ(R)
    Sigma = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 0.99:
            arg = np.sqrt((1 - xi) / (1 + xi))
            f = (1 - (2 / np.sqrt(1 - xi**2)) * np.arctanh(arg)) / (xi**2 - 1)
        elif xi > 1.01:
            arg = np.sqrt((xi - 1) / (xi + 1))
            f = (1 - (2 / np.sqrt(xi**2 - 1)) * np.arctan(arg)) / (xi**2 - 1)
        else:
            f = 1.0 / 3.0
        Sigma[i] = 2 * r_s_kpc * rho_s * f

    # Σ̄(<R)
    Sigma_mean = np.zeros_like(x)
    for i, xi in enumerate(x):
        if xi < 0.99:
            arg = np.sqrt((1 - xi) / (1 + xi))
            g = (np.log(xi / 2) + (2 / np.sqrt(1 - xi**2)) * np.arctanh(arg)) / xi**2
        elif xi > 1.01:
            arg = np.sqrt((xi - 1) / (xi + 1))
            g = (np.log(xi / 2) + (2 / np.sqrt(xi**2 - 1)) * np.arctan(arg)) / xi**2
        else:
            g = 1 + np.log(0.5)
        Sigma_mean[i] = 4 * r_s_kpc * rho_s * g / xi**2

    # Convert from Msun/kpc^2 to Msun/pc^2
    # 1 kpc = 1000 pc → 1 kpc^2 = 10^6 pc^2
    DeltaSigma_kpc2 = Sigma_mean - Sigma
    DeltaSigma_pc2 = DeltaSigma_kpc2 / 1e6

    return DeltaSigma_pc2


def fit_nfw_cluster(
    R_kpc: np.ndarray,
    DeltaSigma_obs: np.ndarray,
    DeltaSigma_err: np.ndarray,
    z_lens: float,
    initial_M200_1e14: float = 10.0,
    initial_c: Optional[float] = None,
    Om0: float = 0.3,
    Ode0: float = 0.7
) -> NFWFit:
    """
    Fit NFW profile to cluster lensing data.

    Parameters
    ----------
    R_kpc : array
        Projected radii [kpc]
    DeltaSigma_obs : array
        Observed ΔΣ(R) [Msun/pc^2]
    DeltaSigma_err : array
        Uncertainties on ΔΣ(R) [Msun/pc^2]
    z_lens : float
        Cluster redshift
    initial_M200_1e14 : float
        Initial guess for M200 [10^14 Msun]
    initial_c : float, optional
        Initial guess for c (default: Duffy08 relation)
    Om0, Ode0 : float
        Cosmology

    Returns
    -------
    result : NFWFit
        Best-fit NFW parameters and statistics
    """
    # Initial guess for concentration
    if initial_c is None:
        initial_c = concentration_duffy08(initial_M200_1e14, z_lens)

    # Grid search (simple alternative to scipy.optimize.minimize)
    M200_grid = np.logspace(np.log10(initial_M200_1e14 * 0.3), np.log10(initial_M200_1e14 * 3), 30)
    c_grid = np.logspace(np.log10(max(initial_c * 0.3, 1.0)), np.log10(min(initial_c * 3, 20)), 25)

    chi2_best = np.inf
    M200_best, c_best = initial_M200_1e14, initial_c

    for M200 in M200_grid:
        for c in c_grid:
            # Compute NFW prediction
            r_s, rho_s = nfw_params_from_M200_c(M200, c, z_lens, Om0, Ode0)
            DeltaSigma_pred = nfw_delta_sigma_analytical(R_kpc, r_s, rho_s)

            # Chi-squared
            chi2 = np.sum(((DeltaSigma_obs - DeltaSigma_pred) / DeltaSigma_err)**2)

            if chi2 < chi2_best:
                chi2_best = chi2
                M200_best, c_best = M200, c

    # Compute best-fit NFW parameters
    r_s_best, rho_s_best = nfw_params_from_M200_c(M200_best, c_best, z_lens, Om0, Ode0)
    r200_best = r200_from_M200(M200_best, z_lens, Om0, Ode0)

    # Statistics
    n_data = len(R_kpc)
    n_params = 2  # M200, c
    dof = n_data - n_params

    # BIC = -2 ln L + k ln n
    # For Gaussian likelihood: -2 ln L = χ²
    BIC = chi2_best + n_params * np.log(n_data)

    return NFWFit(
        M200=M200_best,
        c=c_best,
        r200=r200_best,
        r_s=r_s_best,
        rho_s=rho_s_best,
        chi2=chi2_best,
        dof=dof,
        n_params=n_params,
        BIC=BIC
    )


if __name__ == "__main__":
    # Test with mock cluster
    print("=" * 70)
    print("NFW LENSING BASELINE: TEST")
    print("=" * 70)

    # Mock data: NFW with M200=10, c=4
    z_lens = 0.3
    M200_true = 10.0  # 10^14 Msun
    c_true = 4.0

    r_s_true, rho_s_true = nfw_params_from_M200_c(M200_true, c_true, z_lens)
    r200_true = r200_from_M200(M200_true, z_lens)

    print(f"\nTrue NFW parameters:")
    print(f"  M200 = {M200_true:.2f} × 10^14 Msun")
    print(f"  c    = {c_true:.2f}")
    print(f"  r200 = {r200_true:.1f} kpc")
    print(f"  r_s  = {r_s_true:.1f} kpc")

    # Generate mock data
    R_mock = np.logspace(np.log10(100), np.log10(1500), 12)
    DeltaSigma_true = nfw_delta_sigma_analytical(R_mock, r_s_true, rho_s_true)
    DeltaSigma_err = 0.1 * DeltaSigma_true + 1e7  # 10% + floor
    DeltaSigma_obs = DeltaSigma_true + np.random.randn(len(R_mock)) * DeltaSigma_err

    # Fit NFW
    fit = fit_nfw_cluster(R_mock, DeltaSigma_obs, DeltaSigma_err, z_lens)

    print(f"\nBest-fit NFW parameters:")
    print(f"  M200 = {fit.M200:.2f} × 10^14 Msun  (true: {M200_true:.2f})")
    print(f"  c    = {fit.c:.2f}  (true: {c_true:.2f})")
    print(f"  r200 = {fit.r200:.1f} kpc  (true: {r200_true:.1f} kpc)")
    print(f"  r_s  = {fit.r_s:.1f} kpc  (true: {r_s_true:.1f} kpc)")
    print(f"\nFit statistics:")
    print(f"  χ² = {fit.chi2:.2f}")
    print(f"  dof = {fit.dof}")
    print(f"  χ²/dof = {fit.chi2/fit.dof:.3f}")
    print(f"  BIC = {fit.BIC:.2f}")

    # Fractional errors
    frac_M = abs(fit.M200 - M200_true) / M200_true * 100
    frac_c = abs(fit.c - c_true) / c_true * 100

    print(f"\nRecovery:")
    print(f"  ΔM200/M200 = {frac_M:.1f}%")
    print(f"  Δc/c = {frac_c:.1f}%")

    if frac_M < 20 and frac_c < 30:
        print("\n✓ PASS: NFW fitter recovers parameters")
    else:
        print("\n✗ FAIL: NFW fitter does not recover parameters")

    print("=" * 70 + "\n")
