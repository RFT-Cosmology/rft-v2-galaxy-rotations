from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from .case import GalaxyCase


SolverFn = Callable[[GalaxyCase], List[float]]


def newtonian_baseline(case: GalaxyCase) -> List[float]:
    """Quadrature-add supplied baryonic components."""
    disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
    if disk.shape != gas.shape:
        raise ValueError("Disk and gas arrays must match in length")

    if case.v_baryon_bulge_kms is not None:
        bulge = np.asarray(case.v_baryon_bulge_kms, dtype=float)
        if bulge.shape != disk.shape:
            raise ValueError("Bulge array must match disk length")
    else:
        bulge = np.zeros_like(disk)

    total_sq = np.maximum(disk ** 2 + gas ** 2 + bulge ** 2, 0.0)
    return np.sqrt(total_sq).tolist()


# Note: MOND and NFW are handled specially in CLI (they return params too)
# so they're not in this simple SolverFn registry
SOLVERS: Dict[str, SolverFn] = {
    "newtonian": newtonian_baseline,
}

AVAILABLE_SOLVERS = tuple(list(SOLVERS.keys()) + ["rft_geom", "rft_kernel", "mond", "nfw_fit"])
