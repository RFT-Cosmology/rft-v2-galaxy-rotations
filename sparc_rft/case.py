from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "cases" / "GalaxyCase.schema.json"


Number = float


def _to_float_list(name: str, values: Sequence[Any]) -> List[float]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be an array of numbers")
    result: List[float] = []
    for idx, item in enumerate(values):
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            raise ValueError(f"{name}[{idx}] is not a finite number") from None
    if not result:
        raise ValueError(f"{name} must contain at least one entry")
    return result


def _validate_lengths(lengths: Mapping[str, int]) -> None:
    unique_lengths = {length for length in lengths.values()}
    if len(unique_lengths) != 1:
        lengths_repr = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise ValueError(f"All radial arrays must have the same length ({lengths_repr})")


def _coerce_optional_float(label: str, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric when provided") from exc


@dataclass(frozen=True)
class GalaxyCase:
    name: str
    r_kpc: List[float]
    v_obs_kms: List[float]
    sigma_v_kms: List[float]
    v_baryon_disk_kms: List[float]
    v_baryon_gas_kms: List[float]
    v_baryon_bulge_kms: Optional[List[float]] = None
    distance_mpc: Optional[float] = None
    inclination_deg: Optional[float] = None
    notes: str = ""

    def as_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        if self.v_baryon_bulge_kms is None:
            payload.pop("v_baryon_bulge_kms", None)
        if self.distance_mpc is None:
            payload.pop("distance_mpc", None)
        if self.inclination_deg is None:
            payload.pop("inclination_deg", None)
        if not self.notes:
            payload.pop("notes", None)
        return payload

    @property
    def n_points(self) -> int:
        return len(self.r_kpc)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GalaxyCase":
        data = validate_case_payload(payload)
        return cls(**data)


def validate_case_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Case payload must be a mapping")

    required = [
        "name",
        "r_kpc",
        "v_obs_kms",
        "sigma_v_kms",
        "v_baryon_disk_kms",
        "v_baryon_gas_kms",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    name = payload["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")

    arrays = {
        "r_kpc": _to_float_list("r_kpc", payload["r_kpc"]),
        "v_obs_kms": _to_float_list("v_obs_kms", payload["v_obs_kms"]),
        "sigma_v_kms": _to_float_list("sigma_v_kms", payload["sigma_v_kms"]),
        "v_baryon_disk_kms": _to_float_list("v_baryon_disk_kms", payload["v_baryon_disk_kms"]),
        "v_baryon_gas_kms": _to_float_list("v_baryon_gas_kms", payload["v_baryon_gas_kms"]),
    }

    bulge_raw = payload.get("v_baryon_bulge_kms")
    bulge_list: Optional[List[float]] = None
    if bulge_raw is not None:
        bulge_list = _to_float_list("v_baryon_bulge_kms", bulge_raw)
        arrays["v_baryon_bulge_kms"] = bulge_list

    _validate_lengths({key: len(val) for key, val in arrays.items()})

    distance = _coerce_optional_float("distance_mpc", payload.get("distance_mpc"))
    inclination = _coerce_optional_float("inclination_deg", payload.get("inclination_deg"))

    notes = payload.get("notes") or ""
    if not isinstance(notes, str):
        raise ValueError("notes must be a string if provided")

    return {
        "name": name.strip(),
        "r_kpc": arrays["r_kpc"],
        "v_obs_kms": arrays["v_obs_kms"],
        "sigma_v_kms": arrays["sigma_v_kms"],
        "v_baryon_disk_kms": arrays["v_baryon_disk_kms"],
        "v_baryon_gas_kms": arrays["v_baryon_gas_kms"],
        "v_baryon_bulge_kms": bulge_list,
        "distance_mpc": distance,
        "inclination_deg": inclination,
        "notes": notes.strip(),
    }


def load_case(path: Any) -> GalaxyCase:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return GalaxyCase.from_dict(payload)


def dump_case(case: GalaxyCase, path: Any) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(case.as_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")


def case_arrays(case: GalaxyCase) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], np.ndarray]:
    r = np.asarray(case.r_kpc, dtype=float)
    disk = np.asarray(case.v_baryon_disk_kms, dtype=float)
    gas = np.asarray(case.v_baryon_gas_kms, dtype=float)
    bulge = None
    if case.v_baryon_bulge_kms is not None:
        bulge = np.asarray(case.v_baryon_bulge_kms, dtype=float)
    return r, disk, bulge, gas


def baryon_speed_squared(
    disk: np.ndarray,
    gas: np.ndarray,
    bulge: Optional[np.ndarray] = None,
) -> np.ndarray:
    total = disk ** 2 + gas ** 2
    if bulge is not None:
        total = total + bulge ** 2
    return total


def baryon_speed(case: GalaxyCase) -> np.ndarray:
    r, disk, bulge, gas = case_arrays(case)
    total = baryon_speed_squared(disk, gas, bulge)
    return np.sqrt(np.clip(total, 0.0, None))
