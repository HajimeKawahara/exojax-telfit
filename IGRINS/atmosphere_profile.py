"""Load a dated terrestrial profile on pressure layers above Gemini South."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"


def load_profile(nlayer: int = 20) -> dict:
    """Return hydrostatic layers; see cache metadata for measured/modelled fields."""
    if nlayer < 4:
        raise ValueError("Use at least four atmospheric layers")
    metadata = json.loads((CACHE / "atmosphere_profile.json").read_text())
    with np.load(CACHE / "atmosphere_profile.npz") as data:
        p_native = data["pressure_bar"][::-1]
        temperature_native = data["temperature_K"][::-1]
        p_water = data["humidity_pressure_bar"][::-1]
        water_native = data["water_vmr"][::-1]
        surface_pressure = float(data["surface_pressure_bar"])
    # Resolve the wet, pressure-broadened lower atmosphere without spending
    # most layers on the small dry column at very low pressures.
    upper_layers = max(2, nlayer // 4)
    edges = np.r_[np.geomspace(.001, .1, upper_layers + 1),
                  np.linspace(.1, surface_pressure, nlayer - upper_layers + 1)[1:]]
    pressure = np.sqrt(edges[1:] * edges[:-1])
    temperature = np.interp(np.log(pressure), np.log(np.r_[.001, p_native]),
                            np.r_[270.65, temperature_native])
    water = np.exp(np.interp(np.log(pressure), np.log(np.r_[.001, .1, p_water]),
                             np.log(np.r_[3e-6, 3e-6, water_native])))
    # Trace-gas VMRs are referenced to dry air. Their dry-to-moist conversion is
    # negligible at this precision but included consistently in the profile.
    vmr = {name: value * (1 - water)
           for name, value in metadata["reference_dry_vmr"].items()}
    vmr["H2O"] = water
    mean_mass = 28.9647 * (1 - water) + 18.01528 * water
    column = 1e6 * np.diff(edges) / (metadata["gravity_cgs"] * mean_mass * 1.66053906660e-24)
    metadata = dict(metadata, nlayer=nlayer,
                    layer_grid="Log pressure from1 to100hPa; linear pressure from100hPa to site",
                    pressure_top_boundary_bar=float(edges[0]),
                    pressure_bottom_boundary_bar=float(edges[-1]),
                    reference_vertical_columns_cm2={name: float(np.sum(column * values))
                                                    for name, values in vmr.items()},
                    reference_pwv_mm=float(np.sum(column * water) * 18.01528 * 1.66053906660e-24 * 10))
    return {"pressure_bar": pressure, "delta_pressure_bar": np.diff(edges),
            "temperature_K": temperature, "vmr": vmr,
            "mean_molecular_weight": mean_mass, "number_column_cm2": column,
            "pressure_boundaries_bar": edges, "gravity_cgs": metadata["gravity_cgs"],
            "metadata": metadata}
