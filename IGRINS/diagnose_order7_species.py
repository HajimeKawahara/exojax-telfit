"""Separate physical-initialization telluric transmission by molecule."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np
import pandas as pd

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from line_strength_support import native_grid

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main() -> None:
    source = OUT / "order7_physical_e30.json"
    metadata = json.loads(source.read_text())
    p = metadata["parameters"]
    with np.load(source.with_suffix(".npz")) as saved:
        wave = saved["wavelength_um"][0]
    grid = native_grid(wave, metadata["dv_kms"])
    step = float(c * np.log(grid[1] / grid[0]))
    velocity = jnp.arange(-int(60 / step), int(60 / step) + 1) * step
    profile = load_profile(metadata["layers"])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    airmass = float(info.iloc[30]["Air Mass"])
    width = p["ip_sigma_o5_kms"] * np.exp(p["log_ip_width_e30"])
    offset = p["calibration_rv_o5_kms"] + p["calibration_rv_e30_kms"]
    target = (1e4 / wave) * (1 + p["stretch_o5_kms"] * np.linspace(-1, 1, len(wave)) / c)
    fraction = p["ip_wing_fraction"]

    def response(tau):
        intrinsic = jnp.exp(-jnp.asarray(tau))
        core = sampling(target, grid, ipgauss(intrinsic, velocity, width), offset)
        wing = sampling(target, grid, ipgauss(intrinsic, velocity, 2 * width), offset)
        return np.asarray((1 - fraction) * core + fraction * wing)

    arrays = dict(wavelength_um=wave, cube_index=np.array([5]), display_order=np.array([7]),
                  exposure=np.array([30]))
    statistics, optical_depths = {}, []
    for species in metadata["species"]:
        db = load_mdb(species, grid[0] - metadata["line_margin_cm1"],
                      grid[-1] + metadata["line_margin_cm1"], metadata["line_cutoff"])
        tau, column = profile_tau(OpaDirect(db, grid), grid, profile, species,
                                  metadata["pressure_shift"], metadata["reference_self_broadening"])
        scale = 10 ** p["log10_water_e30" if species == "H2O" else f"log10_scale_{species}"]
        optical_depths.append(airmass * scale * tau)
        transmission = response(optical_depths[-1])
        arrays[f"transmission_{species}"] = transmission
        deepest = int(np.argmin(transmission))
        statistics[species] = dict(line_count=len(db.nu_lines), fitted_profile_scale=scale,
            vertical_column_cm2=float(column * scale), minimum_transmission=float(transmission.min()),
            maximum_absorption_percent=float(100 * (1 - transmission.min())),
            mean_absorption_percent=float(100 * np.mean(1 - transmission)),
            deepest_wavelength_um=float(wave[deepest]), pixels_below_0_95=int(np.sum(transmission < .95)),
            pixels_below_0_5=int(np.sum(transmission < .5)), pixels_below_0_15=int(np.sum(transmission < .15)))
    combined = response(np.sum(optical_depths, axis=0))
    arrays["transmission_all"] = combined
    species = metadata["species"]
    depths = np.array([1 - arrays[f"transmission_{molecule}"] for molecule in species])
    dominant = np.argmax(depths, axis=0)
    arrays["dominant_species_index"] = dominant
    strong = combined < .5
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), display_order=7, cube_index=5,
        exposure=30, stage="Physical initialization before shared stellar/telluric line-strength corrections",
        instrumental_profile="Same fitted symmetric two-Gaussian mixture and wavelength sampling as physical frame 30",
        minimum_combined_transmission=float(combined.min()),
        strongest_combined_absorption_percent=float(100 * (1 - combined.min())),
        combined_pixels_below_0_5=int(strong.sum()), species=species, per_species=statistics,
        strong_pixel_dominant_species_counts={name: int(np.sum(strong & (dominant == i)))
                                              for i, name in enumerate(species)},
        limitations=["Species-isolated transmissions are convolved separately for display; their product is not the combined convolved transmission.",
                     "Dominance here compares modeled isolated depths, not a unique residual attribution.",
                     "The CO and CO2 physical column scales are at the lower fit bound."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [source, source.with_suffix(".npz"), Path(__file__)]})
    np.savez_compressed(OUT / "order7_physical_species.npz", **arrays)
    (OUT / "order7_physical_species.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(per_species=statistics, strong_pixel_dominant_species_counts=result["strong_pixel_dominant_species_counts"]), indent=2))


if __name__ == "__main__":
    main()
