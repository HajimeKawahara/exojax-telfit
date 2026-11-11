"""Measure the spectral effect of restoring the dry-air mass budget after H2O scaling.

Only dry molecular layer columns change. Temperatures, pressure broadening,
stellar parameters, instrumental parameters, and fitted continua remain fixed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import numpy as np
import pandas as pd

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import cross_section
from hitran_cache import load_mdb
from profile_opacity import prepare_profile_opacity
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="inv5_tp3")
    args = parser.parse_args()
    meta_path = OUT / f"{args.tag}.json"
    meta = json.loads(meta_path.read_text())
    if any(meta.get(key, False) for key in ["direct_ip", "ip_epoch", "ip_asymmetry", "ip_wavelength"]):
        raise ValueError("This check uses the constant symmetric instrumental model.")
    with np.load(OUT / f"{args.tag}.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    profile = load_profile(meta["layers"])
    scale = meta["pressure_scale"]
    for key in ["pressure_bar", "delta_pressure_bar", "pressure_boundaries_bar", "number_column_cm2"]:
        profile[key] = profile[key] * scale
    profile["temperature_K"] += meta["temperature_offset_K"]
    state = meta["fitted_profile"]
    delta = np.asarray(state["temperature_delta_K"])
    water_weights = np.asarray(state["water_layer_weights"])
    dry_ratio = np.asarray(state["mass_conserving_dry_column_ratio"])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    parameters = meta["parameters"]
    baseline = np.zeros_like(data["component"])
    corrected = np.zeros_like(baseline)
    for b, order in enumerate(data["orders"]):
        nu = 1e4/data["wavelength_um"][b]
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/meta["dv_kms"]))+1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = c*np.log(grid[1]/grid[0])
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, parameters["stellar_extra_sigma_kms"])
        layers = {}
        for species in meta["species"]:
            margin = meta["line_margin_cm1"]
            opa = OpaDirect(load_mdb(species, grid[0]-margin, grid[-1]+margin,
                                     crit=meta["line_cutoff"]), grid)
            table = prepare_profile_opacity(opa, grid, profile, species, cross_section,
                                             meta["pressure_shift"], meta["reference_self_broadening"])
            layers[species] = np.asarray(table.layer_tau(delta))
        x = np.linspace(-1, 1, len(nu))
        target = nu * (1+(parameters[f"stretch_o{order}_kms"]*x
                         + parameters.get(f"quadratic_o{order}_kms", 0.)*(2*x*x-1))/c)
        width = parameters[f"ip_sigma_o{order}_kms"]
        fraction = parameters.get("ip_wing_fraction", 0.)
        for e, exposure in enumerate(data["exposures"]):
            stellar_rv = parameters["stellar_rv_reference_kms"] + float(
                info.iloc[exposure]["RV [km/s]"]-info.iloc[data["exposures"][0]]["RV [km/s]"])
            shifted_star = sampling(grid, grid, star, stellar_rv)
            airmass = float(info.iloc[exposure]["Air Mass"])
            rv = parameters[f"calibration_rv_o{order}_kms"] + parameters.get(f"calibration_rv_e{exposure}_kms", 0.)
            reference_tau = np.zeros(n)
            consistent_tau = np.zeros(n)
            for species, values in layers.items():
                factor = 10**parameters[f"log10_water_e{exposure}" if species == "H2O" else f"log10_scale_{species}"]
                weights = water_weights if species == "H2O" else np.ones(len(delta))
                new_weights = weights if species == "H2O" else dry_ratio[e]
                reference_tau += factor*np.sum(values*weights[:, None], axis=0)
                consistent_tau += factor*np.sum(values*new_weights[:, None], axis=0)
            for tau, output in [(reference_tau, baseline), (consistent_tau, corrected)]:
                intrinsic = shifted_star*jnp.exp(-airmass*tau)
                core = sampling(target, grid, ipgauss(intrinsic, velocity, width), rv)
                wing = sampling(target, grid, ipgauss(intrinsic, velocity, 2*width), rv)
                output[b, e] = np.asarray((1-fraction)*core+fraction*wing)
    reconstruction_error = float(np.max(np.abs(baseline-data["component"])))
    np.testing.assert_allclose(baseline, data["component"], rtol=0, atol=1e-11)
    difference = corrected-baseline
    changed_model = corrected*data["continuum"]
    result = dict(source_fit=args.tag, source_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
                  reconstructed_component_max_error=reconstruction_error,
                  rms_component_difference=float(np.sqrt(np.mean(difference**2))),
                  max_component_difference=float(np.max(np.abs(difference))),
                  maximum_dry_layer_column_fractional_change=float(np.max(np.abs(dry_ratio-1))), metrics={},
                  interpretation="Read-only dry-molecular-column mass-consistency check. No parameter or continuum refit; pressure-broadening prescription unchanged.")
    for key in ["train_mask", "score_mask"]:
        mask = data[key]
        before = data["observed"]-data["model"]
        after = data["observed"]-changed_model
        result["metrics"][key] = dict(n_pixels=int(mask.sum()),
                                        baseline_fixed_scale_rms=float(np.sqrt(np.mean(before[mask]**2))),
                                        consistent_fixed_scale_rms=float(np.sqrt(np.mean(after[mask]**2))),
                                        rms_component_difference=float(np.sqrt(np.mean(difference[mask]**2))),
                                        max_component_difference=float(np.max(np.abs(difference[mask]))))
    (OUT / "hydrostatic_column_check.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
