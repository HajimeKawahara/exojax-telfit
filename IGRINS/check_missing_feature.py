"""Check omitted HITRAN species around the persistent order34 residual."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import numpy as np

from atmosphere_profile import load_profile
from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss
from exojax.utils.constants import c
from fit_joint import profile_tau
from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent


def main():
    result = json.loads((ROOT / "results/joint_star_profile.json").read_text())
    p = result["parameters"]
    wavelength = 1.5735264701102794
    index = 605
    x = 2 * index / 899 - 1
    observed_nu = 1e4 / wavelength
    sampled_nu = observed_nu * (1 + p["stretch_o34_kms"] * x / c) * (1 + p["calibration_rv_o34_kms"] / c)
    velocity = np.linspace(-100., 100., 802)
    grid = sampled_nu * np.exp(velocity / c)
    dv = velocity[1] - velocity[0]
    kernel = np.arange(-int(25/dv), int(25/dv)+1) * dv
    profile = load_profile(20)
    airmass = 1.127
    total = np.zeros(len(grid))
    records = {}
    for species in ["CH4", "N2O", "CO"]:
        mdb = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-30)
        if len(mdb.nu_lines):
            opa = OpaDirect(mdb, grid)
            tau, column = profile_tau(opa, grid, profile, species)
        else:
            tau = np.zeros(len(grid))
            column = float(np.sum(profile["number_column_cm2"] * profile["vmr"][species]))
        total += tau
        absorption = 1 - np.asarray(ipgauss(np.exp(-airmass*tau), kernel, p["ip_sigma_o34_kms"]))
        records[species] = {
            "n_lines": len(mdb.nu_lines), "reference_vertical_column_cm2": column,
            "depth_percent_at_residual_peak": float(100*np.interp(sampled_nu, grid, absorption)),
            "max_depth_percent_within_10_kms": float(100*np.max(absorption[np.abs(velocity) <= 10])),
        }
    for scale in [1., 10.]:
        absorption = 1 - np.asarray(ipgauss(np.exp(-airmass*scale*total), kernel, p["ip_sigma_o34_kms"]))
        records[f"combined_{scale:g}x_reference"] = {
            "depth_percent_at_residual_peak": float(100*np.interp(sampled_nu, grid, absorption)),
            "max_depth_percent_within_10_kms": float(100*np.max(absorption[np.abs(velocity) <= 10])),
        }
    report = {"observed_wavelength_um": wavelength, "corrected_sampling_wavenumber_cm1": sampled_nu,
              "exposure": 0, "airmass": airmass, "ip_sigma_kms": p["ip_sigma_o34_kms"],
              "line_threshold_cm": 1e-30, "results": records,
              "scope": "Existing cached HITRAN lines, nominal dated-profile dry abundances. This bounds the local depth only under these model assumptions; no atomic identification is attempted."}
    (ROOT / "results/missing_feature_check.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
