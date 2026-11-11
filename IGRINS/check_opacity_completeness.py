"""Audit omitted species, weak lines and distant wings on fixed fitted pixels.

This diagnostic adds opacity at nominal abundances without changing the fitted
stellar model, atmospheric parameters or IP. It is not a new retrieval.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
MISSING_SPECIES = ["CH4", "N2O", "CO", "O2", "O3"]


def reference_missing_profile(nlayer=20):
    """Attach documented RFM O2/O3 shapes to the existing dated P/T grid."""
    profile = load_profile(nlayer)
    raw = (ROOT / "cache/rfm_midlatitude_night.atm").read_text()
    fields = {}
    key = None
    for line in raw.splitlines():
        line = line.split("!")[0].strip()
        if not line:
            continue
        if line.startswith("*"):
            key = line.split()[0][1:]
            fields[key] = []
        elif key:
            fields[key].extend(float(value) for value in line.split())
    pressure = np.asarray(fields["PRE"])[::-1] / 1000
    for species in ["O2", "O3"]:
        vmr = np.asarray(fields[species])[::-1] * 1e-6
        profile["vmr"][species] = np.exp(np.interp(
            np.log(profile["pressure_bar"]), np.log(pressure), np.log(vmr)))
    profile["metadata"]["additional_species_profile"] = {
        "source": "https://eodg.atm.ox.ac.uk/RFM/atm/ngt.atm",
        "description": "MIPAS 2001 midlatitude night O2/O3 VMR; interpolated in log pressure/log VMR",
        "limitation": "Reference composition, not measured at Cerro Pachon in 2020; same 1 hPa top as other species"}
    return profile


def subset_mdb(db, mask):
    """Subset all line-indexed arrays, including isotope partition ratios."""
    result = copy.copy(db)
    n = len(db.nu_lines)
    for name, value in vars(db).items():
        if hasattr(value, "shape") and value.shape == (n,):
            setattr(result, name, value[mask])
    result.qr_interp_lines = lambda t, reference=296.: db.qr_interp_lines(t, reference)[mask]
    return result


def selected_tau(species, grid, profile, selector=None, margin=3.):
    """Sum line chunks so complete lists do not allocate an excessive matrix."""
    db = load_mdb(species, grid[0]-margin, grid[-1]+margin, crit=0.)
    mask = np.asarray(db.Sij0) > 0
    if selector is not None:
        mask &= selector(db)
    db = subset_mdb(db, mask)
    tau = np.zeros(len(grid))
    for start in range(0, len(db.nu_lines), 900):
        chunk = subset_mdb(db, np.arange(len(db.nu_lines))[start:start+900])
        tau += profile_tau(OpaDirect(chunk, grid), grid, profile, species,
                           shifted=True, self_broadening=False)[0]
    return tau, len(db.nu_lines)


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def main():
    meta = json.loads((OUT / "focus_ip.json").read_text())
    p = meta["parameters"]
    with np.load(OUT / "focus_ip.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(OUT / "telluric_mask.npz") as saved:
        allowed = saved["mask"]
    profile = reference_missing_profile(meta["layers"])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    exposures = data["exposures"]
    orders = data["orders"]
    names = [f"missing_{s}" for s in MISSING_SPECIES] + [
        "missing_all", "weak_1e-30", "weak_all", "wings_25", "complete"]
    components = {name: np.zeros_like(data["component"]) for name in ["baseline"] + names}
    standalone = {s: np.zeros_like(data["component"]) for s in MISSING_SPECIES}
    reference_columns = {s: float(np.sum(profile["number_column_cm2"]*profile["vmr"][s]))
                         for s in MISSING_SPECIES}
    counts = {}
    for b, order in enumerate(orders):
        nu = 1e4 / data["wavelength_um"][b]
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/meta["dv_kms"])) + 1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        stellar = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        stellar = ipgauss(stellar, velocity, p["stellar_extra_sigma_kms"])
        tau = {}
        counts[int(order)] = {}
        for species in ["H2O", "CO2"]:
            selections = {
                "baseline": (lambda db: np.asarray(db.Sij0) >= 1e-28, 3.),
                "weak_1e-30": (lambda db: (np.asarray(db.Sij0) >= 1e-30) & (np.asarray(db.Sij0) < 1e-28), 3.),
                "weak_all": (lambda db: np.asarray(db.Sij0) < 1e-28, 3.),
                "wings_25": (lambda db: (db.nu_lines < grid[0]-3) | (db.nu_lines > grid[-1]+3), 25.)}
            for label, (select, margin) in selections.items():
                key = f"{species}_{label}"
                tau[key], counts[int(order)][key] = selected_tau(species, grid, profile, select, margin)
                print(f"Order {order}: {key} {counts[int(order)][key]} lines", flush=True)
        for species in MISSING_SPECIES:
            tau[species], counts[int(order)][species] = selected_tau(species, grid, profile, margin=25.)
            print(f"Order {order}: {species} {counts[int(order)][species]} lines", flush=True)
        x = np.linspace(-1, 1, len(nu))
        calibrated_nu = nu*(1+p[f"stretch_o{order}_kms"]*x/c)
        beta = p[f"ip_sigma_o{order}_kms"]
        wing = p["ip_wing_fraction"]
        def convolve_sample(intrinsic, exposure):
            convolved = ((1-wing)*ipgauss(intrinsic, velocity, beta)
                         + wing*ipgauss(intrinsic, velocity, 2*beta))
            rv = p[f"calibration_rv_o{order}_kms"] + p.get(f"calibration_rv_e{exposure}_kms", 0.)
            return np.asarray(sampling(calibrated_nu, grid, convolved, rv))
        for e, exposure in enumerate(exposures):
            airmass = float(info.iloc[exposure]["Air Mass"])
            relative_rv = float(info.iloc[exposure]["RV [km/s]"]-info.iloc[exposures[0]]["RV [km/s]"])
            star = sampling(grid, grid, stellar, p["stellar_rv_reference_kms"]+relative_rv)
            water = 10**p[f"log10_water_e{exposure}"]
            co2 = 10**p["log10_scale_CO2"]
            baseline_tau = water*tau["H2O_baseline"] + co2*tau["CO2_baseline"]
            extra = {f"missing_{s}": tau[s] for s in MISSING_SPECIES}
            extra["missing_all"] = sum(tau[s] for s in MISSING_SPECIES)
            for label in ["weak_1e-30", "weak_all", "wings_25"]:
                extra[label] = water*tau[f"H2O_{label}"] + co2*tau[f"CO2_{label}"]
            extra["complete"] = extra["missing_all"] + extra["weak_all"] + extra["wings_25"]
            components["baseline"][b,e] = convolve_sample(star*jnp.exp(-airmass*baseline_tau), exposure)
            for name in names:
                components[name][b,e] = convolve_sample(star*jnp.exp(-airmass*(baseline_tau+extra[name])), exposure)
            for species in MISSING_SPECIES:
                standalone[species][b,e] = 1-convolve_sample(jnp.exp(-airmass*tau[species]), exposure)
    error = float(np.max(np.abs(components["baseline"]-data["component"])))
    np.testing.assert_allclose(components["baseline"], data["component"], atol=2e-12, rtol=2e-12)
    basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, allowed.shape[1]), 3)
    report = dict(
        scope="Orders 13/34/40, pixels600:1500, exposures0/78/20, focus_ip fixed star/IP/column scales",
        reconstruction_max_error=error, line_counts=counts, nominal_vertical_columns_cm2=reference_columns,
        profiles=profile["metadata"]["additional_species_profile"], variants={}, peaks=[],
        assumptions=["Omitted CH4/N2O/CO use existing nominal constant dry VMRs, not fitted abundances",
                     "O2/O3 use a MIPAS reference composition; not a dated retrieval",
                     "All isotope lines available in the cached HITRAN response are retained",
                     "No added pressure self broadening, CIA or continuum opacity in this test",
                     "Distant-wing test expands line-center coverage, not a physical wing truncation prescription",
                     "Direct model differences use frozen parameters; RMS after continuum reprojection changes only polynomial coefficients",
                     "Negative data-model residual can be reduced by added absorption; positive residual cannot be corrected at frozen continuum by added opacity"])
    output = {"orders": orders, "exposures": exposures, "wavelength_um": data["wavelength_um"],
              "included_mask": allowed, "baseline_component": components["baseline"]}
    for name in names:
        delta = components["baseline"]-components[name]
        model = components[name]*data["continuum"]
        projected = np.empty_like(model)
        for b in range(len(orders)):
            for e in range(len(exposures)):
                weights = data["train_mask"][b,e] | data["calibration_mask"][b,e]
                design = basis*components[name][b,e,:,None]
                normal = design[weights].T@design[weights]
                coeff = np.linalg.solve(normal+1e-12*np.eye(4), design[weights].T@data["observed"][b,e,weights])
                projected[b,e] = components[name][b,e]*(basis@coeff)
        selected = np.broadcast_to(allowed[:,None,:], delta.shape)
        report["variants"][name] = dict(
            maximum_added_absorption_percent=float(100*np.max(delta[selected])),
            rms_model_change_percent=100*rms(delta[selected]),
            training_rms_fixed_continuum_percent=100*rms((data["observed"]-model)[data["train_mask"]]),
            heldout_rms_fixed_continuum_percent=100*rms((data["observed"]-model)[data["score_mask"]]),
            training_rms_refit_continuum_percent=100*rms((data["observed"]-projected)[data["train_mask"]]),
            heldout_rms_refit_continuum_percent=100*rms((data["observed"]-projected)[data["score_mask"]]),
            per_order=[dict(order=int(order), maximum_added_absorption_percent=float(100*np.max(delta[b,:,allowed[b]])),
                            rms_model_change_percent=100*rms(delta[b,:,allowed[b]])) for b,order in enumerate(orders)])
        output[f"delta_{name}"] = delta
    for species in MISSING_SPECIES:
        output[f"standalone_{species}"] = standalone[species]
    peaks = json.loads((OUT / "focus_residual_peaks.json").read_text())["peaks"]
    for peak in peaks:
        b = int(np.flatnonzero(orders == peak["order"])[0])
        k = int(np.argmin(abs(data["wavelength_um"][b]-peak["wavelength_um"])))
        item = dict(peak)
        item["added_absorption_percent_by_exposure"] = {
            name: (100*(components["baseline"][b,:,k]-components[name][b,:,k])).tolist()
            for name in names}
        report["peaks"].append(item)
    paths = [Path(__file__), OUT/"focus_ip.json", OUT/"focus_ip.npz", OUT/"telluric_mask.npz",
             ROOT/"cache/rfm_midlatitude_night.atm"] + [ROOT/f"cache/{s}.par" for s in ["H2O", "CO2"]+MISSING_SPECIES]
    report["sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    (OUT/"opacity_completeness.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT/"opacity_completeness.npz", **output)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    colors = dict(zip(["missing_all", "weak_all", "wings_25"], ["tab:red", "tab:blue", "tab:green"]))
    for b,(order,ax) in enumerate(zip(orders, axes)):
        wave = data["wavelength_um"][b]
        for name,color in colors.items():
            ax.plot(wave,100*(components["baseline"][b,1]-components[name][b,1]), color=color, lw=.9, label=name)
        ax.set(title=f"Order {order}; exposure 78; nominal missing-gas columns", ylabel="Added absorption (%)")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Observed vacuum wavelength (micron; assumed convention)")
    fig.savefig(OUT/"opacity_completeness.png", dpi=150)
    plt.close(fig)
    print(json.dumps({name:result["maximum_added_absorption_percent"] for name,result in report["variants"].items()}, indent=2))


if __name__ == "__main__":
    main()
