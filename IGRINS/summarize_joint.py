"""Summarize matched stellar/slab/profile fits and reproducible inputs."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TAGS = ["joint_nostar_slab", "joint_star_slab", "joint_star_profile"]
LABELS = ["No star + slab", "PHOENIX + slab", "PHOENIX + 20 layers"]


def main():
    rows = []
    records = []
    for tag, label in zip(TAGS, LABELS):
        result = json.loads((OUT / f"{tag}.json").read_text())
        data = np.load(OUT / f"{tag}.npz")
        fixed_residual = data["observed"] - data["model"]
        row = {"model": label}
        for phase, key in [("train", "train_mask"), ("validation", "score_mask")]:
            mask = data[key]
            row[f"{phase}_fixed_scale_rms_percent"] = float(100*np.sqrt(np.mean(fixed_residual[mask]**2)))
            row[f"{phase}_continuum_rms_percent"] = float(100*np.sqrt(np.mean(data["residual_continuum"][mask]**2)))
        rows.append(row)
        records.append((result, data))
    with (OUT / "joint_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    orders = records[0][0]["orders"]
    positions = np.arange(len(orders))
    width = .24
    for k, ((result, data), label) in enumerate(zip(records, LABELS)):
        residual = data["observed"] - data["model"]
        for ax, key in zip(axes, ["train_mask", "score_mask"]):
            values = [100*np.sqrt(np.mean(residual[b][data[key][b]]**2)) for b in range(len(orders))]
            ax.bar(positions+(k-1)*width, values, width=width, label=label)
    axes[0].set_title("Training: exposures 0 and 78")
    axes[1].set_title("Conditional validation: exposure 39")
    axes[0].set_ylabel("RMS (% of fixed observed flux scale)")
    for ax in axes:
        ax.set_xticks(positions, [f"Order {o}" for o in orders])
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    fig.suptitle("WASP-77A / IGRINS: matched pixels and normalization")
    fig.tight_layout(); fig.savefig(OUT / "joint_comparison.png", dpi=170); plt.close(fig)

    profile_result = records[-1][0]
    times = json.loads((OUT / "time_metadata.json").read_text())
    weather = json.loads((ROOT / "cache/atmosphere_profile.json").read_text())
    stellar = json.loads((ROOT / "cache/stellar_phoenix.json").read_text())
    checks = json.loads((OUT / "joint_output_checks.json").read_text())
    script_names = ["fit_joint.py", "prepare_stellar.py", "stellar_template.py",
                    "prepare_atmosphere.py", "atmosphere_profile.py", "hitran_cache.py",
                    "check_joint.py", "check_joint_outputs.py", "summarize_joint.py"]
    environment = dict(python=sys.version, executable=sys.executable,
                       package_versions={name: importlib.metadata.version(name) for name in
                                         ["jax", "jaxlib", "numpy", "scipy", "astropy", "xarray", "hitran-api"]},
                       scripts_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                                       for name in script_names},
                       inputs={"stellar": stellar, "meteorology": weather},
                       results_checks=checks["status"])
    (OUT / "joint_environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    report = f"""WASP-77A / IGRINS: stellar spectrum and terrestrial T-P implementation

Result
------
The stellar template removes most of the previously unexplained deep
absorption. A fixed dated multi-layer atmosphere further improves the
conditional prediction of an exposure excluded from shared-parameter fitting.
This is a telluric/stellar foreground experiment, not a planet retrieval.

Model                                    Train RMS     Validation RMS
                                         fixed observed flux scale (%)
"""
    for row in rows:
        report += (f"{row['model']:40s}{row['train_fixed_scale_rms_percent']:10.4f}"
                   f"{row['validation_fixed_scale_rms_percent']:16.4f}\n")
    report += "\nFor reference, RMS relative to each fitted continuum (%):\n"
    for row in rows:
        report += (f"{row['model']:40s}{row['train_continuum_rms_percent']:10.4f}"
                   f"{row['validation_continuum_rms_percent']:16.4f}\n")
    report += f"""
Matched comparison
------------------
Orders13,34,40; pixels600:1500 (900 pixels/order); H2O and CO2 opacity.
Train on exposures0 and78. Reserve exposure39 for conditional validation.
Shared stellar and atmospheric/instrumental parameters use training exposures
only. Validation water amplitude, a common wavelength offset, and independent
cubic continua use alternating80-pixel calibration blocks with5-pixel guards.
The other guarded blocks (1095 pixels total) are scored. The validation
observed-flux normalization uses only calibration pixels. No residual clipping.
This is conditional prediction after nuisance calibration, not a completely
unseen exposure. Wavelengths in training spectra overlap validation wavelengths.
All3 models use the same flux arrays, masks, continuum basis and normalization.
These joint-slab results are not directly equivalent to the earlier individual
order fits, which had independent atmospheric T/P and log-cubic continua.

info.csv contains the dates
---------------------------
The column is Time (BJD); it also includes phase, RV, airmass, humidity andSNR.
The CSV does not specify BJD_TDB vsBJD_UTC or exposure midpoint convention.
Assuming BJD_TDB and iteratively subtracting barycentric light-travel time:
first frame: {times['utc_start_assumed']} UTC
last frame:  {times['utc_end_assumed']} UTC
The printed milliseconds are numerical precision, not established accuracy.
BJD_UTC would shift inferred UTC by approximately69s, small compared with
the6-hourly meteorological data. Per-frame conversion: time_metadata.csv.
CSV RV + Astropy barycentric correction is approximately1.683km/s, consistent
with the published systemic1.6845km/s. Thus the observed stellar relative RV
uses the positive CSV variation; absolute stellar RV is still fitted.

Stellar model
-------------
PHOENIX HiRes: Teff5600K, logg4.5, solar metal abundance; vacuum wavelengths.
Native spectral sampling retained; supplied library spectra have no addedIP.
Normalize by a smooth pseudo-continuum from99.5percentiles in500km/s bins.
Fixed rotation v sin i=4km/s with linear limb-darkening coefficient0.6;
one common Gaussian extra stellar width is fitted, distinct from instrumentalIP.
One common stellar RV plus the measured relative time variation.
Stellar T(P), abundances and individual line strengths are not retrieved.
The stellar spectrum is multiplied by atmospheric transmission BEFORE IP.
No empirical per-pixel stellar template is learned from the observations.

Earth atmosphere
----------------
Actual dated NOAA PSL NCEP/NCAR Reanalysis1 fields at2020-12-14 00/06UTC.
Four surrounding2.5-degree cells are bilinearly interpolated to GeminiSouth;
the two times are interpolated to the middle exposure and held fixed allnight.
This is coarse reanalysis, NOT an on-site sounding or surface measurement.
Temperature/geopotential:17pressure levels; specific humidity:8levels.
The site pressure inferred from height and altitude2722m is
{weather['surface_pressure_bar']:.6f}bar. Layers below the observatory are excluded.
10hPa-to1hPa T is extended toward270.65K; water above300hPa is extended toward
3ppm by100hPa. These upper extensions are explicit assumptions. Above1hPa omitted.
Dry trace abundances are reference values, not meteorological measurements.
CO2 reference410ppm times a night-shared fit multiplier. H2O has one
column/PWV multiplier per exposure, with fixed relative vertical distribution.
Hydrostatic layer columns use dP/(g*mean_air_molecular_mass), including the
reference water dependence of mean mass. No extra inverse-temperature factor.
Pressure/temperature/air widths remain physical layer quantities.
The slab comparison instead fits a single common effective T and P.
Default comparisons omit self broadening and pressure shift in BOTH models.
Optional --self-broadening uses widths at the reference composition, frozen
as column amplitudes change; --shift applies delta_air times air pressure/atm.
No free temperature perturbation was needed for this first model comparison.

Optimization and checks
-----------------------
Existing ExoJAX OpaDirect and instrumental kernels; float64JAX onCPU.
Fixed-profile vertical optical-depth templates are cached by data hashes.
Independent cubic linear continua are solved analytically at every evaluation;
the exact JAX Jacobian includes the continuum solution's parameter dependence.
Bounded scipy least_squares TRF fits shared parameters then validation nuisance
parameters. Scalar finite differences check autodiff directional derivatives.
All3 fits and all validation calibrations converged. No fitted bound was hit
in the final PHOENIX/profile model. Output arrays are finite and continua positive.
Saved-output checks verify identical observed arrays, disjoint masks, no heldout
normalization leakage, reference column conservation and stored RMS consistency.
The original observational pickle and CSV files are unchanged.
Layer convergence below uses fixed physical parameters AND fixed continua:
"""
    for n, check in profile_result["layer_convergence"].items():
        report += f"{n:>2s} layers vs20: RMS difference {100*check['rms_model_difference']:.5f}%, max {100*check['max_model_difference']:.5f}% continuum\n"
    report += "\nFitted PWV (model-dependent, millimeters):\n"
    for exposure, column in zip(profile_result["train_exposures"]+[profile_result["heldout_exposure"]],
                               profile_result["vertical_columns_cm-2"]["H2O"]):
        report += f"Exposure {exposure}: {column*18.01528*1.66053906660e-24*10:.4f}\n"
    report += """
Reproduce from repository root
------------------------------
python IGRINS/prepare_stellar.py
python IGRINS/prepare_atmosphere.py
python IGRINS/fit_joint.py --atmosphere slab --no-star
python IGRINS/fit_joint.py --atmosphere slab
python IGRINS/fit_joint.py --atmosphere profile --check-layers
python IGRINS/check_joint.py
python IGRINS/check_joint_outputs.py
python IGRINS/summarize_joint.py
Network is required only for missing input caches. Warm-start --initial accepts
an existing result JSON. --orders, --train, --holdout and --pixels select data.
Main outputs: joint_*.json/.npz/.png; comparison: joint_comparison.csv/.png.
Python/package versions, script hashes and input provenance: joint_environment.json.
The archived joint_star_profile_reference_self experiment used frozen reference
self widths; it is outside the matched default table above.

Limits and next step
-------------------
The remaining1-3percent residuals are still correlated, with visible mismatch
in individual stellar features and telluric shapes. No pixel uncertainties are
supplied, so these RMS values are not reducedchi2 or significance estimates.
PHOENIX stellar parameters, pseudo-continuum, fixedrotation and fixed nightly
IP remain approximations. Coarse reanalysis may miss local temperature/humidity
structure. Planet emission and extraction systematics remain unmodeled.
Further work can test neighboring stellar templates and a constrained nightly
temperature correction, then profile evolution/IP changes, with the same heldout
protocol. The present results cover3orders and3exposures, not the whole cube.

Sources
-------
Observations and system: https://arxiv.org/html/2312.13069v1
Data: https://zenodo.org/records/10382053
PHOENIX library: https://arxiv.org/html/1303.5632v2
NOAA reanalysis: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html
Raw download URLs andSHA256 are in cache/stellar_phoenix.json and atmosphere_*.json.
"""
    (OUT / "REPORT_JOINT.txt").write_text(report)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
