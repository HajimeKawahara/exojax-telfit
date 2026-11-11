"""Collect the reproducible first-pass telluric experiment outputs."""
from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "packages": {name: importlib.metadata.version(name) for name in
                 ["jax", "jaxlib", "numpy", "scipy", "matplotlib", "hitran-api"]},
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "local_source_change": "src/exojax/signal/convolve.py (see REPORT.txt)",
    "device": "CPU, float64",
}
(OUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
rows = []
for path in sorted(OUT.glob("o*.json")):
    result = json.loads(path.read_text())
    if "rms_continuum" not in result:
        continue
    rows.append({
        "run": path.stem, "order": result["order_index"],
        "exposure": result["exposure_index"], "n_pixels": result["n_pixels"],
        "rms_percent_continuum": 100 * result["rms_continuum"],
        "rms_percent_fixed_scale": 100 * result["rms_fixed_scale"],
        "mad_percent_continuum": 100 * result["mad_continuum"],
        "lag1": result["lag1_correlation"], "success": result["success"],
        "nfev": result["nfev"], "bounds_hit": ", ".join(result["bounds_hit"]),
        "jacobian_relative_error": result["jacobian_directional_relative_error"],
    })
with (OUT / "summary.csv").open("w") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

baselines = [r for r in rows if r["exposure"] == 0 and r["run"].endswith("baseline_gaussian_trf")]
fig, axes = plt.subplots(len(baselines), 2, figsize=(14, 2.6 * len(baselines)), squeeze=False)
for axes_row, row in zip(axes, baselines):
    data = np.load(OUT / (row["run"] + ".npz"))
    wave = data["wavelength_um"]
    axes_row[0].plot(wave, data["flux"] / data["continuum"], color="0.3", lw=.65, label="Data / continuum")
    axes_row[0].plot(wave, data["transmission"], color="tab:orange", lw=.8, label="ExoJAX")
    axes_row[0].set_ylabel("Normalized flux")
    axes_row[0].set_title(f"Order {row['order']} | baseline")
    axes_row[1].plot(wave, 100 * data["residual_continuum"], color="0.3", lw=.6)
    axes_row[1].axhline(0, color="tab:orange", lw=.7)
    axes_row[1].set_ylabel("Residual (%)")
    axes_row[1].set_title(f"RMS {row['rms_percent_continuum']:.2f}% | all selected pixels")
    for ax in axes_row:
        ax.set_xlabel("Wavelength (micron; vacuum assumed)")
axes[0, 0].legend(fontsize=8)
fig.suptitle("WASP-77A / IGRINS 2020-12-14, exposure 0: telluric feasibility", fontsize=14)
fig.tight_layout()
fig.savefig(OUT / "baseline_comparison.png", dpi=160)
plt.close(fig)

report = """IGRINS / WASP-77A b: initial ExoJAX telluric fitting experiment

Scope
-----
This is an initial feasibility fit of the telluric component of observed
stellar spectra. It is NOT a planetary atmosphere retrieval and is not a
fit of all 44 orders and 79 exposures. No planet signal extraction or SVD
was performed. Existing local observations were not modified.

Data
----
Source: https://zenodo.org/records/10382053
Paper reduction context: https://arxiv.org/html/2312.13069v1
cube14v3.pic: wavelengths (44,1848), flux (44,79,1848).
Cube and all CSV hashes match the Zenodo record (see data_inventory.json).
Only the 2020-12-14 cube is present; the other nights have metadata only.
Micron wavelengths and vacuum convention are assumed. A free velocity offset
and linear stretch are fitted; no barycentric correction is applied.
The paper describes frame alignment/interpolation; errors may be correlated.
No pixel error array, stellar template, or confirmed air/vacuum convention is
included. The count-like spectra retain stellar and telluric absorption.

Model
-----
HITRAN2024 live line lists, all available isotopes, 4050-6900 cm^-1.
https://hitran.org/docs/hitran-papers/
Raw files and SHA256, actual download dates, URLs: cache/*.par and cache/*.json.
HAPI 1.2.2.3 TIPS-2021 partition sums interpolated at 1 K (150-400 K).
Existing OpaDirect Voigt opacity (thermal, air-pressure, natural widths),
exp(-sum(N_species*sigma_species)), Gaussian IP, and ExoJAX sampling.
Column strengths include natural isotopic abundance; isotope masses are used.
One effective slab with fitted T and P; molecular self-pressure is set to zero.
Independent columns for relevant H2O/CO2/CH4/N2O/CO; cubic log-continuum.
Temperature 180-330 K, pressure 0.03-1.1 bar, IP sigma 1-8 km/s.
IP mixture experiment: positive mixture of same-center Gaussians with widths
sigma and 2*sigma; wing fraction 0-0.5. This does not test arbitrary/asymmetric IPs.
Pressure shift experiment uses existing low-level LPF with line-center shift
delta_air * P / 1.01325. No shift-temperature law or self shift is assumed.
Continuum is applied after convolution, a smooth-continuum approximation.

Optimization and validation
---------------------------
Float64 JAX CPU; exact autodiff Jacobian; scipy least_squares TRF (bounded
Gauss-Newton trust-region). LM uses logistic parameter bounds and is tested
as a warm-start refinement of the order13 baseline, not a global comparison.
The initialization scans RV -90..90 km/s, followed by joint optimization.
Unweighted linear flux residuals, no residual clipping. Pixels600:1500 by
default. All variants of a given interval/exposure use the same observations.
The fixed-scale RMS uses the same observed 95th-percentile flux normalization
for comparisons. Continuum RMS is easier to interpret but its denominator is
fitted. Bounds, convergence, nfev and directional derivative errors are saved.
No reduced chi-square or formal parameter uncertainties are claimed.
The metadata median SNR of exposure0 is about195 (inverse ~0.51%), not a
per-pixel noise estimate. Correlated residuals remain above this rough scale.
o13_fine_grid halves velocity grid spacing to check numerical convergence.
o34_exploratory and o40_exploratory refit short windows chosen AFTER looking
at these data. They are descriptive, not independent validation or confirmed
stellar-free windows; their RMS must not replace the full-window score.

Reproduction (from repository root)
-----------------------------------
python IGRINS/prepare_hitran.py
python IGRINS/inspect_data.py
python IGRINS/fit_telluric.py --order 13
python IGRINS/fit_telluric.py --order 34
python IGRINS/fit_telluric.py --order 40
python IGRINS/fit_telluric.py --order 5
python IGRINS/fit_telluric.py --order 13 --exposure 39
python IGRINS/fit_telluric.py --order 13 --exposure 78
For an order K, pressure shift: add --shift --initial
IGRINS/results/oKK_e00_baseline_gaussian_trf.json
For IP wings: add --shift --ip mixture --initial
IGRINS/results/oKK_e00_shift_gaussian_trf.json
LM: --order 13 --optimizer lm --initial IGRINS/results/o13_e00_baseline_gaussian_trf.json
Fine grid: --order 13 --dv 0.375 --initial IGRINS/results/o13_e00_baseline_gaussian_trf.json --tag o13_fine_grid
Exploratory: --order 34 --pixels 606 752 --tag o34_exploratory
Exploratory: --order 40 --pixels 1082 1267 --tag o40_exploratory
python IGRINS/summarize_fits.py
Each fit produces JSON parameters/diagnostics, NPZ spectra, and PNG residuals.

Repository changes
------------------
/IGRINS/ excluded by .gitignore; all fitting code, data and results stay there.
ExoJAX convolve_same now specifies inverse FFT length and slices the exact
input length. This fixes odd-sized spectra and a one-element kernel; even and
long kernels also match scipy.signal.convolve(...,mode='same').
Regression plus existing convolution/IP/rotation tests: 42 passed.
JAX_PLATFORMS=cpu python -m pytest tests/unittests/signal/test_convolve.py tests/unittests/signal/test_ola.py -q
JAX_PLATFORMS=cpu python -m pytest tests/unittests/postproc/test_response.py tests/unittests/postproc/test_spin_rotation.py tests/unittests/postproc/test_spin_rotation_trans.py -q

Results: RMS percent of fitted continuum (see summary.csv for fixed-scale RMS)
-----------------------------------------------------------------------------
"""
for row in rows:
    report += (f"{row['run']:43s} RMS={row['rms_percent_continuum']:8.4f}% "
               f"MAD={row['mad_percent_continuum']:7.4f}% "
               f"success={row['success']} bounds={row['bounds_hit'] or 'none'}\n")
report += """
Interpretation
--------------
Strong atmospheric line locations and depths are reproduced. Large unexplained
absorption troughs and continuum bias limit broad H-band fits; a stellar model
is needed before attributing those residuals to telluric physics. Pressure
shift and this simple symmetric IP extension have only a small effect in
the tested intervals. This is not evidence that those effects are unimportant
at higher precision, or that the line database is error-free. No controlled
HITRAN2020-versus-2024 comparison was made.
Next: model or independently mask stellar lines, jointly fit exposures with
an atmospheric vertical profile, then test wavelength-dependent/asymmetric
IP and pressure shifts on the resulting residuals. Slab T/P/columns should
not be interpreted as a measured terrestrial or planetary atmospheric profile.

Specific checks on order13, exposure0
-----------------------------------
TRF baseline RMS 2.363500%; warm-start LM 2.363500% (same local minimum).
Pressure-shift RMS 2.363330%; fine-grid baseline RMS 2.363672%.
The tiny pressure-shift RMS improvement is comparable to the grid dependence.
Pressure shifts do change fitted RV by about 0.106 km/s; absence of a large
RMS improvement does not mean that pressure shifts have no physical effect.
Coarse/fine fitted-model difference: RMS 0.01925%, maximum 0.19267% continuum.
Shift plus symmetric IP mixture RMS 2.354425%, wing fraction about0.157;
this small descriptive gain is not a significance claim for a more flexible IP.
Order13 exposures0/39/78 baseline RMS: 2.3635%, 2.3308%, 2.2914%.
"""
(OUT / "REPORT.txt").write_text(report)
print(f"Summarized {len(rows)} completed runs in {OUT}")
