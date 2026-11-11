"""Check discovery residuals against observing cadence and OH wavelengths.

The reduced cube has AB-pair frames, not individual A/B nods. Frame parity is
therefore only an empirical cadence diagnostic, never a nod identification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
C_KMS = 299792.458


def main():
    with np.load(OUT/"transfer_diagnostics.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    exposures = data["exposures"]
    assert not np.isin(exposures, [10, 30, 50, 70]).any()
    info = pd.read_csv(ROOT/"data/20201214_info.csv").iloc[exposures]
    parameters = json.loads((OUT/"focus_ip.json").read_text())["parameters"]
    oh_path = ROOT/"cache/oh_rousselot_v2.dat"
    oh = np.loadtxt(oh_path)
    time = info["Time (BJD)"].to_numpy()
    elapsed = 24*(time-time[0])
    snr = info["Med. SNR"].to_numpy()
    covariates = dict(time_hours=elapsed, airmass=data["airmass"], slant_water=data["slant_water"],
                      snr=snr, inverse_snr_squared=1/snr**2)
    targets = [(40, 1.50110417145), (34, 1.57049866677), (13, 2.07682583662),
               (40, 1.50104603413)]
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for ax, (order, target) in zip(axes.flat, targets):
        b = int(np.flatnonzero(data["orders"] == order)[0])
        wave = data["wavelength_um"][b]
        k = int(np.argmin(abs(wave-target)))
        y = data["fixed_residual"][b, :, k]
        component = data["fixed_ip_model"][b, :, k]/data["fixed_ip_continuum"][b, :, k]
        observed = data["observed"][b, :, k]/data["fixed_ip_continuum"][b, :, k]
        x = np.linspace(-1, 1, len(wave))[k]
        atmospheric_wave = wave[k]/((1+parameters[f"stretch_o{order}_kms"]*x/C_KMS)
                                    *(1+parameters[f"calibration_rv_o{order}_kms"]/C_KMS))
        velocity = C_KMS*(oh[:, 0]/(1e4*atmospheric_wave)-1)
        nearest = np.argsort(abs(velocity))[:4]
        matches = [dict(vacuum_wavelength_A=float(oh[j, 0]), relative_intensity=float(oh[j, 1]),
                        velocity_from_feature_kms=float(velocity[j])) for j in nearest]
        correlations = {name: dict(pearson=float(pearsonr(value, y).statistic),
                                   spearman=float(spearmanr(value, y).statistic))
                        for name, value in covariates.items()}
        t = 2*(elapsed-elapsed.min())/np.ptp(elapsed)-1
        smooth = np.polynomial.chebyshev.chebvander(t, 3)
        parity = (-1.)**exposures
        design = np.column_stack([smooth, parity])
        q, *_ = np.linalg.lstsq(design, y, rcond=None)
        smooth_q, *_ = np.linalg.lstsq(smooth, y, rcond=None)
        baseline_sse = float(np.sum((y-smooth@smooth_q)**2))
        parity_sse = float(np.sum((y-design@q)**2))
        rows.append(dict(order=order, wavelength_um=float(wave[k]), original_pixel=int(1499-k),
                         mean_residual_percent=float(100*y.mean()), scatter_percent=float(100*y.std()),
                         observed_flux_range=[float(observed.min()), float(observed.max())],
                         component_flux_range=[float(component.min()), float(component.max())],
                         correlations=correlations,
                         parity_even_minus_odd_after_cubic_time_percent=float(200*q[-1]),
                         parity_fraction_of_detrended_sse_removed=float(1-parity_sse/baseline_sse),
                         nearest_oh_lines=matches, oh_within_10kms=int(np.sum(abs(velocity)<10)),
                         residual_percent=(100*y).tolist()))
        ax.scatter(elapsed, 100*y, c=data["airmass"], cmap="viridis", s=18,
                   label="Single-pixel residual")
        ax.plot(elapsed, 100*(smooth@smooth_q), color="tab:red", lw=1, label="Cubic time trend")
        ax.set(xlabel="Elapsed time (hours)", ylabel="Residual (%)",
               title=f"Order {order} | {wave[k]:.6f} micron")
        ax.axhline(0, color=".6", lw=.6)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Additive-residual checks; discovery frames only (color = airmass)")
    fig.savefig(OUT/"background_diagnostics.png", dpi=150)
    plt.close(fig)
    result = dict(reserved_exposures_not_selected=[10, 30, 50, 70], exposures=exposures.tolist(),
                  features=rows,
                  observing_context={
                      "paper": "https://arxiv.org/html/2312.13069v1#S2",
                      "facts": ["70 s exposures in an AB-BA nodding pattern; an AB pair is one 140 s frame",
                                "Facility IGRINS PLP extracted the spectra",
                                "Frames were aligned to the final frame on a common wavelength grid",
                                "Counts were normalized and low-throughput edges discarded"],
                      "limits": "The public info CSV does not identify each nod or provide extracted sky spectra, pixel masks, or uncertainties."},
                  oh_reference={"url": "https://www.eso.org/sci/facilities/paranal/decommissioned/isaac/tools/oh/list_v2.0.dat",
                                "wavelengths": "vacuum Angstrom", "intensities": "approximate relative photon units",
                                "line_count": len(oh), "sha256": hashlib.sha256(oh_path.read_bytes()).hexdigest()},
                  interpretation=[
                      "No listed OH line lies within 10 km/s of these four target residuals",
                      "A positive residual exceeding the local modeled continuum cannot be fixed by adding absorbing lines alone",
                      "Absence of an OH match does not exclude other sky emission, background subtraction, or continuum artifacts",
                      "Empirical frame parity is not a physical A/B assignment because each cube frame is already an AB pair",
                      "Airmass, time, SNR and fitted water column covary; correlations do not isolate a physical cause",
                      "The 1.570499 micron feature is close to the fitted interval boundary (pixel600), not the detector edge"],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), OUT/"transfer_diagnostics.npz", ROOT/"data/20201214_info.csv"]})
    (OUT/"background_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps([{key: value for key, value in row.items() if key != "residual_percent"} for row in rows], indent=2))


if __name__ == "__main__":
    main()
