"""Locate residual features in observer and stellar frames across three nights.

This local centroid diagnostic uses empirical telluric anchors. It does not fit
a new stellar template, change the fixed mask, or retrieve other-night TP.
"""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import find_peaks

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"
C_KMS = 299792.458
NIGHTS = ["06", "14", "21"]
FEATURES = [(40, 1.501046034, -1, "Unknown absorption"),
            (40, 1.494855416, -1, "Unknown absorption on a telluric wing"),
            (40, 1.501104171, 1, "Positive residual"),
            (34, 1.570498667, 1, "Positive residual on a telluric wing"),
            (34, 1.573526470, -1, "Previously excluded unknown absorption"),
            (34, 1.575462050, -1, "Strong stellar-line control")]


def fit_local(wave, flux, center, sign=-1, window=12., quadratic=False):
    velocity = C_KMS*(wave/center-1)
    selected = abs(velocity) < window
    x, raw = velocity[selected], flux[selected]
    scale = np.percentile(raw, 95)
    y = raw/scale
    if len(y) < 9 or not np.all(np.isfinite(y)):
        raise ValueError("Insufficient finite local samples")
    n_poly = 3 if quadratic else 2
    basis = np.polynomial.polynomial.polyvander(x/window, n_poly-1)
    def model(q):
        return basis@q[:n_poly]+sign*q[n_poly]*np.exp(-.5*((x-q[n_poly+1])/q[n_poly+2])**2)
    initial = np.r_[np.median(y), np.zeros(n_poly-1), .05, 0., 2.5]
    lower = np.r_[.2, np.full(n_poly-1, -1.), 0., -8., .6]
    upper = np.r_[1.8, np.full(n_poly-1, 1.), 1.5, 8., 9.]
    # Several center starts prevent a nearby atmospheric feature from winning
    # only because of the starting point; the local target remains predeclared.
    trials = []
    for offset in [-3., 0., 3.]:
        q0 = initial.copy(); q0[n_poly+1] = offset
        fit = least_squares(lambda q: model(q)-y, q0, bounds=(lower, upper),
                            max_nfev=300, x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-10)
        trials.append(fit)
    fit = min(trials, key=lambda value: value.cost)
    q = fit.x
    continuum = basis@q[:n_poly]
    resid = y-model(q)
    covariance = np.linalg.pinv(fit.jac.T@fit.jac)*np.sum(resid**2)/max(len(y)-len(q), 1)
    return dict(center_kms=float(q[n_poly+1]), sigma_kms=float(q[n_poly+2]),
                amplitude=float(q[n_poly]), center_formal_error_kms=float(np.sqrt(max(covariance[n_poly+1, n_poly+1], 0))),
                rms=float(np.sqrt(np.mean(resid**2))), success=bool(fit.success),
                near_bound=bool(np.any((q-lower < .001*(upper-lower)) | (upper-q < .001*(upper-lower)))),
                n_pixels=len(y), velocity=x, normalized_flux=y, parameters=q)


def serializable_fit(fit):
    return {key: value for key, value in fit.items() if key not in ["velocity", "normalized_flux", "parameters"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--night21-index-offset", type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    suffix = "_offset1" if args.night21_index_offset else ""
    with np.load(OUT/"residual_components.npz") as saved:
        components = {key: saved[key] for key in saved.files}
    data, inventory = {}, []
    for night in NIGHTS:
        path = ROOT/"data"/f"cube{night}v3.pic"
        with path.open("rb") as stream:
            wavelength, cube = pickle.load(stream)
        info = pd.read_csv(ROOT/"data"/f"202012{night}_info.csv")
        mismatch = cube.shape[1]-len(info)
        if mismatch != 0 and not (night == "21" and mismatch == 1):
            raise ValueError("Cube and metadata frame counts differ")
        nearest = int(np.argmin(abs(info["Air Mass"].to_numpy()-1.2)))
        metadata_indices = np.arange(max(0, nearest-2), min(len(info), nearest+3))
        indices = metadata_indices+(args.night21_index_offset if night == "21" else 0)
        if night == "14" and np.isin(indices, [10, 30, 50, 70]).any():
            raise ValueError("Do not select the reserved final frames for localization")
        spectra = cube[:, indices].copy()
        del cube
        medians = np.median(spectra/np.percentile(spectra, 95, axis=2)[:, :, None], axis=1)
        data[night] = dict(wave=wavelength, spectra=spectra, median=medians, indices=indices,
                           stellar_rv=float(np.median(info.iloc[metadata_indices]["RV [km/s]"])))
        inventory.append(dict(night=f"2020-12-{night}", wavelength_shape=list(wavelength.shape),
                              selected_shape=list(spectra.shape), selected_cube_indices=indices.tolist(),
                              selected_metadata_indices=metadata_indices.tolist(),
                              cube_minus_metadata_frame_count=mismatch,
                              index_mapping_assumption="Unresolved missing metadata row; test local offset "+str(args.night21_index_offset) if mismatch else "One-to-one matching counts",
                              airmass_range=[float(info.iloc[metadata_indices]["Air Mass"].min()), float(info.iloc[metadata_indices]["Air Mass"].max())],
                              stellar_rv_kms=data[night]["stellar_rv"],
                              cube_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))

    anchors, calibrations = {}, {}
    calibration_fig, calibration_axes = plt.subplots(3, 1, figsize=(11, 9), constrained_layout=True)
    for b, order in enumerate(components["orders"]):
        transmission = np.median(components["telluric_only"][b], axis=0)
        star_depth = np.max(1-components["stellar_only"][b], axis=0)
        peaks, _ = find_peaks(1-transmission, prominence=.08, distance=12)
        selected = [k for k in peaks if .1 < 1-transmission[k] < .8 and 8 < k < len(transmission)-8
                    and np.max(star_depth[k-4:k+5]) < .02]
        records = []
        for k in selected:
            center = float(components["wavelength_um"][b, k])
            fits = {}
            for night in NIGHTS:
                fit = fit_local(data[night]["wave"][order], data[night]["median"][order], center,
                                window=16., quadratic=False)
                fits[night] = serializable_fit(fit)
            # Saturated or boundary-limited anchor fits do not define a reliable
            # local wavelength reference. This affects only this diagnostic.
            usable = all(fit["success"] and not fit["near_bound"] and .04 < fit["amplitude"] < .9
                         for fit in fits.values())
            records.append(dict(wavelength_um=center, usable=usable, fits=fits))
        anchors[str(order)] = records
        usable = [record for record in records if record["usable"]]
        if len(usable) < 3:
            raise ValueError(f"Only {len(usable)} usable telluric anchors in order {order}")
        centers = np.array([record["wavelength_um"] for record in usable])
        midpoint, span = centers.mean(), np.ptp(centers)
        x = (centers-midpoint)/span
        design = np.column_stack([np.ones(len(x)), x])
        calibration = {}
        ax = calibration_axes[b]
        for night in NIGHTS:
            offsets = np.array([record["fits"][night]["center_kms"]-record["fits"]["14"]["center_kms"] for record in usable])
            fit = least_squares(lambda q: design@q-offsets, [np.median(offsets), 0.],
                                loss="soft_l1", f_scale=.15)
            residual = offsets-design@fit.x
            scatter = float(1.4826*np.median(abs(residual-np.median(residual))))
            calibration[night] = dict(offset_kms=float(fit.x[0]), slope_kms=float(fit.x[1]),
                                      wavelength_midpoint_um=float(midpoint), wavelength_span_um=float(span),
                                      anchor_scatter_kms=scatter, usable_anchors=len(usable))
            ax.plot(centers, offsets, "o", label=f"Dec {night}")
            ordered = np.argsort(centers)
            ax.plot(centers[ordered], (design@fit.x)[ordered], lw=1)
        calibrations[str(order)] = calibration
        ax.set(xlabel="Observed vacuum wavelength (micron, assumed)", ylabel="Offset from Dec14 (km/s)",
               title=f"Order {order}: empirical telluric anchors, {len(usable)} retained")
        ax.legend(fontsize=8)
    calibration_fig.savefig(OUT/f"crossnight_calibration{suffix}.png", dpi=150)
    plt.close(calibration_fig)

    def calibration_at(order, night, wave):
        q = calibrations[str(order)][night]
        return q["offset_kms"]+q["slope_kms"]*(wave-q["wavelength_midpoint_um"])/q["wavelength_span_um"]

    features, plots = [], []
    for order, center, sign, label in FEATURES:
        fits = {}
        for night in NIGHTS:
            q = data[night]
            # Offset the search center by the independently determined telluric
            # calibration. Stellar motion remains an unconstrained measurement.
            offset = calibration_at(order, night, center)
            shifted_center = center*(1+offset/C_KMS)
            fit = fit_local(q["wave"][order], q["median"][order], shifted_center,
                            sign=sign, window=12., quadratic=True)
            alternatives = [fit_local(q["wave"][order], q["median"][order], shifted_center,
                                      sign=sign, window=window, quadratic=True)["center_kms"]
                            for window in [10., 12., 16.]]
            jackknife = []
            scaled = q["spectra"][order]/np.percentile(q["spectra"][order], 95, axis=1)[:, None]
            for j in range(len(q["indices"])):
                median = np.median(np.delete(scaled, j, axis=0), axis=0)
                jackknife.append(fit_local(q["wave"][order], median, shifted_center,
                                          sign=sign, window=12., quadratic=True)["center_kms"])
            jackknife = np.asarray(jackknife)
            jackknife_error = float(np.sqrt((len(jackknife)-1)/len(jackknife)*np.sum((jackknife-jackknife.mean())**2)))
            fits[night] = dict(**serializable_fit(fit), calibration_kms=float(offset),
                               window_centers_kms=alternatives,
                               window_scatter_kms=float(np.std(alternatives)),
                               jackknife_center_error_kms=jackknife_error)
            plots.append((order, center, label, night, fit))
        reference_center = fits["14"]["center_kms"]
        shifts = {}
        for night in ["06", "21"]:
            measured = fits[night]["center_kms"]-reference_center
            predicted = data[night]["stellar_rv"]-data["14"]["stellar_rv"]
            uncertainty = np.sqrt(fits[night]["jackknife_center_error_kms"]**2
                                  + fits["14"]["jackknife_center_error_kms"]**2
                                  + fits[night]["window_scatter_kms"]**2
                                  + fits["14"]["window_scatter_kms"]**2
                                  + calibrations[str(order)][night]["anchor_scatter_kms"]**2)
            shifts[night] = dict(measured_after_telluric_alignment_kms=float(measured),
                                 stellar_prediction_kms=float(predicted),
                                 combined_diagnostic_spread_kms=float(uncertainty),
                                 error_from_stationary_kms=float(measured),
                                 error_from_stellar_motion_kms=float(measured-predicted))
        features.append(dict(order=order, reference_wavelength_um=center, description=label,
                             fits=fits, relative_shifts=shifts,
                             caveat="Gaussian centroid on a local quadratic background; blends and fit-window dependence can dominate"))

    for frame in ["observer", "stellar"]:
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for ax, feature in zip(axes.flat, features):
            for order, center, label, night, fit in plots:
                if order != feature["order"] or center != feature["reference_wavelength_um"]:
                    continue
                x = fit["velocity"].copy()
                if frame == "stellar":
                    x -= data[night]["stellar_rv"]-data["14"]["stellar_rv"]
                ax.plot(x, fit["normalized_flux"], "o-", ms=3, lw=.8, label=f"Dec {night}")
            ax.axvline(0, color=".6", lw=.7)
            unstable = any(fit["near_bound"] or not fit["success"] for fit in feature["fits"].values())
            ax.set(xlabel=f"Velocity in {frame} comparison frame (km/s)", ylabel="Local p95-normalized flux",
                   title=f"{feature['reference_wavelength_um']:.6f} micron\n{feature['description']}"+("\nCentroid fit unstable" if unstable else ""))
            ax.legend(fontsize=8)
        fig.suptitle("Cross-night localization near airmass1.2; empirical telluric alignment")
        fig.savefig(OUT/f"crossnight_{frame}{suffix}.png", dpi=150)
        plt.close(fig)
    report = dict(inventory=inventory, calibration_anchors=anchors, calibrations=calibrations,
                  features=features,
                  method=["Select five consecutive frames nearest airmass1.2 from each night using metadata only",
                          "Normalize each frame by its order p95 and median-combine the selected frames",
                          "Preselect telluric anchors from the existing Dec14 model:10-80% depth, prominence8%, weak modeled stellar absorption",
                          "Fit local Gaussian anchor centroids; discard saturated or bound-limited anchor fits only for this wavelength diagnostic",
                          "Robust affine wavelength offsets from multiple anchors align each order to Dec14",
                          "Compare local feature centroids with observer stationarity and the CSV relative stellar RV",
                          "Repeat fit windows and leave-one-frame-out medians to expose centroid instability"],
                  limits=["Plots show observed local p95-normalized flux, without division by the fitted background",
                          "Bound-limited feature fits are invalid for motion attribution; they remain saved only as failure diagnostics",
                          "Dec21 cube has51 frames but CSV has50; local row mapping is conditional on the stated offset hypothesis",
                          "No global spectrum fit or TP retrieval for the other nights",
                          "No main fitting mask or stellar template was changed",
                          "The combined diagnostic spread is not a calibrated statistical confidence interval",
                          "Telluric blends and atmospheric profile differences may bias local Gaussian centroids",
                          "No new frame from the four reserved Dec14 exposures is used for localization"],
                  source="https://zenodo.org/records/10382053",
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/f"crossnight_localization{suffix}.json").write_text(json.dumps(report, indent=2)+"\n")
    for feature in features:
        print(feature["reference_wavelength_um"], feature["relative_shifts"], flush=True)


if __name__ == "__main__":
    main()
