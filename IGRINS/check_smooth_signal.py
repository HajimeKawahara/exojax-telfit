"""Measure artificial narrow-line response to fixed smooth nuisance operators.

This checks only the added smoother. It is not a full planet retrieval or an
injection before telluric fitting and instrumental convolution.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
LIGHT_SPEED_KMS = 299792.458
WIDTHS_KMS = [6.0, 10.0, 20.0]
KP_KMS = [0.0, 100.0, 200.0]
DEPTH = 1e-3


def summarize(values):
    values = np.asarray(values, dtype=float)
    return dict(count=int(values.size), minimum=float(values.min()),
                p05=float(np.percentile(values, 5)), median=float(np.median(values)),
                p95=float(np.percentile(values, 95)), maximum=float(values.max()))


def isolated_response(velocity, operator, score_mask, width_kms):
    """Test signed Gaussian amplitudes and local line contrast over many centers."""
    sigma = width_kms / np.sqrt(8 * np.log(2))
    available = np.flatnonzero(score_mask)
    # Sample by wavelength position, not by response or residual size.
    centers = available[::6]
    offset = velocity[:, None] - velocity[centers][None, :]
    signal = -DEPTH * np.exp(-0.5 * (offset / sigma) ** 2)
    after = signal - operator @ signal
    score = np.broadcast_to(score_mask[:, None], signal.shape)
    power = np.sum(np.where(score, signal**2, 0), axis=0)
    raw_gain = np.sum(np.where(score, signal * after, 0), axis=0) / power
    contrast_gain = []
    center_retention = []
    for j, center in enumerate(centers):
        window = score_mask & (np.abs(offset[:, j]) <= 35)
        s = signal[window, j]
        q = after[window, j]
        s -= s.mean()
        q -= q.mean()
        contrast_gain.append(float(s @ q / (s @ s)))
        # A direct core depth can include removed broad wings; report separately.
        center_retention.append(float(after[center, j] / signal[center, j]))
    edge_distance = np.minimum(np.abs(velocity[centers]-velocity.min()),
                               np.abs(velocity[centers]-velocity.max()))
    boundary_groups = {}
    for name, keep in [("within_35_kms_of_order_edge", edge_distance <= 35),
                       ("order_interior", edge_distance > 35)]:
        if keep.any():
            boundary_groups[name] = dict(raw_gain=summarize(raw_gain[keep]),
                                        contrast_gain=summarize(np.asarray(contrast_gain)[keep]))
    return dict(centers=centers.tolist(), raw_gain=raw_gain.tolist(),
                contrast_gain=contrast_gain, center_retention=center_retention,
                boundary_groups=boundary_groups,
                summary=dict(raw_gain=summarize(raw_gain),
                             contrast_gain=summarize(contrast_gain),
                             center_retention=summarize(center_retention)))


def forest_signal(velocity, phase, component, width_kms, kp_kms, seed):
    """Create a hypothetical moving absorption forest at observed resolution."""
    rng = np.random.default_rng(seed)
    centers = np.linspace(velocity.min() + 100, velocity.max() - 100, 24)
    centers += rng.uniform(-20, 20, centers.size)
    strengths = rng.uniform(0.3, 1.0, centers.size)
    sigma = width_kms / np.sqrt(8 * np.log(2))
    shifts = kp_kms * np.sin(2 * np.pi * phase)
    offset = velocity[None, :, None] - centers[None, None, :] - shifts[:, None, None]
    forest = np.sum(strengths[None, None] * np.exp(-0.5 * (offset / sigma) ** 2), axis=-1)
    # Multiplying the saved component approximates attenuation at detector
    # resolution. It is not a substitute for injection before IP convolution.
    return -DEPTH * component * forest


def forest_response(signal, operator, score_mask):
    after = signal - signal @ operator.T
    s = signal[:, score_mask]
    q = after[:, score_mask]
    raw_gain = float(np.sum(s * q) / np.sum(s * s))
    # Remove the per-order/per-exposure constant to separate broad/DC loss.
    sc = s - s.mean(axis=1, keepdims=True)
    qc = q - q.mean(axis=1, keepdims=True)
    return dict(raw_numerator=float(np.sum(s * q)), raw_denominator=float(np.sum(s * s)),
                contrast_numerator=float(np.sum(sc * qc)),
                contrast_denominator=float(np.sum(sc * sc)), raw_gain=raw_gain,
                mean_removed_gain=float(np.sum(sc * qc) / np.sum(sc * sc)),
                norm_ratio=float(np.linalg.norm(q) / np.linalg.norm(s)),
                correlation=float(np.sum(s*q) / np.linalg.norm(s) / np.linalg.norm(q)))


def main():
    from smooth_residuals import CONFIGS, smoother_matrix

    source_path = OUT / "motion_clean_control.npz"
    with np.load(source_path) as saved:
        data = {key: saved[key] for key in saved.files}
    info_path = ROOT / "data" / "20201214_info.csv"
    info = np.genfromtxt(info_path, delimiter=",", names=True)
    phases = info["Phase"][data["exposures"][:-1]]
    included = data["included_mask"]
    calibration = data["calibration_mask"][:, -1]
    scoring = data["score_mask"][:, -1]
    velocities = LIGHT_SPEED_KMS * np.log(data["wavelength_um"] /
                                         np.median(data["wavelength_um"], axis=1)[:, None])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  source="motion_clean_control", injected_exposures=data["exposures"][:-1].tolist(),
                  depth=DEPTH, fwhm_kms=WIDTHS_KMS, hypothetical_kp_kms=KP_KMS,
                  additional_linearity_factors=[-1.0, .1, 3.0],
                  definitions={
                      "raw_gain": "dot(s, (I-H)s) / dot(s,s), signed, on scoring pixels",
                      "isolated_contrast_gain": "Same projection after subtracting means in a +/-35 km/s window",
                      "forest_mean_removed_gain": "Same projection after per-exposure/order mean subtraction",
                      "blocked": "Fit smoother on calibration pixels, score on disjoint scoring blocks",
                      "all_included": "Fit and score smoother on included pixels; production self-subtraction test",
                  },
                  limitations=[
                      "Fixed smoother only; the earlier telluric and continuum fits are not rerun after injection.",
                      "Observed-resolution artificial Gaussian lines, not a physical planet spectrum.",
                      "Forest attenuation by the saved component is approximate; injection is after IP convolution.",
                      "Kp=0/100/200 km/s are hypothetical controls, not measured orbital parameters.",
                      "Retention is conditional on these widths, masks, line patterns and fixed hyperparameters.",
                      "No RMS improvement or line-retention number is a planet detection significance.",
                  ], settings=[])
    rows = []
    largest_linearity_error = 0.0
    for config in CONFIGS:
        settings = dict(name=config["name"], config=config, protocols={})
        for protocol, fit_masks, score_masks in [("blocked", calibration, scoring),
                                                  ("all_included", included, included)]:
            hs = [smoother_matrix(wave, fit_mask, config)
                  for wave, fit_mask in zip(data["wavelength_um"], fit_masks)]
            isolated_by_width = {}
            for width in WIDTHS_KMS:
                by_order = {}
                for b, order in enumerate(data["orders"]):
                    by_order[str(order)] = isolated_response(velocities[b], hs[b], score_masks[b], width)
                pooled = {}
                for metric in ["raw_gain", "contrast_gain", "center_retention"]:
                    pooled[metric] = summarize([x for values in by_order.values() for x in values[metric]])
                isolated_by_width[str(width)] = dict(pooled=pooled, per_order=by_order)
                rows.append(dict(name=config["name"], protocol=protocol, signal="isolated",
                                 fwhm_kms=width, kp_kms="", raw_gain=pooled["raw_gain"]["median"],
                                 contrast_gain=pooled["contrast_gain"]["median"],
                                 contrast_p05=pooled["contrast_gain"]["p05"]))
            forests = []
            for width in WIDTHS_KMS:
                for kp in KP_KMS:
                    parts = []
                    for b, order in enumerate(data["orders"]):
                        signal = forest_signal(velocities[b], phases, data["component"][b, :-1],
                                               width, kp, seed=20261111 + int(order))
                        parts.append(forest_response(signal, hs[b], score_masks[b]))
                        # Check injected-minus-uninjected arithmetic on a real residual.
                        baseline = data["residual_continuum"][b, :-1]
                        delta = ((baseline+signal) - (baseline+signal)@hs[b].T
                                 - (baseline - baseline@hs[b].T))
                        exact = signal-signal@hs[b].T
                        error = float(np.max(np.abs(delta-exact)))
                        largest_linearity_error = max(largest_linearity_error, error)
                        np.testing.assert_allclose(delta, exact, rtol=1e-9, atol=1e-13)
                        if width == 10 and kp == 100:
                            for factor in [-1.0, .1, 3.0]:
                                scaled = ((baseline+factor*signal)
                                          - (baseline+factor*signal)@hs[b].T
                                          - (baseline - baseline@hs[b].T))
                                largest_linearity_error = max(largest_linearity_error,
                                    float(np.max(np.abs(scaled-factor*exact))))
                                np.testing.assert_allclose(scaled, factor*exact,
                                                           rtol=1e-9, atol=1e-13)
                    raw = sum(p["raw_numerator"] for p in parts) / sum(p["raw_denominator"] for p in parts)
                    contrast = sum(p["contrast_numerator"] for p in parts) / sum(p["contrast_denominator"] for p in parts)
                    forests.append(dict(fwhm_kms=width, kp_kms=kp, raw_gain=raw,
                                        mean_removed_gain=contrast, per_order=parts))
                    rows.append(dict(name=config["name"], protocol=protocol, signal="forest",
                                     fwhm_kms=width, kp_kms=kp, raw_gain=raw,
                                     contrast_gain=contrast, contrast_p05=""))
            settings["protocols"][protocol] = dict(isolated=isolated_by_width, forests=forests)
        gate = settings["protocols"]["all_included"]["isolated"]["10.0"]["pooled"]["raw_gain"]
        settings["qualification"] = dict(fwhm_kms=10.0, metric="raw_signed_gain",
                                          median=gate["median"], p05=gate["p05"],
                                          median_minimum=.95, p05_minimum=.90,
                                          passed=gate["median"] >= .95 and gate["p05"] >= .90)
        result["settings"].append(settings)
        print(config["name"], "done", flush=True)
    result["injected_difference_max_error"] = largest_linearity_error
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in [source_path, info_path, Path(__file__), ROOT/"smooth_residuals.py"]}
    (OUT / "smooth_signal_retention.json").write_text(json.dumps(result, indent=2)+"\n")
    with (OUT / "smooth_signal_retention.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), constrained_layout=True)
    names = [setting["name"] for setting in result["settings"]]
    for column, protocol in enumerate(["blocked", "all_included"]):
        for row, metric in enumerate(["raw_gain", "contrast_gain"]):
            ax = axes[row, column]
            for color, width in enumerate(WIDTHS_KMS):
                summaries = [setting["protocols"][protocol]["isolated"][str(width)]["pooled"][metric]
                             for setting in result["settings"]]
                medians = np.array([s["median"] for s in summaries])
                p05 = np.array([s["p05"] for s in summaries])
                ax.plot(names, 100*medians, ".-", color=f"C{color}", label=f"FWHM {width:g} km/s: median")
                ax.plot(names, 100*p05, ":", color=f"C{color}", alpha=.7)
            ax.axhline(95, color="0.5", ls="--", lw=.8)
            ylabel = "Raw matched-amplitude retention (%)" if row == 0 else "Local line-contrast retention (%)"
            ax.set(title=protocol.replace("_", " "), ylabel=ylabel)
            ax.tick_params(axis="x", rotation=70, labelsize=7)
            ax.legend(fontsize=8)
    fig.suptitle("Artificial-line response to the added fixed smoother; dotted curves: 5th percentile")
    fig.savefig(OUT / "smooth_signal_retention.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
