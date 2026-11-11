"""Compare smooth additive residual corrections with frozen physical spectra.

Calibration-only predictions and all-included high-pass filtering are reported
separately. The original fitted continuum remains the unit of every residual.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
from scipy.linalg import cho_factor, cho_solve

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
EXPOSURES = [10, 30, 50, 70]
CONFIGS = [dict(name="none", kind="none")]
CONFIGS += [dict(name=f"poly{degree}", kind="polynomial", degree=degree)
            for degree in [1, 3, 5, 7]]
CONFIGS += [dict(name=f"gp_l{length}_a{amplitude:.2f}", kind="gp",
                 length_kms=float(length), amplitude=amplitude, noise=.0045)
            for length in [30, 60, 120, 240, 480] for amplitude in [.01, .03]]


def smoother_matrix(wavelength_um, fit_mask, config):
    """Return H such that H @ residual predicts a smooth full-grid correction."""
    wave = np.asarray(wavelength_um)
    fit = np.asarray(fit_mask, dtype=bool)
    velocity = C_KMS * np.log(wave / np.exp(np.mean(np.log(wave))))
    result = np.zeros((wave.size, wave.size))
    if config["kind"] == "none":
        return result
    if config["kind"] == "polynomial":
        x = 2 * (velocity - velocity.min()) / np.ptp(velocity) - 1
        basis = np.polynomial.chebyshev.chebvander(x, config["degree"])
        result[:, fit] = basis @ np.linalg.pinv(basis[fit])
    elif config["kind"] == "gp":
        distance = (velocity[:, None] - velocity[None, fit]) / config["length_kms"]
        kernel = config["amplitude"]**2 * np.exp(-.5 * distance**2)
        covariance = kernel[fit] + config["noise"]**2 * np.eye(fit.sum())
        result[:, fit] = cho_solve(cho_factor(covariance, lower=True), kernel.T).T
    else:
        raise ValueError(f"Unknown smooth model: {config}")
    return result


def load_fit(tag):
    with np.load(OUT / f"{tag}.npz") as saved:
        return {key: saved[key] for key in saved.files}


def metrics(residual, continuum, mask, orders):
    keep = np.broadcast_to(mask[:, None], residual.shape)
    fixed = residual * continuum
    return dict(n_samples=int(keep.sum()),
                continuum_rms_percent=float(100*np.sqrt(np.mean(residual[keep]**2))),
                fixed_rms_percent=float(100*np.sqrt(np.mean(fixed[keep]**2))),
                continuum_bias_percent=float(100*np.mean(residual[keep])),
                per_order={str(order): dict(
                    continuum_rms_percent=float(100*np.sqrt(np.mean(residual[b][keep[b]]**2))),
                    fixed_rms_percent=float(100*np.sqrt(np.mean(fixed[b][keep[b]]**2))))
                    for b, order in enumerate(orders)})


def main():
    source = load_fit("motion_clean_control")
    wave = source["wavelength_um"]
    orders = source["orders"]
    included = source["included_mask"]
    calibration = source["calibration_mask"][:, -1]
    score = source["score_mask"][:, -1]
    dev = source["residual_continuum"][:, -1:]
    dev_continuum = source["continuum"][:, -1:]
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), configs=CONFIGS,
                definition="Add C*g to the physical model; g=H*((observed-model)/C). Original C is fixed.",
                kernel="Zero-mean squared-exponential GP; amplitude and noise in original continuum units",
                velocity="299792.458 * ln(wavelength / geometric-mean wavelength), km/s",
                selection="Smallest development e20 calibration-only scoring RMS among injection-qualified candidates",
                fidelity_gate="All-included isolated FWHM10 km/s raw signed gain: median>=0.95 and fifth percentile>=0.90",
                modes=dict(calibration_only="Fit on calibration blocks; predict disjoint score blocks",
                           all_included="Fit on all included pixels; score pixels participate, so descriptive filtered RMS"),
                limitations=["Fixed physical parameters; no joint telluric refit in this experiment.",
                             "All-included filtering requires applying the same operator to planetary templates.",
                             "The GP noise parameter is a fixed regularization scale, not a measured pixel-noise model.",
                             "No time-dependent common residual template is subtracted."])
    (OUT / "smooth_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    development, operators, effective_df = {}, {}, {}
    for config in CONFIGS:
        name = config["name"]
        development[name], operators[name], effective_df[name] = {}, {}, {}
        for mode, mask in [("calibration_only", calibration), ("all_included", included)]:
            matrices = np.array([smoother_matrix(w, m, config) for w, m in zip(wave, mask)])
            operators[name][mode] = matrices
            effective_df[name][mode] = np.trace(matrices, axis1=1, axis2=2).tolist()
            correction = np.einsum("bij,bej->bei", matrices, dev)
            development[name][mode] = metrics(dev-correction, dev_continuum, score, orders)
    ranked = sorted(CONFIGS, key=lambda config: development[config["name"]]["calibration_only"]["continuum_rms_percent"])
    selection = dict(development=development,
                     development_rank=[config["name"] for config in ranked],
                     unconstrained_chosen_model=ranked[0]["name"],
                     selection_status="Awaiting synthetic injection fidelity gate; final scores are not selection input")
    # Save the development ranking before opening any of the evaluation fits.
    (OUT / "smooth_selection.json").write_text(json.dumps(selection, indent=2)+"\n")

    residuals, continua, inputs = [], [], [OUT / "motion_clean_control.npz"]
    for exposure in EXPOSURES:
        tag = f"nonstellar_control_e{exposure}"
        data = load_fit(tag)
        np.testing.assert_array_equal(data["included_mask"], included)
        np.testing.assert_array_equal(data["score_mask"][:, -1], score)
        np.testing.assert_array_equal(data["calibration_mask"][:, -1], calibration)
        np.testing.assert_allclose(data["model"][:, :-1], source["model"][:, :-1], atol=1e-12, rtol=0)
        residuals.append(data["residual_continuum"][:, -1])
        continua.append(data["continuum"][:, -1])
        inputs.append(OUT / f"{tag}.npz")
    residual = np.stack(residuals, axis=1)
    continuum = np.stack(continua, axis=1)
    rows, final, saved = [], {}, {}
    for config in CONFIGS:
        name = config["name"]
        final[name] = {}
        for mode in ["calibration_only", "all_included"]:
            matrices = operators[name][mode]
            correction = np.einsum("bij,bej->bei", matrices, residual)
            remaining = residual-correction
            final[name][mode] = metrics(remaining, continuum, score, orders)
            baseline = np.sum(residual[np.broadcast_to(score[:, None], residual.shape)]**2)
            final[name][mode]["squared_error_reduction_fraction"] = float(
                1-np.sum(remaining[np.broadcast_to(score[:, None], residual.shape)]**2)/baseline)
            final[name][mode]["all_included_metrics"] = metrics(remaining, continuum, included, orders)
            saved[f"correction_{name}_{mode}"] = correction
            rows.append(dict(model=name, mode=mode,
                             development_rms_percent=development[name][mode]["continuum_rms_percent"],
                             scoring_rms_percent=final[name][mode]["continuum_rms_percent"],
                             fixed_rms_percent=final[name][mode]["fixed_rms_percent"],
                             squared_error_reduction_fraction=final[name][mode]["squared_error_reduction_fraction"]))
    # Altering held-out scoring values cannot affect calibration-only predictions.
    changed = residual.copy()
    changed[np.broadcast_to(score[:, None], residual.shape)] += 123.
    for modes in operators.values():
        np.testing.assert_allclose(np.einsum("bij,bej->bei", modes["calibration_only"], changed),
                                   np.einsum("bij,bej->bei", modes["calibration_only"], residual), atol=1e-12, rtol=0)
    result = dict(plan=plan, orders=orders.tolist(), exposures=EXPOSURES,
                  included_pixels=int(included.sum()), scoring_pixels_per_exposure=int(score.sum()),
                  development=development, final=final, effective_degrees_of_freedom=effective_df,
                  calibration_leakage_check="passed",
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in inputs+[Path(__file__)]})
    (OUT / "smooth_comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    with (OUT / "smooth_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    np.savez_compressed(OUT / "smooth_results.npz", wavelength_um=wave, orders=orders,
                        exposures=EXPOSURES, included_mask=included, calibration_mask=calibration,
                        score_mask=score, residual_continuum=residual, continuum=continuum, **saved)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, mode in zip(axes, ["calibration_only", "all_included"]):
        x = np.arange(len(CONFIGS))
        ax.plot(x, [development[c["name"]][mode]["continuum_rms_percent"] for c in CONFIGS], "o-", label="Development exposure 20")
        ax.plot(x, [final[c["name"]][mode]["continuum_rms_percent"] for c in CONFIGS], "o-", label="Four evaluation exposures")
        ax.axhline(final["none"][mode]["continuum_rms_percent"], color="0.5", ls="--")
        ax.set(xticks=x, xticklabels=[c["name"] for c in CONFIGS], ylabel="Original-continuum RMS (%)",
               title="Calibration-only prediction" if mode == "calibration_only" else "All-included filtering (score participates)")
        ax.tick_params(axis="x", rotation=70, labelsize=8)
        ax.legend(fontsize=8)
    fig.savefig(OUT / "smooth_comparison.png", dpi=150)
    plt.close(fig)
    print("Development ranking:", selection["development_rank"])
    for row in rows:
        print(f"{row['model']:18s} {row['mode']:17s} dev {row['development_rms_percent']:.6f}% score {row['scoring_rms_percent']:.6f}%")


if __name__ == "__main__":
    main()
