"""Independently verify smooth residual corrections and scoring arithmetic."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def reference_prediction(wave, mask, config, residual):
    """Predict each exposure using direct solves and an independent QR fit."""
    if config["kind"] == "none":
        return np.zeros_like(residual)
    velocity = 299792.458 * np.log(wave)
    if config["kind"] == "polynomial":
        coordinate = 2 * (velocity - velocity.min()) / np.ptp(velocity) - 1
        basis = np.polynomial.chebyshev.chebvander(coordinate, config["degree"])
        orthogonal, triangular = np.linalg.qr(basis[mask], mode="reduced")
        coefficients = np.linalg.solve(triangular, orthogonal.T @ residual[:, mask].T)
        return (basis @ coefficients).T
    if config["kind"] == "gp":
        fit_velocity = velocity[mask]
        covariance = config["amplitude"] ** 2 * np.exp(
            -0.5 * ((fit_velocity[:, None] - fit_velocity[None, :])
                    / config["length_kms"]) ** 2)
        covariance += config["noise"] ** 2 * np.eye(mask.sum())
        coefficients = np.linalg.solve(covariance, residual[:, mask].T)
        cross_covariance = config["amplitude"] ** 2 * np.exp(
            -0.5 * ((velocity[:, None] - fit_velocity[None, :])
                    / config["length_kms"]) ** 2)
        return (cross_covariance @ coefficients).T
    raise ValueError(config["kind"])


def rms(values, mask):
    return float(100 * np.sqrt(np.mean(values[mask] ** 2)))


def audit_injections(original):
    """Recompute isolated-line raw and local-contrast gains without shared code."""
    path = OUT / "smooth_signal_retention.json"
    if not path.exists():
        return dict(status="not available")
    injection = json.loads(path.read_text())
    qualified = []
    largest_error = 0.0
    for setting in injection["settings"]:
        pooled_all_gain = {}
        for protocol, protocol_report in setting["protocols"].items():
            if protocol == "blocked":
                fit_mask = original["calibration_mask"][:, -1]
                score_mask = original["score_mask"][:, -1]
            else:
                fit_mask = original["included_mask"]
                score_mask = fit_mask
            for width_string, width_report in protocol_report["isolated"].items():
                all_gain = []
                for b, order in enumerate(original["orders"]):
                    local = width_report["per_order"][str(order)]
                    centers = np.asarray(local["centers"])
                    velocity = 299792.458 * np.log(original["wavelength_um"][b])
                    offsets = velocity[None, :] - velocity[centers, None]
                    sigma = float(width_string) / np.sqrt(8 * np.log(2))
                    signal = -injection["depth"] * np.exp(-0.5 * (offsets / sigma) ** 2)
                    after = signal - reference_prediction(original["wavelength_um"][b], fit_mask[b],
                                                           setting["config"], signal)
                    before_score, after_score = signal[:, score_mask[b]], after[:, score_mask[b]]
                    gain = np.sum(before_score * after_score, axis=1) / np.sum(before_score ** 2, axis=1)
                    contrast = []
                    for j in range(len(centers)):
                        window = score_mask[b] & (np.abs(offsets[j]) <= 35)
                        line = signal[j, window] - signal[j, window].mean()
                        filtered = after[j, window] - after[j, window].mean()
                        contrast.append(float(line @ filtered / (line @ line)))
                    np.testing.assert_allclose(gain, local["raw_gain"], atol=1e-9, rtol=0)
                    np.testing.assert_allclose(contrast, local["contrast_gain"], atol=1e-9, rtol=0)
                    largest_error = max(largest_error, float(np.max(np.abs(gain-local["raw_gain"]))),
                                        float(np.max(np.abs(np.asarray(contrast)-local["contrast_gain"]))))
                    all_gain.extend(gain)
                np.testing.assert_allclose(np.median(all_gain), width_report["pooled"]["raw_gain"]["median"], atol=1e-9, rtol=0)
                np.testing.assert_allclose(np.percentile(all_gain, 5), width_report["pooled"]["raw_gain"]["p05"], atol=1e-9, rtol=0)
                if protocol == "all_included":
                    pooled_all_gain[width_string] = all_gain
        gate = pooled_all_gain["10.0"]
        passed = np.median(gate) >= .95 and np.percentile(gate, 5) >= .90
        assert passed == setting["qualification"]["passed"]
        if passed:
            qualified.append(setting["name"])
    return dict(status="passed", independent_maximum_gain_difference=largest_error,
                raw_gain_qualified_candidates=qualified,
                interpretation="Raw gain includes removed broad/DC signal; local contrast conditions out a window mean.")


def main():
    report_path = OUT / "smooth_comparison.json"
    array_path = OUT / "smooth_results.npz"
    report = json.loads(report_path.read_text())
    with np.load(array_path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    with np.load(OUT / "motion_clean_control.npz") as source:
        original = {key: source[key] for key in source.files}
    residual, continuum = [], []
    input_paths = [report_path, array_path, OUT / "motion_clean_control.npz", Path(__file__)]
    injection_path = OUT / "smooth_signal_retention.json"
    if injection_path.exists():
        input_paths.append(injection_path)
    for exposure in report["exposures"]:
        path = OUT / f"nonstellar_control_e{exposure}.npz"
        input_paths.append(path)
        with np.load(path) as fitted:
            np.testing.assert_array_equal(fitted["included_mask"], original["included_mask"])
            np.testing.assert_array_equal(fitted["wavelength_um"], original["wavelength_um"])
            np.testing.assert_array_equal(fitted["model"][:, :-1], original["model"][:, :-1])
            current = (fitted["observed"][:, -1] - fitted["model"][:, -1]) / fitted["continuum"][:, -1]
            residual.append(current)
            continuum.append(fitted["continuum"][:, -1])
    residual = np.stack(residual, axis=1)
    continuum = np.stack(continuum, axis=1)
    np.testing.assert_array_equal(residual, arrays["residual_continuum"])
    np.testing.assert_array_equal(continuum, arrays["continuum"])
    dev = (original["observed"][:, -1:] - original["model"][:, -1:]) / original["continuum"][:, -1:]
    score = original["score_mask"][:, -1]
    calibration = original["calibration_mask"][:, -1]
    included = original["included_mask"]
    np.testing.assert_array_equal(score, arrays["score_mask"])
    np.testing.assert_array_equal(calibration, arrays["calibration_mask"])
    np.testing.assert_array_equal(included, arrays["included_mask"])
    assert not np.any(score & calibration)
    full_score = np.broadcast_to(score[:, None], residual.shape)
    full_included = np.broadcast_to(included[:, None], residual.shape)
    dev_score = np.broadcast_to(score[:, None], dev.shape)
    baseline = np.mean(residual[full_score] ** 2)
    records = []
    for config in report["plan"]["configs"]:
        name = config["name"]
        for mode, mask in [("calibration_only", calibration), ("all_included", included)]:
            correction = np.array([reference_prediction(w, m, config, values)
                                   for w, m, values in zip(original["wavelength_um"], mask, residual)])
            maximum_difference = float(np.max(np.abs(correction - arrays[f"correction_{name}_{mode}"])))
            np.testing.assert_allclose(correction, arrays[f"correction_{name}_{mode}"], atol=2e-11, rtol=2e-9)
            remaining = residual - correction
            metrics = report["final"][name][mode]
            np.testing.assert_allclose(rms(remaining, full_score), metrics["continuum_rms_percent"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(rms(remaining * continuum, full_score), metrics["fixed_rms_percent"], atol=1e-9, rtol=0)
            np.testing.assert_allclose(rms(remaining, full_included), metrics["all_included_metrics"]["continuum_rms_percent"], atol=1e-9, rtol=0)
            reduction = float(1 - np.mean(remaining[full_score] ** 2) / baseline)
            np.testing.assert_allclose(reduction, metrics["squared_error_reduction_fraction"], atol=1e-9, rtol=0)
            dev_correction = np.array([reference_prediction(w, m, config, values)
                                       for w, m, values in zip(original["wavelength_um"], mask, dev)])
            np.testing.assert_allclose(rms(dev - dev_correction, dev_score),
                                       report["development"][name][mode]["continuum_rms_percent"], atol=1e-9, rtol=0)
            for b, order in enumerate(original["orders"]):
                local_mask = np.broadcast_to(score[b], remaining[b].shape)
                np.testing.assert_allclose(rms(remaining[b], local_mask),
                                           metrics["per_order"][str(order)]["continuum_rms_percent"], atol=1e-9, rtol=0)
            if mode == "calibration_only":
                altered = residual.copy()
                altered[full_score] += 1.234
                for b, (w, m) in enumerate(zip(original["wavelength_um"], mask)):
                    np.testing.assert_array_equal(reference_prediction(w, m, config, altered[b]), correction[b])
            records.append(dict(model=name, mode=mode,
                                maximum_correction_difference=maximum_difference,
                                score_rms_percent=rms(remaining, full_score)))
    ranking = sorted(report["plan"]["configs"],
                     key=lambda c: report["development"][c["name"]]["calibration_only"]["continuum_rms_percent"])
    selection = json.loads((OUT / "smooth_selection.json").read_text())
    assert selection["development_rank"] == [c["name"] for c in ranking]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  scoring_samples=int(full_score.sum()), records=records,
                  isolated_injection_audit=audit_injections(original),
                  checked=["Independent residual reconstruction from original observed/model/continuum arrays",
                           "Independent QR polynomial fit and direct linear-solve GP prediction",
                           "Calibration/scoring masks and unchanged training predictions",
                           "All candidate corrections and development/evaluation RMS values",
                           "Per-order, prior-flux-unit and all-included RMS values",
                           "Scoring perturbations leave calibration-only correction unchanged",
                           "Development-only candidate ranking"],
                  limitations=["This audit does not establish the retention of real planetary molecular spectra.",
                               "All-included output RMS uses pixels involved in smoothing and is descriptive.",
                               "The fitted physical model is frozen; its earlier signal attenuation is not measured."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in input_paths})
    destination = OUT / "smooth_independent_audit.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Passed: {len(records)} candidate/mode combinations; maximum correction discrepancy "
          f"{max(r['maximum_correction_difference'] for r in records):.3g}")


if __name__ == "__main__":
    main()
