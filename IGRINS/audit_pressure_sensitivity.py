"""Audit saved pressure sensitivity spectra without JAX or opacity calculations."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    diagnostic = json.loads((OUT / "pressure_sensitivity.json").read_text())
    manifest = json.loads((OUT / "adopted_telluric_model.json").read_text())
    saved = np.load(OUT / "pressure_sensitivity.npz")
    adopted = np.load(OUT / "adopted_telluric_fit.npz")
    source_checks = {}
    for relative, expected in diagnostic["source_sha256"].items():
        actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        source_checks[relative] = actual == expected
    assert all(source_checks.values())
    product_hash = hashlib.sha256((OUT / "adopted_telluric_fit.npz").read_bytes()).hexdigest()
    assert product_hash == manifest["product_sha256"]
    for name in ["wavelength_um", "included_mask", "score_mask"]:
        np.testing.assert_array_equal(saved[name], adopted[name])
    np.testing.assert_array_equal(adopted["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    bases = []
    for wave, keep in zip(adopted["wavelength_um"], adopted["included_mask"]):
        log_wave = np.log(wave)
        coordinate = 2 * (log_wave - log_wave.min()) / np.ptp(log_wave) - 1
        basis = np.polynomial.chebyshev.chebvander(coordinate, 5)
        q, upper = np.linalg.qr(basis[keep], mode="reduced")
        bases.append((basis, q, upper, keep))

    def project(values):
        output = []
        for row, (basis, q, upper, keep) in zip(values, bases):
            coefficients = np.linalg.solve(upper, q.T @ row[:, keep].T)
            output.append(row - (basis @ coefficients).T)
        return np.array(output)

    def metrics(component):
        residual = project(adopted["y_over_c"] - component)
        result = {}
        for scope, es, keep in [("training", slice(0, 5), adopted["included_mask"]),
                                ("evaluation_score", slice(5, None), adopted["score_mask"]),
                                ("evaluation_included", slice(5, None), adopted["included_mask"])]:
            rows = residual[:, es]
            mask = np.broadcast_to(keep[:, None], rows.shape)
            result[scope] = dict(n_samples=int(mask.sum()),
                rms_percent=float(100 * np.sqrt(np.mean(rows[mask]**2))),
                fixed_rms_percent=float(100 * np.sqrt(np.mean(
                    (rows * adopted["continuum"][:, es])[mask]**2))),
                per_order={str(order):float(100 * np.sqrt(np.mean(row[:, pixels]**2)))
                    for order, row, pixels in zip(adopted["orders"], rows, keep)})
        return result, residual

    recomputed = {}
    errors = []
    for tag in ["baseline", *diagnostic["cases"]]:
        actual, _ = metrics(saved[f"component_{tag}"])
        expected = diagnostic["baseline_metrics"] if tag == "baseline" else diagnostic["cases"][tag]["metrics"]
        for scope in actual:
            assert actual[scope]["n_samples"] == expected[scope]["n_samples"]
            for key in ["rms_percent", "fixed_rms_percent"]:
                errors.append(abs(actual[scope][key] - expected[scope][key]))
            for order in actual[scope]["per_order"]:
                errors.append(abs(actual[scope]["per_order"][order] - expected[scope]["per_order"][order]))
        recomputed[tag] = actual
    assert max(errors) < 1e-9
    _, residual = metrics(saved["component_baseline"])
    residual_error = float(np.max(abs(residual - saved["filtered_residual_baseline"])))
    assert residual_error < 1e-10
    baseline_error = float(np.max(abs(saved["component_baseline"] - adopted["component"])))
    assert baseline_error < 1e-10
    directions = np.stack([project((saved[f"component_{name}_plus"]
                                   - saved[f"component_{name}_minus"]) / 2)
                           for name in diagnostic["parameter_names"]], axis=-1)
    direction_error = float(np.max(abs(directions - saved["filtered_scaled_directions"])))
    assert direction_error < 1e-10
    keep = np.broadcast_to(adopted["included_mask"][:, None], residual[:, :5].shape)
    matrix = directions[:, :5][keep]
    y = residual[:, :5][keep]
    sigma = .0045
    normal = matrix.T @ matrix / sigma**2 + np.eye(4)
    rhs = matrix.T @ y / sigma**2
    scaled = np.linalg.solve(normal, rhs)
    assert np.max(abs(scaled)) < 3
    parameter_error = float(np.max(abs(.1 * scaled - saved["global_parameters"])))
    assert parameter_error < 1e-9
    exact_change = project(saved["component_global_width_fit_fixed_strengths"] - saved["component_baseline"])
    global_error = float(np.max(abs(exact_change - directions @ scaled)))
    claimed_global = diagnostic["cases"]["global_width_fit_fixed_strengths"]["linear_prediction_max_error"]
    assert abs(global_error - claimed_global) < 1e-10
    changed = residual.copy()
    changed[:, 5:] = 1234.
    altered_scaled = np.linalg.solve(normal, matrix.T @ changed[:, :5][keep] / sigma**2)
    np.testing.assert_array_equal(scaled, altered_scaled)
    joint_error = diagnostic["cases"]["local_joint_width_strength_fit"]["linear_prediction_training_error"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        method="Independent NumPy degree-five QR projection of saved exact spectra; no JAX or opacity evaluation",
        source_hash_checks=source_checks, adopted_product_sha256=product_hash,
        metrics=recomputed, maximum_metric_difference_percentage_points=max(errors),
        baseline_component_difference=baseline_error,
        filtered_residual_difference=residual_error, filtered_direction_difference=direction_error,
        independently_resolved_global_parameters=(.1 * scaled).tolist(),
        global_parameter_difference=parameter_error,
        global_linearization_max_error_flux=global_error,
        global_linearization_max_error_percent=100 * global_error,
        global_normal_equation_relative_residual=float(np.linalg.norm(normal @ scaled - rhs) / np.linalg.norm(rhs)),
        joint_linearization_training_error_flux_reported=joint_error,
        joint_linearization_training_error_percent_reported=100 * joint_error,
        joint_linearization_independently_verified=False,
        joint_verification_limit="The saved product omits the 269-column intensity tangent and its prediction. The exact joint-fit RMS is independently verified; its reported linearization error cannot be reconstructed from saved spectra alone.",
        leakage_review=dict(training_exposures=adopted["exposures"][:5].tolist(),
            evaluation_exposures=adopted["exposures"][5:].tolist(),
            altered_evaluation_data_changes_global_fit=False,
            source_review="Both least-squares fits use residual[:, :5] and directions[:, :5]. Strength derivatives use y_over_c[:, :5]. Evaluation spectra enter only metrics after the fixed trial coefficients or training-derived coefficients have been chosen.",
            limit="Conditional on the pre-existing adopted physical model, masks and strength corrections; this does not audit earlier model-selection history."),
        inspected_source_sha256=hashlib.sha256((ROOT / "diagnose_pressure_sensitivity.py").read_bytes()).hexdigest())
    (OUT / "pressure_sensitivity_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["passed", "maximum_metric_difference_percentage_points",
          "global_parameter_difference", "global_linearization_max_error_flux",
          "joint_linearization_training_error_flux_reported"]}, indent=2))


if __name__ == "__main__":
    main()
