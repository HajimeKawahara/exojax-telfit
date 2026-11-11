"""Independently audit saved profile experiments using NumPy QR projections."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    meta = json.loads((OUT / "vertical_profile_experiment.json").read_text())
    data = np.load(OUT / "vertical_profile_experiment.npz")
    adopted = np.load(OUT / "adopted_telluric_fit.npz")
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
              for name, expected in meta["source_sha256"].items()}
    assert all(hashes.values())
    for name in ["wavelength_um", "included_mask", "score_mask"]:
        np.testing.assert_array_equal(data[name], adopted[name])
    bases = []
    for wave, keep in zip(data["wavelength_um"], data["included_mask"]):
        x = np.log(wave)
        x = 2 * (x - x.min()) / np.ptp(x) - 1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, upper = np.linalg.qr(basis[keep], mode="reduced")
        bases.append((basis, q, upper, keep))

    def project(values):
        return np.array([row - (basis @ np.linalg.solve(upper, q.T @ row[:, keep].T)).T
            for row, (basis, q, upper, keep) in zip(values, bases)])

    differences, metrics = [], {}
    for name, case in meta["cases"].items():
        residual = project(adopted["y_over_c"] - data[f"component_{name}"])
        actual = {}
        for scope, es, mask in [("training", slice(0, 5), data["included_mask"]),
                                ("evaluation_score", slice(5, None), data["score_mask"]),
                                ("evaluation_included", slice(5, None), data["included_mask"])]:
            rows = residual[:, es]
            keep = np.broadcast_to(mask[:, None], rows.shape)
            value = float(100 * np.sqrt(np.mean(rows[keep]**2)))
            differences.append(abs(value - case["metrics"][scope]["rms_percent"]))
            actual[scope] = value
        metrics[name] = actual
    assert max(differences) < 1e-9
    baseline = data["component_abundance_consistent_self"]
    directions = np.stack([project((data[f"component_{name}_plus"]
                                   - data[f"component_{name}_minus"]) / 2)
                            for name in meta["parameter_names"]], axis=-1)
    np.testing.assert_allclose(directions, data["filtered_directions"], atol=1e-11, rtol=0)
    residual = project(adopted["y_over_c"] - baseline)
    keep = np.broadcast_to(data["included_mask"][:, None], residual[:, :5].shape)
    matrix = directions[:, :5][keep]
    y = residual[:, :5][keep]
    normal = matrix.T @ matrix / .0045**2 + np.eye(3)
    solution = np.linalg.solve(normal, matrix.T @ y / .0045**2)
    # Unconstrained and bounded solutions coincide only if the optimum is interior.
    interior = bool(np.max(abs(solution)) < 3)
    if interior:
        np.testing.assert_allclose(solution, data["fixed_scaled_parameters"], atol=1e-8, rtol=0)
    strength = data["training_strength_tangent"]
    errors = {}
    for name, design, vector in [
        ("profile_fixed_strengths", matrix, data["fixed_scaled_parameters"]),
        ("self_strength_profile", strength, data["self_strength_delta"]),
        ("profile_joint_strengths", np.column_stack([strength, matrix]), data["joint_delta"])]:
        exact = project(data[f"component_{name}"] - baseline)[:, :5][keep]
        error = float(np.sqrt(np.mean((exact - design @ vector)**2)))
        expected = meta["cases"][name]["linear_prediction_training_error"]
        assert abs(error - expected) < 1e-11
        errors[name] = dict(flux=error, percent=100 * error)
    reference_column = data["reference_number_column_cm2"] @ data["reference_water_vmr"]
    column_ratios = {tag: float(data["reference_number_column_cm2"] @ data[f"water_vmr_{tag}"]
                                / reference_column) for tag in ["fixed", "joint"]}
    assert max(abs(ratio - 1) for ratio in column_ratios.values()) < 1e-12
    objectives = {}
    for name, z, q in [
        ("adopted", data["adopted_strength_z"], np.zeros(3)),
        ("abundance_consistent_self", data["adopted_strength_z"], np.zeros(3)),
        ("profile_fixed_strengths", data["adopted_strength_z"], data["fixed_scaled_parameters"]),
        ("self_strength_profile", data["adopted_strength_z"] + data["self_strength_delta"], np.zeros(3)),
        ("profile_joint_strengths", data["adopted_strength_z"] + data["joint_delta"][:-3], data["joint_delta"][-3:])]:
        filtered = project(adopted["y_over_c"] - data[f"component_{name}"])[:, :5][keep]
        data_term = float(np.sum((filtered / .0045)**2))
        prior_term = float(np.sum((z / data["strength_prior_widths"])**2) + q @ q)
        objectives[name] = dict(data=data_term, prior=prior_term, total=data_term + prior_term)
    report = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
        method="Independent NumPy degree-five QR; saved spectra and tangent products only",
        source_hash_checks=hashes, metrics=metrics,
        maximum_rms_difference_percentage_points=max(differences),
        linearization_training_rms_errors=errors,
        fixed_unconstrained_solution_is_interior=interior,
        fixed_independent_scaled_solution=solution.tolist(),
        exact_training_objectives=objectives,
        joint_profile_at_bound=[name for name, value in zip(meta["parameter_names"], data["joint_delta"][-3:])
                                if abs(value) > 3 - 1e-8],
        interpretation="Exact candidate spectra are verified. The joint local solution reaches the lower-temperature amplitude bound; it is not a converged nonlinear atmospheric retrieval.",
        water_column_ratios=column_ratios,
        leakage_review="Both profile and joint solves construct residuals and response matrices from the first five exposures only; evaluation spectra only enter reporting.")
    (OUT / "vertical_profile_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "maximum_rms_difference_percentage_points": max(differences),
                      "linearization_training_rms_errors": errors}, indent=2))


if __name__ == "__main__":
    main()
