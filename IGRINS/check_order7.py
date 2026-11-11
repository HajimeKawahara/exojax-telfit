"""Independently audit the Order 7 arrays without rebuilding or refitting opacity."""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values: np.ndarray) -> float:
    return float(100 * np.sqrt(np.mean(values ** 2)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    product = OUT / "order7_stellar_fit.npz"
    metadata = json.loads((OUT / "order7_stellar_fit.json").read_text())
    assert digest(product) == metadata["product_sha256"]
    with np.load(product) as saved:
        a = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(a["orders"], [5])
    np.testing.assert_array_equal(a["display_order"], [7])
    np.testing.assert_array_equal(a["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    assert a["included_mask"].shape == (1, 1848) and a["included_mask"].all()
    pixel = np.arange(1848)
    gate = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    np.testing.assert_array_equal(a["score_mask"], gate[None])
    assert int(gate.sum()) == 773
    for value in a.values():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all()
    physical = []
    physical_arrays = []
    for exposure in [10, 30, 50, 70]:
        tag = f"order7_physical_e{exposure}"
        physical.append(json.loads((OUT / f"{tag}.json").read_text()))
        with np.load(OUT / f"{tag}.npz") as saved:
            physical_arrays.append({key: saved[key] for key in saved.files})
    for value in physical_arrays:
        for key in ["observed", "continuum", "component"]:
            np.testing.assert_array_equal(value[key][:, :5], physical_arrays[0][key][:, :5])
    expected = {}
    for key in ["observed", "continuum", "component"]:
        expected[key] = np.concatenate([physical_arrays[0][key][:, :5]]
                                       + [value[key][:, -1:] for value in physical_arrays], axis=1)
    np.testing.assert_array_equal(a["continuum"], expected["continuum"])
    np.testing.assert_array_equal(a["y_over_c"], expected["observed"] / expected["continuum"])
    np.testing.assert_array_equal(a["component_baseline"], expected["component"])
    wave = a["wavelength_um"][0]
    np.testing.assert_array_equal(wave, physical_arrays[0]["wavelength_um"][0])
    x = 2 * (np.log(wave) - np.min(np.log(wave))) / np.ptp(np.log(wave)) - 1
    design = np.polynomial.chebyshev.chebvander(x, 5)
    orthogonal, _ = np.linalg.qr(design)
    scopes = [("training", slice(0, 5), a["included_mask"]),
              ("evaluation_score", slice(5, None), a["score_mask"]),
              ("evaluation_included", slice(5, None), a["included_mask"])]
    rows, q5_errors, metric_errors = {}, {}, []
    for name in ["baseline", "stellar", "refit"]:
        raw = a["y_over_c"] - a[f"component_{name}"]
        residual = raw - (raw @ orthogonal) @ orthogonal.T
        q5_errors[name] = float(np.max(np.abs(residual - a[f"residual_{name}"])))
        assert q5_errors[name] < 1e-11
        rows[name] = residual
        for scope, selection, mask in scopes:
            selected = residual[:, selection]
            keep = np.broadcast_to(mask[:, None], selected.shape)
            saved = metadata["metrics"][name][scope]
            assert saved["n_samples"] == int(keep.sum())
            value = rms(selected[keep])
            metric_errors.append(abs(value - saved["rms_percent"]))
            assert abs(value - saved["rms_percent"]) < 1e-10
            fixed_value = rms((selected * a["continuum"][:, selection])[keep])
            assert abs(fixed_value - saved["fixed_rms_percent"]) < 1e-10
            assert abs(value - saved["per_order"]["5"]) < 1e-10
    np.testing.assert_allclose(a["residual_filtered"], rows["refit"], rtol=0, atol=1e-11)
    np.testing.assert_array_equal(a["component"], a["component_refit"])
    np.testing.assert_array_equal(a["residual_unfiltered"], a["y_over_c"] - a["component"])
    minimum_t = np.min(a["telluric_only_baseline"][:, :5], axis=1)
    star_depth = np.max(1 - a["stellar_only_baseline"][:, :5], axis=1)
    np.testing.assert_array_equal(a["calibration_mask"], minimum_t > .95)
    np.testing.assert_array_equal(a["calibration_telluric"], a["telluric_only_baseline"])
    with np.load(OUT / "order7_physical_species.npz") as saved:
        frame30 = int(np.flatnonzero(a["exposures"] == 30)[0])
        independent_transmission_error = float(np.max(np.abs(
            saved["transmission_all"] - a["telluric_only_baseline"][0, frame30])))
        assert independent_transmission_error < 1e-11
    subsets = dict(strong_stellar=star_depth > .02, deep_telluric=minimum_t <= .15,
                   strong_telluric=minimum_t < .5,
                   moderate_telluric=(minimum_t >= .5) & (minimum_t < .95),
                   clear_telluric=minimum_t >= .95)
    subset_rows = {}
    for label, mask in subsets.items():
        subset_rows[label] = dict(n_wavelengths=int(mask.sum()), n_samples=4 * int(mask.sum()))
        keep = np.broadcast_to(mask[:, None], a["component"][:, 5:].shape)
        for name, residual in rows.items():
            saved = metadata["metrics"][name]["depth_subsets"][label]
            value = rms(residual[:, 5:][keep]) if keep.any() else None
            assert saved["n_wavelengths"] == int(mask.sum()) and saved["n_samples"] == int(keep.sum())
            assert (saved["rms_percent"] is None and value is None) or abs(value - saved["rms_percent"]) < 1e-10
            subset_rows[label][f"{name}_rms_percent"] = value
    q, z = a["q"], a["z"]
    free = a["stellar_free_mask"]
    np.testing.assert_array_equal(q[~free], np.ones((~free).sum()))
    assert q.min() >= 0 and np.all(q <= a["stellar_upper"] + 1e-10)
    assert np.all(a["stellar_prior"] > 0)
    assert np.all(a["strength_prior"] >= 1) and np.all(a["strength_prior"] <= 10)
    expected_prior = 1 + 9 * a["strength_contrast"] ** 2 / (a["strength_contrast"] ** 2 + .005 ** 2)
    np.testing.assert_array_equal(expected_prior, a["strength_prior"])
    uncertainty = np.array([record["uncertainty_upper"] for record in metadata["line_records"]])
    intensity_ratio = 1 + uncertainty * z
    assert np.min(intensity_ratio) >= -1e-10
    largest_intensity = int(np.argmax(intensity_ratio))
    species_counts = dict(Counter(record["species"] for record in metadata["line_records"]))
    assert metadata["free_strength_count"] == len(z) == len(uncertainty)
    assert species_counts == metadata["free_strength_species"]
    assert metadata["stellar_free_features"] == int(free.sum())
    assert metadata["stellar_candidate_features"] == len(q)
    for fit in metadata["fits"].values():
        assert all(row["success"] for row in fit)
    telluric_objective = float(np.sum((rows["refit"][:, :5] / .0045) ** 2)
                                + np.sum((z / a["strength_prior"]) ** 2))
    telluric_objective_error = abs(telluric_objective - metadata["fits"]["telluric"][0]["objective"])
    assert telluric_objective_error < 1e-6
    calibration = a["calibration_mask"][0]
    calibration_q, _ = np.linalg.qr(design[calibration])
    raw_stellar = (a["y_over_c"] - a["component_stellar"])[0, :5][:, calibration]
    filtered_stellar = raw_stellar - (raw_stellar @ calibration_q) @ calibration_q.T
    stellar_objective = float(np.sum((filtered_stellar / .0045) ** 2)
                              + np.sum(((q[free] - 1) / a["stellar_prior"][free]) ** 2))
    stored_stellar_objective = metadata["fits"]["stellar"][0].get("objective")
    stellar_objective_error = (abs(stellar_objective - stored_stellar_objective)
                               if stored_stellar_objective is not None else None)
    assert stellar_objective_error is None or stellar_objective_error < 1e-6
    legacy_hashes = metadata.get("legacy_adopted_sha256", {})
    for filename, expected_hash in legacy_hashes.items():
        assert digest(ROOT / filename) == expected_hash
    for filename, expected_hash in metadata["source_sha256"].items():
        assert digest(ROOT / filename) == expected_hash
    helper = ast.parse((ROOT / "fit_line_strength.py").read_text())
    context_function = next(node for node in helper.body if isinstance(node, ast.FunctionDef)
                            and node.name == "build_context")
    defaults = [ast.literal_eval(value) for value in context_function.args.defaults]
    assert defaults == ["motion_clean_control", "nonstellar_control_e", ("H2O", "CO2")]
    sources = [Path(__file__), product, OUT / "order7_stellar_fit.json",
               ROOT / "fit_line_strength.py", ROOT / "fit_blase_stellar.py",
               ROOT / "fit_order7_stellar.py", OUT / "order7_physical_species.npz",
               OUT / "order7_physical_species.json"]
    sources += [path for name in ["order7_polish_initial.npz", "order7_solver_attempts.json"]
                if (path := OUT / name).exists()]
    star = a["stellar_only"]
    assert np.min(star) >= 0
    assert np.min(a["telluric_only"]) >= -1e-12 and np.max(a["telluric_only"]) <= 1 + 1e-12
    max_location = np.unravel_index(np.argmax(star), star.shape)
    overfill = star > 1 + 1e-6
    results = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        user_order_label=7, released_cube_index=5,
        full_included_wavelengths=1848, evaluation_included_samples=7392,
        scoring_wavelengths=773, evaluation_scoring_samples=3092,
        all_absorption_depths_retained=True, physical_source_arrays_match=True,
        shared_training_arrays_identical=True, independent_q5_max_error=max(q5_errors.values()),
        independent_physical_transmission_max_error=independent_transmission_error,
        independent_metric_max_error=max(metric_errors), subset_metrics=subset_rows,
        independent_telluric_objective_error=telluric_objective_error,
        independent_stellar_objective_error=stellar_objective_error,
        evaluation_score_rms_percent=rms(rows["refit"][:, 5:][:, :, gate]),
        evaluation_all_rms_percent=rms(rows["refit"][:, 5:]),
        stellar_calibration_pixels=int(a["calibration_mask"].sum()),
        stellar_amplitude_bounds_valid=True, fixed_stellar_amplitudes_equal_one=True,
        stellar_candidates=len(q), stellar_free=int(free.sum()),
        free_telluric_strengths=len(z), free_strength_species=species_counts,
        positive_prior_widths=True, intensity_multiplier_range=[float(intensity_ratio.min()), float(intensity_ratio.max())],
        largest_intensity_correction=dict(line=metadata["line_records"][largest_intensity],
            multiplier=float(intensity_ratio[largest_intensity]),
            reference_contrast=float(a["strength_contrast"][largest_intensity]),
            assumed_prior_widths=float(z[largest_intensity] / a["strength_prior"][largest_intensity])),
        near_zero_intensity_count=int(np.sum(intensity_ratio < 1e-5)),
        stellar_amplitude_range=[float(q.min()), float(q.max())],
        stellar_free_near_zero_count=int(np.sum(q[free] < 1e-4)),
        stellar_free_upper_bound_count=int(np.sum(a["stellar_upper"][free] - q[free] < 1e-4)),
        stellar_overfill=dict(maximum_flux=float(star.max()), sample_count=int(overfill.sum()),
            evaluation_sample_count=int(overfill[:, 5:].sum()),
            worst_exposure=int(a["exposures"][max_location[1]]),
            worst_wavelength_um=float(wave[max_location[2]])),
        physical_parameters_at_bounds=physical[0]["bounds_hit"],
        numerically_stabilized=metadata["numerically_stabilized"],
        fit_optimalities={name: [record.get("optimality") for record in fits]
                         for name, fits in metadata["fits"].items()},
        diagnostic_flags=(["Stellar-only model exceeds unity in included pixels."] if overfill.any() else [])
                         + (["Two consecutive small null-refit drifts were not obtained."]
                            if not metadata["numerically_stabilized"] else [])
                         + ["CO and CO2 physical abundances reached their lower nuisance bounds."],
        old_adopted_products_unchanged=True, legacy_protected_file_count=len(legacy_hashes),
        old_default_context_arguments_unchanged=True,
        recorded_source_hashes_match=True,
        shared_fit_data_flow_review="Stellar and telluric targets, calibration mask and model-response selection use only the first five frames; code inspected without refitting.",
        default_compatibility_review="Default species order, source files, arrays and old strength/stellar priors retain the historical path. Runtime historical opacity refit was not repeated.",
        caveats=["Held-out nuisance parameters and Q5 are estimated on evaluation frames; validation remains conditional.",
                 "No independent planet injection test has been performed for this order.",
                 "Strongly absorbing pixels are included regardless of whether their residuals are small."],
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources})
    (OUT / "order7_independent_audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({key: results[key] for key in ["passed", "evaluation_score_rms_percent",
          "evaluation_all_rms_percent", "stellar_overfill", "numerically_stabilized"]}, indent=2))


if __name__ == "__main__":
    main()
