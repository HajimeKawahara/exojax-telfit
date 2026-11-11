"""Independently validate the extended fixed-stellar telluric experiments.

This checks data selection, continuum isolation, optimized parameter groups,
saved metrics, flexible profiles, and numerical identifiability diagnostics.
It does not turn a successful numerical fit into a unique physical inference.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np

from atmosphere_profile import load_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIXED_STAR = ["stellar_rv_reference_kms", "stellar_extra_sigma_kms"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_profile(meta):
    """Reconstruct pressure, node profiles, water columns and mass budget in NumPy."""
    if "fitted_profile" not in meta:
        return None
    reference = load_profile(meta["layers"])
    scale = meta.get("pressure_scale", 1.)
    pressure = reference["pressure_bar"] * scale
    temperature = reference["temperature_K"] + meta["temperature_offset_K"]
    parameters = meta["parameters"]
    nodes = meta["temperature_node_pressures_bar"]
    delta = (np.interp(np.log(pressure), np.log(nodes),
                       [parameters[f"temperature_delta_node{i}_K"] for i in range(len(nodes))])
             if nodes else np.zeros(len(pressure)))
    if meta["water_shape"] == "tilt":
        shape = parameters["water_shape_node0_dex"] * np.log10(pressure / .4)
    elif meta["water_shape"] == "nodes":
        shape = np.interp(np.log(pressure), np.log([.1, .35, .7]),
                          [parameters["water_shape_node0_dex"], parameters["water_shape_node1_dex"], 0.])
    else:
        shape = np.zeros(len(pressure))
    column = reference["number_column_cm2"] * scale
    reference_water = column * reference["vmr"]["H2O"]
    weights = 10.**(shape-shape.max())
    weights *= reference_water.sum() / np.sum(reference_water*weights)
    exposures = meta["train_exposures"] + [meta["heldout_exposure"]]
    water = np.array([reference_water*weights*10**parameters[f"log10_water_e{e}"] for e in exposures])
    mass = column * reference["mean_molecular_weight"]
    dry = (mass[None, :] - 18.01528*water) / 28.9647
    assert np.all(dry > 0)
    vmr = water/(water+dry)
    dry_ratio = dry/(column*(1-reference["vmr"]["H2O"]))
    fitted = meta["fitted_profile"]
    for key, expected in [("pressure_bar", pressure), ("temperature_K", temperature+delta),
                          ("temperature_delta_K", delta), ("water_layer_weights", weights),
                          ("water_layer_columns_cm2", water), ("mass_conserving_water_vmr", vmr),
                          ("mass_conserving_dry_column_ratio", dry_ratio)]:
        np.testing.assert_allclose(fitted[key], expected, rtol=2e-12, atol=2e-12, err_msg=key)
    assert np.all((temperature+delta >= 150) & (temperature+delta <= 400))
    np.testing.assert_allclose(water.sum(axis=1), meta["vertical_columns_cm-2"]["H2O"], rtol=2e-12)
    np.testing.assert_allclose(np.sum(reference_water*weights), reference_water.sum(), rtol=2e-12)
    exact = meta["opacity_interpolation_check"]
    assert exact["rms_component_difference"] < 5e-4
    assert exact["max_component_difference"] < 1e-3
    for model_key, exact_key in [("training_fixed_scale_metrics", "exact_training_fixed_scale_metrics"),
                                 ("heldout_fixed_scale_metrics", "exact_heldout_fixed_scale_metrics")]:
        assert meta[model_key]["n_pixels"] == exact[exact_key]["n_pixels"]
        assert abs(meta[model_key]["rms"]-exact[exact_key]["rms"]) < 5e-4
    return dict(maximum_water_vmr=float(vmr.max()),
                maximum_dry_column_fractional_change=float(np.max(np.abs(dry_ratio-1))),
                molecular_column_conservation_error=float(np.sum(reference_water*weights)/reference_water.sum()-1),
                exact_opacity_rms_component_difference=exact["rms_component_difference"],
                exact_opacity_max_component_difference=exact["max_component_difference"],
                interpretation="Hydrostatic VMR is a diagnostic; the fit keeps the background dry columns fixed. Temperature/water node boundaries and normalized Jacobian conditioning require physical interpretation.")


def check_variant(tag, wavelength, cube, source, matched):
    meta_path, data_path = OUT / f"{tag}.json", OUT / f"{tag}.npz"
    meta = json.loads(meta_path.read_text())
    with np.load(data_path) as saved:
        data = {key: saved[key] for key in saved.files}
    mask_path = Path(meta["pixel_mask"]["path"])
    assert meta["pixel_mask"]["sha256"] == file_hash(mask_path)
    with np.load(mask_path) as saved:
        mask_data = {key: saved[key] for key in saved.files}
    assert meta["fixed_parameters"] == FIXED_STAR
    assert meta["train_exposures"][0] == source["train_exposures"][0] == 0
    for name in FIXED_STAR:
        assert meta["parameters"][name] == source["parameters"][name]
    expected_heldout = [f"log10_water_e{meta['heldout_exposure']}",
                        f"calibration_rv_e{meta['heldout_exposure']}_kms"]
    if meta.get("ip_epoch", False):
        expected_heldout.append(f"log_ip_width_e{meta['heldout_exposure']}")
    quality = {}
    for key in ("training_fit", "heldout_calibration_fit"):
        fit = meta[key]
        assert fit["success"], (key, fit["message"])
        assert fit["jacobian_relative_error"] < 1e-3, (key, fit["jacobian_relative_error"])
        if "optimized_parameters" in fit:
            expected = (expected_heldout if key == "heldout_calibration_fit" else
                        set(meta["parameters"]) - set(FIXED_STAR) - set(expected_heldout))
            reused = bool(fit.get("reused", False))
            if reused:
                assert key == "training_fit" and fit["nfev"] == 0
                assert fit["optimized_parameters"] == []
                assert set(fit["reused_parameters"]) == set(expected)
                names = fit["reused_parameters"]
                previous_path = Path(fit["source_path"])
                previous = json.loads(previous_path.read_text())
                assert file_hash(previous_path) == meta["provenance"]["initial_sha256"]
                assert previous["training_fit"]["success"]
                for name in names + FIXED_STAR:
                    assert meta["parameters"][name] == previous["parameters"][name], name
                omitted = {"holdout", "initial", "tag", "max_nfev", "check_layers", "freeze_shared"}
                current_args = meta["provenance"]["command_arguments"]
                assert current_args["freeze_shared"]
                for name, value in current_args.items():
                    if name not in omitted:
                        assert value == previous["provenance"]["command_arguments"].get(name), name
                with np.load(previous_path.with_suffix(".npz")) as old:
                    for name in ["observed", "model", "component", "continuum"]:
                        np.testing.assert_allclose(data[name][:, :-1], old[name][:, :-1], rtol=0, atol=1e-10,
                                                   err_msg=f"Frozen training {name}")
            else:
                assert set(fit["optimized_parameters"]) == set(expected)
                names = fit["optimized_parameters"]
            singular = np.asarray(fit["normalized_jacobian_singular_values"])
            assert len(singular) == len(names)
            assert np.all(np.isfinite(singular)) and np.all(singular >= 0)
            assert np.all(np.diff(singular) <= 0)
            quality[key] = dict(normalized_jacobian_condition=float(singular[0]/max(singular[-1], 1e-30)),
                                smallest_normalized_singular_value=float(singular[-1]),
                                singular_values_below_1e_minus_3_relative=int(np.count_nonzero(singular < .001*singular[0])),
                                optimized_parameter_count=0 if reused else len(singular),
                                reused_training_diagnostics=reused,
                                reused_parameter_count=len(names) if reused else 0)
    lo, hi = meta["pixels"]
    n_pixel = hi-lo
    pixel = np.arange(n_pixel)
    cal_blocks = ((pixel//80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    score_blocks = ((pixel//80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    degree = meta.get("continuum_degree", 3)
    basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, n_pixel), degree)
    orders, exposures = data["orders"], data["exposures"]
    np.testing.assert_array_equal(exposures, meta["train_exposures"] + [meta["heldout_exposure"]])
    allowed = np.stack([mask_data["mask"][np.flatnonzero(mask_data["orders"] == o).item()] for o in orders])
    np.testing.assert_array_equal(data["included_mask"], allowed)
    training = np.broadcast_to(allowed[:, None, :], data["observed"].shape).copy()
    training[:, -1] = False
    calibration = np.zeros_like(training); calibration[:, -1] = allowed & cal_blocks
    scoring = np.zeros_like(training); scoring[:, -1] = allowed & score_blocks
    for key, expected in [("train_mask", training), ("calibration_mask", calibration), ("score_mask", scoring)]:
        np.testing.assert_array_equal(data[key], expected)
    assert not np.any(training & calibration | training & scoring | calibration & scoring)
    for key in ("observed", "model", "component", "continuum", "residual_continuum"):
        assert np.all(np.isfinite(data[key])), key
    assert np.all(data["continuum"] > 0)
    np.testing.assert_allclose(data["model"], data["component"]*data["continuum"], atol=1e-13)
    np.testing.assert_allclose(data["residual_continuum"],
                               (data["observed"]-data["model"])/data["continuum"], atol=1e-13)
    max_continuum_error, max_perturbation_change = 0., 0.
    for b, order in enumerate(orders):
        np.testing.assert_allclose(data["wavelength_um"][b], wavelength[order, lo:hi][::-1], rtol=0, atol=1e-12)
        flux = cube[order, exposures, lo:hi][:, ::-1]
        scales = np.percentile(flux[:, allowed[b]], 95, axis=1)
        scales[-1] = np.percentile(flux[-1, calibration[b, -1]], 95)
        np.testing.assert_array_equal(data["observed"][b], flux/scales[:, None])
        for e in range(len(exposures)):
            weight = (training | calibration)[b, e].astype(float)
            design = basis * data["component"][b, e, :, None]
            normal = design.T @ (weight[:, None]*design) + 1e-12*np.eye(degree+1)
            observed = data["observed"][b, e]
            continuum = basis @ np.linalg.solve(normal, design.T @ (weight*observed))
            max_continuum_error = max(max_continuum_error, float(np.max(np.abs(continuum-data["continuum"][b, e]))))
            np.testing.assert_allclose(continuum, data["continuum"][b, e], rtol=0, atol=1e-9)
            perturbed = observed + (1-weight)*(10+pixel/n_pixel)
            alternative = basis @ np.linalg.solve(normal, design.T @ (weight*perturbed))
            max_perturbation_change = max(max_perturbation_change, float(np.max(np.abs(alternative-continuum))))
            np.testing.assert_array_equal(alternative, continuum)
    for key, residual, mask in [("training_metrics", data["residual_continuum"], training),
                                ("heldout_score_metrics", data["residual_continuum"], scoring),
                                ("training_fixed_scale_metrics", data["observed"]-data["model"], training),
                                ("heldout_fixed_scale_metrics", data["observed"]-data["model"], scoring)]:
        assert meta[key]["n_pixels"] == np.count_nonzero(mask)
        np.testing.assert_allclose(meta[key]["rms"], np.sqrt(np.mean(residual[mask]**2)), rtol=1e-12)
    group = (tuple(orders), tuple(exposures), tuple(meta["pixels"]), meta["pixel_mask"]["sha256"])
    if group in matched:
        for key in ("observed", "orders", "exposures", "wavelength_um", "included_mask",
                    "train_mask", "calibration_mask", "score_mask"):
            np.testing.assert_array_equal(data[key], matched[group][key])
    else:
        matched[group] = data
    return dict(passed=True, source_json_sha256=file_hash(meta_path), source_npz_sha256=file_hash(data_path),
                fit_code_sha256=meta.get("provenance", {}).get("fit_code_sha256"),
                continuum_reconstruction_max_error=max_continuum_error,
                excluded_flux_perturbation_max_change=max_perturbation_change,
                training_pixels=int(training.sum()), calibration_pixels=int(calibration.sum()),
                scoring_pixels=int(scoring.sum()), training_exposures=meta["train_exposures"],
                heldout_exposure=meta["heldout_exposure"], continuum_degree=degree,
                training_fixed_scale_rms=meta["training_fixed_scale_metrics"]["rms"],
                heldout_fixed_scale_rms=meta["heldout_fixed_scale_metrics"]["rms"],
                bounds_hit=meta["bounds_hit"], profile=check_profile(meta), identifiability=quality)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+")
    parser.add_argument("--output", default="investigation_checks")
    args = parser.parse_args()
    tags = args.tags or [path.stem for path in sorted(list(OUT.glob("inv*.json")) + list(OUT.glob("final_*.json")))
                         if path.with_suffix(".npz").exists()]
    if not tags:
        raise ValueError("No complete investigation fit outputs found.")
    source = json.loads((OUT / "joint_star_profile.json").read_text())
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True, variants={},
                  validation_code_sha256=file_hash(__file__),
                  profile_opacity_code_sha256=file_hash(ROOT / "profile_opacity.py"),
                  fixed_star_source_sha256=file_hash(OUT / "joint_star_profile.json"))
    matched = {}
    for tag in tags:
        try:
            report["variants"][tag] = check_variant(tag, wavelength, cube, source, matched)
        except (AssertionError, ValueError, KeyError) as error:
            report["variants"][tag] = dict(passed=False, failure=repr(error))
            report["passed"] = False
        print(tag, report["variants"][tag]["passed"], flush=True)
    report["matched_exposure_groups"] = len(matched)
    (OUT / f"{args.output}.json").write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("One or more investigation checks failed; see the saved report.")
    print("All investigation checks passed.")


if __name__ == "__main__":
    main()
