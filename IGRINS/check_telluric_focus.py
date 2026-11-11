"""Independently check the fixed-mask telluric comparison outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tags", nargs="+")
    args = parser.parse_args()
    plan = json.loads((RESULTS / "telluric_focus_plan.json").read_text())
    tags = args.tags or list(plan["variants"])
    mask_path = RESULTS / plan["mask"]
    mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    with np.load(mask_path) as saved:
        mask_data = {key: saved[key] for key in saved.files}
    source = json.loads((RESULTS / "joint_star_profile.json").read_text())
    fixed_names = ["stellar_rv_reference_kms", "stellar_extra_sigma_kms"]
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    report = {"passed": False, "mask_sha256": mask_hash, "variants": {}}
    reference = None
    for tag in tags:
        meta = json.loads((RESULTS / f"{tag}.json").read_text())
        with np.load(RESULTS / f"{tag}.npz") as saved:
            data = {key: saved[key] for key in saved.files}
        assert meta["pixel_mask"]["sha256"] == mask_hash
        assert meta["fixed_parameters"] == fixed_names
        assert meta["train_exposures"] == plan["train_exposures"]
        assert meta["train_exposures"][0] == source["train_exposures"][0] == 0
        assert meta["heldout_exposure"] == plan["fresh_conditional_validation_exposure"]
        for name in fixed_names:
            assert meta["parameters"][name] == source["parameters"][name]
        variant = plan["variants"][tag]
        assert meta["pressure_shift"] == variant["shift"]
        assert meta["ip"] == variant["ip"]
        assert meta["temperature_offset_K"] == variant["temperature_offset_K"]
        assert not meta["reference_self_broadening"]
        for key in ("training_fit", "heldout_calibration_fit"):
            assert meta[key]["success"], (tag, key, meta[key])
            assert meta[key]["jacobian_relative_error"] < 1e-3, (tag, key)
        lo, hi = meta["pixels"]
        n_pixel = hi - lo
        pixel = np.arange(n_pixel)
        cal_blocks = ((pixel // 80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
        score_blocks = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
        basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, n_pixel), 3)
        orders, exposures = data["orders"], data["exposures"]
        np.testing.assert_array_equal(exposures, meta["train_exposures"] + [meta["heldout_exposure"]])
        allowed = np.stack([mask_data["mask"][np.flatnonzero(mask_data["orders"] == o).item()]
                            for o in orders])
        np.testing.assert_array_equal(data["included_mask"], allowed)
        training = np.broadcast_to(allowed[:, None, :], data["observed"].shape).copy()
        training[:, -1] = False
        calibration = np.zeros_like(training)
        calibration[:, -1] = allowed & cal_blocks
        scoring = np.zeros_like(training)
        scoring[:, -1] = allowed & score_blocks
        for key, expected in (("train_mask", training), ("calibration_mask", calibration),
                              ("score_mask", scoring)):
            np.testing.assert_array_equal(data[key], expected)
        assert not np.any(training & calibration | training & scoring | calibration & scoring)
        for key in ("observed", "model", "component", "continuum", "residual_continuum"):
            assert np.all(np.isfinite(data[key])), (tag, key)
        assert np.all(data["continuum"] > 0)
        np.testing.assert_allclose(data["model"], data["component"] * data["continuum"], atol=1e-13)
        np.testing.assert_allclose(data["residual_continuum"],
                                   (data["observed"] - data["model"]) / data["continuum"], atol=1e-13)
        max_continuum_error = 0.
        max_perturbation_change = 0.
        for b, order in enumerate(orders):
            np.testing.assert_allclose(data["wavelength_um"][b], wavelength[order, lo:hi][::-1],
                                       rtol=0, atol=1e-12)
            flux = cube[order, exposures, lo:hi][:, ::-1]
            scales = np.percentile(flux[:, allowed[b]], 95, axis=1)
            scales[-1] = np.percentile(flux[-1, calibration[b, -1]], 95)
            np.testing.assert_array_equal(data["observed"][b], flux / scales[:, None])
            for e in range(len(exposures)):
                weight = (training | calibration)[b, e].astype(float)
                design = basis * data["component"][b, e, :, None]
                normal = design.T @ (weight[:, None] * design) + 1e-12 * np.eye(4)
                observed = data["observed"][b, e]
                reconstructed = basis @ np.linalg.solve(normal, design.T @ (weight * observed))
                error = np.max(np.abs(reconstructed - data["continuum"][b, e]))
                max_continuum_error = max(max_continuum_error, float(error))
                np.testing.assert_allclose(reconstructed, data["continuum"][b, e], rtol=0, atol=1e-10)
                # Excluded data include all held-out scoring pixels. Large changes
                # there must not change the independently reconstructed continuum.
                perturbed = observed + (1 - weight) * (10 + pixel / n_pixel)
                alternate = basis @ np.linalg.solve(normal, design.T @ (weight * perturbed))
                change = np.max(np.abs(alternate - reconstructed))
                max_perturbation_change = max(max_perturbation_change, float(change))
                np.testing.assert_array_equal(alternate, reconstructed)
        for key, residual, mask in (
            ("training_metrics", data["residual_continuum"], training),
            ("heldout_score_metrics", data["residual_continuum"], scoring),
            ("training_fixed_scale_metrics", data["observed"] - data["model"], training),
            ("heldout_fixed_scale_metrics", data["observed"] - data["model"], scoring),
        ):
            assert meta[key]["n_pixels"] == np.count_nonzero(mask)
            np.testing.assert_allclose(meta[key]["rms"], np.sqrt(np.mean(residual[mask] ** 2)), rtol=1e-12)
        if reference is None:
            reference = data
        else:
            for key in ("observed", "orders", "exposures", "wavelength_um", "included_mask",
                        "train_mask", "calibration_mask", "score_mask"):
                np.testing.assert_array_equal(data[key], reference[key])
        report["variants"][tag] = {
            "continuum_reconstruction_max_error": max_continuum_error,
            "excluded_flux_perturbation_max_change": max_perturbation_change,
            "training_pixels": int(np.count_nonzero(training)),
            "calibration_pixels": int(np.count_nonzero(calibration)),
            "scoring_pixels": int(np.count_nonzero(scoring)),
            "training_fixed_scale_rms": meta["training_fixed_scale_metrics"]["rms"],
            "heldout_fixed_scale_rms": meta["heldout_fixed_scale_metrics"]["rms"],
            "bounds_hit": meta["bounds_hit"],
        }
    report["passed"] = True
    output = RESULTS / "telluric_focus_checks.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
