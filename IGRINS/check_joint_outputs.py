"""Validate saved joint fits and their conditional-validation partition."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from atmosphere_profile import load_profile

ROOT = Path(__file__).resolve().parent
TAGS = ["joint_nostar_slab", "joint_star_slab", "joint_star_profile"]


def main():
    missing = [str(ROOT/"results"/(tag+suffix))
               for tag in TAGS for suffix in [".json", ".npz"]
               if not (ROOT/"results"/(tag+suffix)).exists()]
    if missing:
        raise FileNotFoundError("Required fits are not complete: " + ", ".join(missing))
    with (ROOT/"data/cube14v3.pic").open("rb") as stream:
        wavelength, flux = pickle.load(stream)
    reference = None
    summary = {}
    for tag in TAGS:
        meta = json.loads((ROOT/"results"/(tag+".json")).read_text())
        with np.load(ROOT/"results"/(tag+".npz")) as archive:
            data = {key: archive[key] for key in archive.files}
        for key, value in data.items():
            assert np.all(np.isfinite(value)), (tag, "Nonfinite array", key)
        for key in ["training_fit", "heldout_calibration_fit"]:
            assert meta[key]["success"], (tag, key, meta[key]["message"])
            assert np.isfinite(meta[key]["jacobian_relative_error"])
            assert meta[key]["jacobian_relative_error"] < 1e-3, (tag, key)
        orders = data["orders"].tolist()
        exposures = data["exposures"].tolist()
        assert orders == meta["orders"]
        assert exposures == meta["train_exposures"] + [meta["heldout_exposure"]]
        assert len(exposures) == len(set(exposures)), tag
        lo, hi = meta["pixels"]
        shape = (len(orders), len(exposures), hi-lo)
        for key in ["observed", "model", "continuum", "component", "residual_continuum",
                    "train_mask", "calibration_mask", "score_mask"]:
            assert data[key].shape == shape, (tag, key, data[key].shape)
        assert np.all(data["continuum"] > 0), tag
        train, calibration, score = [data[key] for key in
                                     ["train_mask", "calibration_mask", "score_mask"]]
        assert not np.any(train & calibration) and not np.any(train & score)
        assert not np.any(calibration & score)
        assert np.all(train[:, :-1]) and not np.any(train[:, -1])
        assert not np.any(calibration[:, :-1]) and not np.any(score[:, :-1])
        assert np.any(calibration[:, -1]) and np.any(score[:, -1])
        for b, order in enumerate(orders):
            cal_pixels = np.flatnonzero(calibration[b, -1])
            score_pixels = np.flatnonzero(score[b, -1])
            assert np.min(np.abs(cal_pixels[:, None]-score_pixels[None, :])) >= 10
            np.testing.assert_array_equal(data["wavelength_um"][b], wavelength[order, lo:hi][::-1])
            raw = flux[order, exposures, lo:hi][:, ::-1]
            scales = np.percentile(raw, 95, axis=1)
            scales[-1] = np.percentile(raw[-1, calibration[b, -1]], 95)
            np.testing.assert_allclose(data["observed"][b], raw/scales[:, None], rtol=1e-13)
        np.testing.assert_allclose(data["model"], data["component"]*data["continuum"], rtol=1e-12)
        np.testing.assert_allclose(data["residual_continuum"],
                                   (data["observed"]-data["model"])/data["continuum"], rtol=1e-12)
        for key, mask in [("training_metrics", train), ("heldout_score_metrics", score)]:
            assert meta[key]["n_pixels"] == int(mask.sum())
            np.testing.assert_allclose(meta[key]["rms"],
                                       np.sqrt(np.mean(data["residual_continuum"][mask]**2)), rtol=1e-12)
        if reference is None:
            reference = data
        else:
            for key in ["orders", "exposures", "wavelength_um", "observed", "train_mask",
                        "calibration_mask", "score_mask"]:
                np.testing.assert_array_equal(data[key], reference[key], err_msg=f"{tag}: comparison changed {key}")
        p = load_profile(meta["layers"])
        np.testing.assert_allclose(np.sum(p["delta_pressure_bar"]),
                                   p["pressure_boundaries_bar"][-1]-p["pressure_boundaries_bar"][0], rtol=1e-14)
        column_checks = {}
        for species, fitted in meta["vertical_columns_cm-2"].items():
            reference_column = np.sum(p["number_column_cm2"]*p["vmr"][species])
            np.testing.assert_allclose(reference_column,
                meta["profile_metadata"]["reference_vertical_columns_cm2"][species], rtol=1e-12)
            parameters = meta["parameters"]
            factors = [10**parameters[f"log10_water_e{exposure}" if species == "H2O"
                                     else f"log10_scale_{species}"] for exposure in exposures]
            np.testing.assert_allclose(fitted, reference_column*np.array(factors), rtol=1e-12)
            exact_cache_matches = 0
            for path in (ROOT/"cache").glob(f"tau_{species}_*.npz"):
                with np.load(path) as saved:
                    column = saved["column"]
                expected = p["number_column_cm2"]*p["vmr"][species]
                if column.shape == expected.shape and np.allclose(column, expected, rtol=1e-12, atol=0):
                    exact_cache_matches += 1
            assert exact_cache_matches > 0, (tag, species, "No cache matches current hydrostatic columns")
            column_checks[species] = {"reference_column_cm2": float(reference_column),
                                      "matching_current_cache_entries": exact_cache_matches}
        fixed_residual = data["observed"]-data["model"]
        summary[tag] = {
            "training_rms_fixed_scale": float(np.sqrt(np.mean(fixed_residual[train]**2))),
            "heldout_score_rms_fixed_scale": float(np.sqrt(np.mean(fixed_residual[score]**2))),
            "training_rms_continuum": meta["training_metrics"]["rms"],
            "heldout_score_rms_continuum": meta["heldout_score_metrics"]["rms"],
            "training_jacobian_relative_error": meta["training_fit"]["jacobian_relative_error"],
            "heldout_jacobian_relative_error": meta["heldout_calibration_fit"]["jacobian_relative_error"],
            "hydrostatic_column_checks": column_checks,
        }
    result = {"status": "All saved-output consistency checks passed", "fits": summary}
    (ROOT/"results/joint_output_checks.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
