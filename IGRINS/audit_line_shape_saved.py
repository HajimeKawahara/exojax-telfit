"""Audit saved line-shape predictions, priors, masks, and CV selection without opacity."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"


def main():
    metadata = json.loads((OUT/"line_shape_fit.json").read_text())
    selection = json.loads((OUT/"line_shape_selection.json").read_text())
    assert metadata["selection"] == selection
    for relative, digest in metadata["source_sha256"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    with np.load(OUT/"line_shape_fit.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    with np.load(OUT/"line_shape_cv.npz") as saved:
        cv_arrays = {key: saved[key] for key in saved.files}
    cv_prediction_path = OUT/"line_shape_cv_predictions.npz"
    cv_predictions = None
    if cv_prediction_path.exists():
        with np.load(cv_prediction_path) as saved:
            cv_predictions = {key: saved[key] for key in saved.files}
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c", "continuum"]:
            np.testing.assert_array_equal(arrays[key], saved[key])
        np.testing.assert_array_equal(arrays["component_adopted"], saved["component"])
        np.testing.assert_array_equal(arrays["theta_adopted"][arrays["line_parameter_indices"]],
                                      saved["parameters"][saved["line_parameter_indices"]])
        contrast = saved["line_contrast"]
    original_records = json.loads((OUT/"quadratic_strength_fit.json").read_text())["line_records"]
    uncertainty = np.array([r["uncertainty_upper"] for r in original_records])
    strength_priors = 1+9*contrast**2/(contrast**2+.005**2)
    line_indices = arrays["line_parameter_indices"]
    width_indices = np.array([r["width_parameter_index"] for r in metadata["shape_records"]], dtype=int)
    center_indices = np.array([r["center_parameter_index"] for r in metadata["shape_records"]], dtype=int)
    np.testing.assert_array_equal(np.sort(np.r_[width_indices, center_indices]), arrays["shape_parameter_indices"])
    assert len(np.unique(np.r_[line_indices, arrays["shape_parameter_indices"]])) == len(arrays["theta_adopted"])
    projection_error = 0.
    cases = {}
    for name, case in metadata["cases"].items():
        component = arrays[f"component_{name}"]
        residual = arrays["y_over_c"]-component
        for b, wave in enumerate(arrays["wavelength_um"]):
            x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
            polynomial = np.polynomial.chebyshev.chebvander(x, 5)
            mask = arrays["included_mask"][b]
            coefficients = np.linalg.lstsq(polynomial[mask], residual[b][:, mask].T, rcond=None)[0]
            residual[b] -= (polynomial @ coefficients).T
        difference = float(np.max(abs(residual-arrays[f"residual_{name}"])))
        projection_error = max(projection_error, difference)
        np.testing.assert_allclose(residual, arrays[f"residual_{name}"], atol=2e-11, rtol=0)
        metrics = {}
        for split, epochs, mask in [("training", slice(0, 5), arrays["included_mask"]),
                                    ("evaluation_score", slice(5, None), arrays["score_mask"]),
                                    ("evaluation_included", slice(5, None), arrays["included_mask"])]:
            rows = residual[:, epochs]
            keep = np.broadcast_to(mask[:, None], rows.shape)
            rms = float(100*np.sqrt(np.mean(rows[keep]**2)))
            fixed = float(100*np.sqrt(np.mean((rows*arrays["continuum"][:, epochs])[keep]**2)))
            assert int(keep.sum()) == case["metrics"][split]["n_samples"]
            np.testing.assert_allclose([rms, fixed], [case["metrics"][split]["rms_percent"],
                                                   case["metrics"][split]["fixed_rms_percent"]], atol=2e-11, rtol=0)
            for b, order in enumerate(arrays["orders"]):
                order_rms = float(100*np.sqrt(np.mean(rows[b][keep[b]]**2)))
                np.testing.assert_allclose(order_rms, case["metrics"][split]["per_order"][str(order)], atol=2e-11, rtol=0)
            common = rows.mean(axis=1)
            common_rms = float(100*np.sqrt(np.mean(common[mask]**2)))
            temporal_rms = float(100*np.sqrt(np.mean((rows-common[:, None])[keep]**2)))
            np.testing.assert_allclose(rms*rms, common_rms*common_rms+temporal_rms*temporal_rms, atol=1e-12, rtol=0)
            metrics[split] = dict(rms_percent=rms, fixed_rms_percent=fixed, n_samples=int(keep.sum()),
                                  common_rms_percent=common_rms, temporal_rms_percent=temporal_rms,
                                  sse=float(np.sum(rows[keep]**2)))
        theta = arrays[f"theta_{name}"]
        factor = 1+uncertainty*theta[line_indices]
        assert factor.min() >= -1e-10
        assert np.max(abs(theta[width_indices])) <= np.log(2.)+1e-10
        assert np.max(abs(theta[center_indices])) <= 1.+1e-10
        mode = case["config"]["mode"]
        if mode not in ["width", "joint"]:
            np.testing.assert_array_equal(theta[width_indices], 0.)
        if mode not in ["center", "joint"]:
            np.testing.assert_array_equal(theta[center_indices], 0.)
        for b, order in enumerate(arrays["orders"]):
            diagnostic = next(d for d in case["fit"] if d["order"] == int(order))
            assert diagnostic["success"]
            assert diagnostic["training_indices"] == list(range(5))
            data_objective = float(np.sum((residual[b, :5][:, arrays["included_mask"][b]]/.0045)**2))
            np.testing.assert_allclose(data_objective, diagnostic["data_objective"], atol=2e-7, rtol=1e-11)
            line_mask = np.array([record["order"] == int(order) for record in original_records])
            prior_objective = float(np.sum((theta[line_indices[line_mask]]/strength_priors[line_mask])**2))
            multiplier = case["config"]["prior_multiplier"]
            for record in metadata["shape_records"]:
                if record["order"] == int(order):
                    prior_objective += (theta[record["width_parameter_index"]]/(record["log_width_prior"]*multiplier))**2
                    prior_objective += (theta[record["center_parameter_index"]]/(record["center_prior_kms"]*multiplier))**2
            np.testing.assert_allclose(prior_objective, diagnostic["prior_objective"], atol=2e-7, rtol=1e-11)
        cases[name] = dict(metrics=metrics, minimum_intensity_factor=float(factor.min()),
                           width_factor_range=np.exp([theta[width_indices].min(), theta[width_indices].max()]).tolist(),
                           center_kms_range=[float(theta[center_indices].min()), float(theta[center_indices].max())],
                           projection_max_error=difference)
    cv = {}
    for name, item in selection["cross_validation"].items():
        assert item["config"] == metadata["cases"][name]["config"]
        assert len(item["folds"]) == 5
        independent_folds = []
        for held, fold in enumerate(item["folds"]):
            assert fold["heldout_exposure"] == int(arrays["exposures"][held])
            assert np.isfinite(cv_arrays[f"theta_{name}_fold{held}"]).all()
            for diagnostic in fold["fit"]:
                assert diagnostic["success"]
                assert diagnostic["training_indices"] == [i for i in range(5) if i != held]
            if cv_predictions is not None:
                saved_prediction = cv_predictions[f"component_{name}_fold{held}"]
                prediction = saved_prediction[:, held] if saved_prediction.ndim == 3 else saved_prediction
                assert prediction.shape == arrays["wavelength_um"].shape
                held_residual = arrays["y_over_c"][:, held]-prediction
                for b, wave in enumerate(arrays["wavelength_um"]):
                    coordinate = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
                    polynomial = np.polynomial.chebyshev.chebvander(coordinate, 5)
                    mask = arrays["included_mask"][b]
                    coefficients = np.linalg.lstsq(polynomial[mask], held_residual[b, mask], rcond=None)[0]
                    held_residual[b] -= polynomial @ coefficients
                independent_sse = float(np.sum(held_residual[arrays["included_mask"]]**2))
                np.testing.assert_allclose(independent_sse, fold["sse"], atol=2e-12, rtol=0)
                independent_folds.append(dict(heldout_exposure=fold["heldout_exposure"],
                                               independent_sse=independent_sse))
        summed = sum(fold["sse"] for fold in item["folds"])
        np.testing.assert_allclose(summed, item["sse"], atol=1e-13, rtol=0)
        rms = float(100*np.sqrt(summed/(5*arrays["included_mask"].sum())))
        np.testing.assert_allclose(rms, item["rms_percent"], atol=1e-12, rtol=0)
        cv[name] = dict(summed_fold_sse=summed, recomputed_rms_percent=rms,
                        independent_prediction_folds=independent_folds)
    ranking = sorted(cv, key=lambda name: cv[name]["summed_fold_sse"])
    assert ranking == selection["ranking"]
    assert ranking[0] == selection["selected_case"]
    paths = [Path(__file__), ROOT/"fit_line_shapes.py", OUT/"line_shape_fit.json", OUT/"line_shape_fit.npz",
             OUT/"line_shape_cv.npz", OUT/"line_shape_selection.json", OUT/"quadratic_strength_fit.json"]
    if cv_predictions is not None:
        paths.append(cv_prediction_path)
    cv_limit = ("Saved fold predictions are independently Q5-projected and their held-fold SSE reconstructed; all fold training-index exclusions and the final ranking are verified."
                if cv_predictions is not None else
                "CV fold SSE arithmetic, fitting-index exclusions and ranking are verified; individual held-fold predictions are not yet available for independent scoring.")
    result = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                  selected_case=ranking[0], independent_projection_max_error=projection_error,
                  input_source_hashes_verified=True, cases=cases, selection_arithmetic=cv,
                  independent_cv_fold_predictions_verified=cv_predictions is not None,
                  limits=["No opacity or optimization rerun; saved final predictions are independently projected and scored.",
                          "Saved training data/prior objectives and parameter bounds are independently reconstructed.",
                          cv_limit,
                          "A successful numerical audit does not identify the physical cause or guarantee planet-signal retention."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (OUT/"line_shape_saved_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=True, selected_case=ranking[0], independent_projection_max_error=projection_error,
                         evaluation_rms={name: case["metrics"]["evaluation_score"]["rms_percent"] for name, case in cases.items()}), indent=2))


if __name__ == "__main__":
    main()
