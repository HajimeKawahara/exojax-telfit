"""Check fitted line shapes, a cold start, and saved cross-validation predictions."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fit_line_shapes import build_context, fit_shapes, predict_components
from fit_line_strength import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    context = build_context()
    report = json.loads((OUT/"line_shape_fit.json").read_text())
    selected = report["selection"]["selected_case"]
    config = report["cases"][selected]["config"]
    with np.load(OUT/"line_shape_fit.npz") as saved:
        expected_theta = saved[f"theta_{selected}"]
        expected_component = saved[f"component_{selected}"]
    pred = predict_components(context, expected_theta)
    np.testing.assert_allclose(pred, expected_component, atol=1e-11, rtol=0)
    predictions = {}
    with np.load(OUT/"line_shape_cv.npz") as saved:
        for name, cv in report["selection"]["cross_validation"].items():
            for held in range(5):
                component = predict_components(context, saved[f"theta_{name}_fold{held}"])
                predictions[f"component_{name}_fold{held}"] = component
    np.savez_compressed(OUT/"line_shape_cv_predictions.npz", **predictions)
    print("Saved 35 CV predictions", flush=True)
    # Starting at HITRAN strengths and zero shape corrections probes warm-start dependence.
    cold, cold_fit = fit_shapes(context, config, initial=np.zeros(context["n_parameters"]))
    cold_prediction = predict_components(context, cold)
    cold_metrics, _ = metrics(context, cold_prediction)
    target = context["y_over_c"].copy()
    target[:, 5:] += 10.
    # Compare identical initial vectors, changing evaluation targets only.
    ordinary, ordinary_fit = fit_shapes(context, config, initial=expected_theta)
    corrupted, corrupt_fit = fit_shapes(context, config, target=target, initial=expected_theta)
    np.testing.assert_array_equal(ordinary, corrupted)
    shape_checks = []
    for block in context["blocks"]:
        theta = expected_theta[block["shape_slice"]]
        n = len(block["records"])
        if n == len(theta):
            continue
        # Check shape-only directions at the fitted solution, including saturated lines.
        direction = np.zeros_like(theta)
        direction[n:] = np.random.default_rng(int(block["order"])).normal(size=len(theta)-n)
        direction /= np.linalg.norm(direction)
        step = 1e-4
        exact = block["evaluate_shape"](theta, True) @ direction
        finite = (block["evaluate_shape"](theta+step*direction)-block["evaluate_shape"](theta-step*direction))/(2*step)
        error = float(np.linalg.norm(exact-finite)/np.linalg.norm(finite))
        assert error < 2e-5, error
        shape_checks.append(dict(order=int(block["order"]), directional_derivative_relative_error=error))
    original_rms = report["cases"][selected]["metrics"]["evaluation_score"]["rms_percent"]
    cold_rms = cold_metrics["evaluation_score"]["rms_percent"]
    assert abs(cold_rms-original_rms) < 1e-5, (cold_rms, original_rms)
    result = dict(passed=True, selected_case=selected,
        reconstruction_max_error=float(np.max(abs(pred-expected_component))),
        cold_start=dict(fit=cold_fit, metrics=cold_metrics, evaluation_rms_difference_pp=cold_rms-original_rms,
                        max_component_difference=float(np.max(abs(cold_prediction-expected_component))),
                        max_parameter_difference=float(np.max(abs(cold-expected_theta)))),
        evaluation_corruption=dict(max_parameter_difference=float(np.max(abs(ordinary-corrupted))),
                                   normal_fit=ordinary_fit, corrupt_fit=corrupt_fit),
        fitted_shape_derivative_checks=shape_checks,
        cv_predictions="line_shape_cv_predictions.npz",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                       [Path(__file__), ROOT/"fit_line_shapes.py", OUT/"line_shape_fit.json",
                        OUT/"line_shape_fit.npz", OUT/"line_shape_cv.npz", OUT/"line_shape_cv_predictions.npz"]})
    (OUT/"line_shape_solution_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Validation passed", cold_rms, "cold difference", cold_rms-original_rms, flush=True)


if __name__ == "__main__":
    main()
