"""Check two prespecified conservative alternatives at tight solver precision."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

import fit_line_shapes as fitting
from check_line_shape_planet import annotate_parameters, water_signal
from check_line_shape_precision import SOLVER, null_metrics, tight_fit
from check_strength_planet import projection_metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def moving_suite(context, config, baseline):
    """Refit only the positive 1e-3 and 1e-4 moving water-pattern signals."""
    observed, project = context["y_over_c"], context["project"]
    baseline_prediction = fitting.predict_components(context, baseline)
    baseline_residual = project(observed-baseline_prediction)
    result = dict(config=config, cases=[])
    for depth in [1e-3, 1e-4]:
        case = dict(fwhm_kms=6., kp_kms=200., depth=depth)
        signal = water_signal(context, case)
        target = observed.copy()
        target[:, :5] += signal[:, :5]
        np.testing.assert_array_equal(target[:, 5:], observed[:, 5:])
        theta, diagnostic = tight_fit(context, config, target=target, initial=baseline.copy())
        prediction = fitting.predict_components(context, theta)
        response = project(observed+signal-prediction)-baseline_residual
        expected = project(signal)
        exact = expected-project(prediction-baseline_prediction)
        np.testing.assert_allclose(response, exact, rtol=1e-9, atol=1e-13)
        record = dict(**case, fitted_theta=theta.tolist(), fit_diagnostics=diagnostic,
                      injected_minus_uninjected_identity_max_error=float(np.max(abs(response-exact))))
        for name, es, mask in [("training", slice(0, 5), context["allowed"]),
                               ("evaluation_score", slice(5, None), context["score"]),
                               ("evaluation_included", slice(5, None), context["allowed"])]:
            record[name] = projection_metrics(signal[:, es], expected[:, es], response[:, es], mask)
        result["cases"].append(record)
        print(config["name"], depth, record["evaluation_score"], flush=True)
    annotate_parameters(result, context, baseline)
    return result


def main():
    paths = [ROOT/"fit_line_shapes.py", ROOT/"check_line_shape_planet.py",
             ROOT/"check_line_shape_precision.py", OUT/"line_shape_fit.json",
             OUT/"line_shape_fit.npz", OUT/"line_shape_precision.json"]
    hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    meta = json.loads((OUT/"line_shape_fit.json").read_text())
    precise = json.loads((OUT/"line_shape_precision.json").read_text())
    with np.load(OUT/"line_shape_fit.npz") as source:
        arrays = {k:source[k] for k in source.files}
    context = fitting.build_context()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), solver=SOLVER,
                  unchanged_cv_selected_case=meta["selection"]["selected_case"],
                  purpose="Fixed conservative sensitivity cases; no reselection from evaluation or injection metrics",
                  settings={})
    for name in ["width_prior1", "joint_prior1"]:
        config = meta["cases"][name]["config"]
        baseline, fit = tight_fit(context, config, initial=arrays[f"theta_{name}"].copy())
        history, null_history = [fit], []
        for attempt in range(5):
            predicted = fitting.predict_components(context, baseline)
            null_theta, null_fit = tight_fit(context, config, initial=baseline.copy())
            null_predicted = fitting.predict_components(context, null_theta)
            metrics = null_metrics(context, context["project"](predicted-null_predicted))
            accepted = all(m["norm_relative_to_weak_signal"] < .001 for m in metrics.values())
            null_history.append(dict(attempt=attempt, accepted=accepted, metrics=metrics,
                                     fit=null_fit, fitted_theta=null_theta.tolist()))
            print("Null", name, attempt, accepted, metrics["evaluation_score"], flush=True)
            if accepted or attempt == 4:
                break
            baseline = null_theta
            history.append(null_fit)
        suite = moving_suite(context, config, baseline)
        control_cases = precise["settings"]["strength_control"]["water"]["cases"]
        selected_cases = precise["settings"][precise["selected_case"]]["water"]["cases"]
        for case in suite["cases"]:
            for label, cases in [("tight_strength_control", control_cases),
                                 ("tight_cv_selected_case", selected_cases)]:
                matched = next(c for c in cases if all(c[k] == case[k]
                               for k in ["fwhm_kms", "kp_kms", "depth"]))
                case[label] = {k:matched[k] for k in
                               ["training", "evaluation_score", "evaluation_included"]}
        result["settings"][name] = dict(config=config, baseline_theta=baseline.tolist(),
            baseline_fit_history=history, null_history=null_history,
            null_precision_passed=accepted, water=suite)
        (OUT/"line_shape_precision_alternatives_checkpoint.json").write_text(json.dumps(result, indent=2)+"\n")
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == hashes[str(p.relative_to(ROOT))]
    result["source_sha256"] = dict(hashes, **{
        str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/"line_shape_precision_alternatives.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Tight-precision moving-water tests for conservative alternatives", "",
             "The original cross-validation selection is unchanged.",
             "These fixed sensitivity cases were not chosen using evaluation spectra or injected signals.", "",
             "Model Depth Train_gain(%) Eval_gain(%) Eval_correlation"]
    for name, setting in result["settings"].items():
        lines.append(f"{name}: null precision passed = {setting['null_precision_passed']}")
        for case in setting["water"]["cases"]:
            lines.append(f"{name} {case['depth']:g} {100*case['training']['incremental_gain']:.7f} "
                         f"{100*case['evaluation_score']['incremental_gain']:.7f} "
                         f"{case['evaluation_score']['response_correlation']:.9f}")
    lines += ["", "All gains are relative to the same degree-five filter alone.",
              "The water pattern is artificial and this does not measure full-pipeline planet preservation.", ""]
    (OUT/"LINE_SHAPE_PRECISION_ALTERNATIVES_REPORT.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
