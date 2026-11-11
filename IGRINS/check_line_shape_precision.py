"""Audit numerical convergence of paired exact line-shape injection tests.

Only the least-squares stopping tolerances change within each scoped call.
The scientific model, masks, priors, bounds and selected configuration do not.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

import fit_line_shapes as fitting
from check_line_shape_planet import run_water_suite, water_signal

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOLVER = dict(ftol=1e-12, xtol=1e-12, gtol=1e-8, max_nfev=500)


def tight_fit(context, config, target=None, initial=None):
    """Temporarily override optimizer precision, retaining the exact fitter."""
    original = fitting.least_squares

    def precise(*args, **kwargs):
        kwargs.update(SOLVER)
        return original(*args, **kwargs)

    fitting.least_squares = precise
    try:
        return fitting.fit_shapes(context, config, target=target, initial=initial)
    finally:
        fitting.least_squares = original


def null_metrics(context, response):
    """Measure no-signal drift against the weakest moving injection template."""
    signal = water_signal(context, dict(fwhm_kms=6., kp_kms=200., depth=1e-4))
    expected = context["project"](signal)
    result = {}
    for name, es, mask in [("training", slice(0, 5), context["allowed"]),
                           ("evaluation_score", slice(5, None), context["score"]),
                           ("evaluation_included", slice(5, None), context["allowed"])]:
        keep = np.broadcast_to(mask[:, None], response[:, es].shape)
        r, s = response[:, es][keep], expected[:, es][keep]
        result[name] = dict(rms_continuum=float(np.sqrt(np.mean(r*r))),
                            maximum_absolute=float(np.max(abs(r))),
                            projection_as_weak_signal_gain=float(s@r/(s@s)),
                            norm_relative_to_weak_signal=float(np.linalg.norm(r)/np.linalg.norm(s)))
    return result


def main():
    paths = [ROOT/"fit_line_shapes.py", ROOT/"check_line_shape_planet.py",
             OUT/"line_shape_fit.json", OUT/"line_shape_fit.npz",
             OUT/"line_shape_planet.json"]
    original_hashes = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    meta = json.loads((OUT/"line_shape_fit.json").read_text())
    previous = json.loads((OUT/"line_shape_planet.json").read_text())
    with np.load(OUT/"line_shape_fit.npz") as saved:
        arrays = {k:saved[k] for k in saved.files}
    context = fitting.build_context()
    selected = meta["selection"]["selected_case"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  selected_case=selected, solver=SOLVER,
                  method="Scoped optimizer tolerance override around the unmodified exact fit_shapes callback",
                  null_acceptance="Projected null norm below 0.001 times the depth=1e-4 moving-template norm in every scope",
                  settings={}, original_source_sha256=original_hashes)
    for name in dict.fromkeys([selected, "strength_control"]):
        config = meta["cases"][name]["config"]
        theta, diagnostic = tight_fit(context, config, initial=arrays[f"theta_{name}"].copy())
        baseline_history = [diagnostic]
        null_history = []
        for attempt in range(5):
            baseline_prediction = fitting.predict_components(context, theta)
            null_theta, null_diagnostic = tight_fit(context, config, initial=theta.copy())
            null_prediction = fitting.predict_components(context, null_theta)
            response = context["project"](baseline_prediction-null_prediction)
            null = dict(attempt=attempt, fit=null_diagnostic, metrics=null_metrics(context, response),
                        component_max_difference=float(np.max(abs(null_prediction-baseline_prediction))),
                        fitted_theta=null_theta.tolist())
            null["accepted"] = all(v["norm_relative_to_weak_signal"] < .001
                                   for v in null["metrics"].values())
            null_history.append(null)
            print("Null", name, attempt, null["accepted"], null["metrics"]["evaluation_score"], flush=True)
            if null["accepted"] or attempt == 4:
                break
            theta = null_theta
            baseline_history.append(null_diagnostic)
        setting = dict(config=config, baseline_theta=theta.tolist(),
                       baseline_fit_history=baseline_history, null_history=null_history,
                       null_precision_passed=bool(null_history[-1]["accepted"]),
                       original_component_max_drift=float(np.max(abs(
                           fitting.predict_components(context, theta)-arrays[f"component_{name}"]))))
        fit_callback = lambda target, initial: tight_fit(context, config, target=target, initial=initial)
        predict_callback = lambda values: fitting.predict_components(context, values)
        setting["water"] = run_water_suite(context, config, theta, fit_callback, predict_callback)
        for case, old in zip(setting["water"]["cases"], previous["settings"][name]["water"]["cases"], strict=True):
            for key in ["fwhm_kms", "kp_kms", "depth"]:
                assert case[key] == old[key]
            case["previous_default_precision"] = {scope: old[scope] for scope in
                                                   ["training", "evaluation_score", "evaluation_included"]}
        result["settings"][name] = setting
        (OUT/"line_shape_precision_checkpoint.json").write_text(json.dumps(result, indent=2)+"\n")
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest() == original_hashes[str(p.relative_to(ROOT))]
    result["source_sha256"] = dict(original_hashes, **{
        str(Path(__file__).relative_to(ROOT)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (OUT/"line_shape_precision.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Numerical precision audit for exact line-shape injections", "",
             "Model, data, priors, masks and bounds are unchanged; solver tolerances are tightened.",
             json.dumps(SOLVER), "",
             "Model Kp Depth Old_eval_gain(%) Tight_eval_gain(%) Tight_eval_correlation"]
    for name, setting in result["settings"].items():
        lines.append(f"{name}: null precision passed = {setting['null_precision_passed']}")
        lines.append("Final null metrics: "+json.dumps(setting["null_history"][-1]["metrics"]))
        for case in setting["water"]["cases"]:
            lines.append(f"{name} {case['kp_kms']:g} {case['depth']:g} "
                         f"{100*case['previous_default_precision']['evaluation_score']['incremental_gain']:.7f} "
                         f"{100*case['evaluation_score']['incremental_gain']:.7f} "
                         f"{case['evaluation_score']['response_correlation']:.9f}")
    lines += ["", "The original results and fit source files are preserved and their hashes rechecked.",
              "These remain conditional artificial-pattern tests, not full-pipeline planet preservation.", ""]
    (OUT/"LINE_SHAPE_PRECISION_REPORT.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
