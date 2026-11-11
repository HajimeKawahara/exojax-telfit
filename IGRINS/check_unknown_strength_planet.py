"""Compare moving-line retention with and without unknown-strength parameters."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import numpy as np

from check_smooth_signal import forest_signal
from check_strength_planet import projection_metrics
from fit_unknown_strength import build_context, fit_case, predict_components

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
CASES = [dict(fwhm_kms=6., kp_kms=200., depth=1e-3),
         dict(fwhm_kms=10., kp_kms=100., depth=1e-3),
         dict(fwhm_kms=10., kp_kms=0., depth=1e-3),
         dict(fwhm_kms=6., kp_kms=200., depth=1e-4)]


def main():
    started = time.monotonic()
    source = OUT / "unknown_strength_fit.npz"
    if not source.exists():
        raise FileNotFoundError("Run fit_unknown_strength.py before this check.")
    adopted_path = OUT / "adopted_telluric_fit.npz"
    adopted_hash = hashlib.sha256(adopted_path.read_bytes()).hexdigest()
    context = build_context()
    with np.load(source) as saved:
        theta = {name: saved[f"theta_{name}"].copy() for name in ["known_control", "joint30"]}
        stored_components = {name: saved[f"component_{name}"].copy() for name in theta}
        np.testing.assert_array_equal(saved["y_over_c"], context["y_over_c"])
        np.testing.assert_array_equal(saved["included_mask"], context["allowed"])
        np.testing.assert_array_equal(saved["score_mask"], context["score"])
    observed = context["y_over_c"]
    baseline_prediction = {name: predict_components(context, value) for name, value in theta.items()}
    for name, value in baseline_prediction.items():
        np.testing.assert_allclose(value, stored_components[name], atol=1e-10, rtol=0)
    baseline_residual = {name: context["project"](observed-value)
                         for name, value in baseline_prediction.items()}
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][context["exposures"]]
    wave = context["wave"]
    velocity = C_KMS*np.log(wave/np.exp(np.mean(np.log(wave), axis=1))[:, None])
    arrays = dict(orders=context["orders"], exposures=context["exposures"],
                  wavelength_um=wave, included_mask=context["allowed"],
                  score_mask=context["score"],
                  common_signal_attenuation=context["reference_component"])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
                  training_frames=context["exposures"][:5].tolist(),
                  evaluation_frames=context["exposures"][5:].tolist(),
                  primary_fractional_prior_scale=.3,
                  paired_models={"known_control": "Only existing 269 strengths free; new lines fixed.",
                                 "joint30": "Existing 269 and eight added strengths free."},
                  signal="Identical observed-resolution Gaussian forests for both fits, using the original context reference_component for attenuation and fixed per-order seeds.",
                  definition="Gain=dot(Q signal, injected-minus-uninjected residual)/dot(Q signal,Q signal). Paired difference=expanded gain minus known-only gain.",
                  limitations=["Artificial observed-resolution forests, not physical molecular spectra injected before the IP.",
                               "Kp values are hypothetical controls, not measured orbital parameters.",
                               "Only training targets receive injections during fitting; the same full signal is added to held-out observations for response evaluation.",
                               "Conditional on prior fitted atmospheric, stellar, instrumental and extraction parameters; those losses are not measured here.",
                               "Responses above unity can arise from transferred fit-response projection and do not represent extra information.",
                               "Four cases are diagnostic controls, not exhaustive planet-signal validation."], cases=[])
    for index, case in enumerate(CASES):
        case_started = time.monotonic()
        signal = np.array([
            forest_signal(v, phase, part, case["fwhm_kms"], case["kp_kms"],
                          seed=20261111+int(order))*case["depth"]/1e-3
            for v, part, order in zip(velocity, context["reference_component"], context["orders"])])
        expected = context["project"](signal)
        target = observed.copy()
        target[:, :5] += signal[:, :5]
        np.testing.assert_array_equal(target[:, 5:], observed[:, 5:])
        row = dict(**case, models={})
        arrays[f"signal_{index}"] = signal
        arrays[f"filtered_signal_{index}"] = expected
        for name, scale in [("known_control", 0.), ("joint30", .3)]:
            injected_theta, diagnostics = fit_case(context, scale=scale, target=target, initial=theta[name])
            prediction = predict_components(context, injected_theta)
            response = context["project"](observed+signal-prediction)-baseline_residual[name]
            exact = expected-context["project"](prediction-baseline_prediction[name])
            identity_error = float(np.max(np.abs(response-exact)))
            np.testing.assert_allclose(response, exact, atol=1e-12, rtol=1e-9)
            report = dict(
                training=projection_metrics(signal[:, :5], expected[:, :5], response[:, :5], context["allowed"]),
                evaluation_score=projection_metrics(signal[:, 5:], expected[:, 5:], response[:, 5:], context["score"]),
                evaluation_included=projection_metrics(signal[:, 5:], expected[:, 5:], response[:, 5:], context["allowed"]),
                fit_diagnostics=diagnostics,
                response_identity_max_error=identity_error,
                parameter_change_max=float(np.max(np.abs(injected_theta-theta[name]))),
                per_order={str(order): dict(
                    training=projection_metrics(signal[b:b+1, :5], expected[b:b+1, :5], response[b:b+1, :5], context["allowed"][b:b+1]),
                    evaluation_score=projection_metrics(signal[b:b+1, 5:], expected[b:b+1, 5:], response[b:b+1, 5:], context["score"][b:b+1]))
                    for b, order in enumerate(context["orders"])})
            row["models"][name] = report
            arrays[f"theta_{name}_{index}"] = injected_theta
            arrays[f"response_{name}_{index}"] = response
            arrays[f"prediction_{name}_{index}"] = prediction
        known, expanded = row["models"]["known_control"], row["models"]["joint30"]
        row["paired_gain_difference"] = dict(
            training=float(expanded["training"]["incremental_gain"]-known["training"]["incremental_gain"]),
            evaluation_score=float(expanded["evaluation_score"]["incremental_gain"]-known["evaluation_score"]["incremental_gain"]),
            per_order={str(order): float(expanded["per_order"][str(order)]["evaluation_score"]["incremental_gain"]
                                        -known["per_order"][str(order)]["evaluation_score"]["incremental_gain"])
                       for order in context["orders"]})
        row["seconds"] = time.monotonic()-case_started
        result["cases"].append(row)
        print(json.dumps(dict(case=case,
                              known_pooled_gain=known["evaluation_score"]["incremental_gain"],
                              expanded_pooled_gain=expanded["evaluation_score"]["incremental_gain"],
                              known_order13_gain=known["per_order"]["13"]["evaluation_score"]["incremental_gain"],
                              expanded_order13_gain=expanded["per_order"]["13"]["evaluation_score"]["incremental_gain"],
                              difference=row["paired_gain_difference"])), flush=True)
    paths = [Path(__file__), ROOT/"fit_unknown_strength.py", ROOT/"fit_line_strength.py",
             ROOT/"check_strength_planet.py", ROOT/"check_smooth_signal.py",
             ROOT/"data/20201214_info.csv", source, OUT/"unknown_strength_fit.json", adopted_path]
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result["seconds"] = time.monotonic()-started
    np.savez_compressed(OUT/"unknown_strength_planet.npz", **arrays)
    (OUT/"unknown_strength_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Paired planet-line response: unknown-strength extension", "",
             "Gain percentages relative to the same degree-five projection; four evaluation frames only.",
             "Delta is expanded minus known-only gain in percentage points, not a residual RMS change.", "",
             "FWHM | Kp | Depth | Known pooled | Expanded pooled | Known order13 | Expanded order13 | Delta order13"]
    for row in result["cases"]:
        k, e = row["models"]["known_control"], row["models"]["joint30"]
        lines.append(f"{row['fwhm_kms']:g} | {row['kp_kms']:g} | {row['depth']:g} | "
                     f"{100*k['evaluation_score']['incremental_gain']:.6f}% | "
                     f"{100*e['evaluation_score']['incremental_gain']:.6f}% | "
                     f"{100*k['per_order']['13']['evaluation_score']['incremental_gain']:.6f}% | "
                     f"{100*e['per_order']['13']['evaluation_score']['incremental_gain']:.6f}% | "
                     f"{100*row['paired_gain_difference']['per_order']['13']:+.6f}")
    lines += ["", "Limits:"]+result["limitations"]
    (OUT/"UNKNOWN_STRENGTH_PLANET_REPORT.txt").write_text("\n".join(lines)+"\n")
    assert hashlib.sha256(adopted_path.read_bytes()).hexdigest() == adopted_hash
    print("Paired injection check complete", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
