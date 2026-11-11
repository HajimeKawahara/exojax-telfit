"""Audit a stellar-amplitude fit and measure paired artificial-planet response."""

from __future__ import annotations

import argparse
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
from smooth_residuals import smoother_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
CASES = [dict(kind="forest", fwhm_kms=6., kp_kms=200., depth=1e-3),
         dict(kind="forest", fwhm_kms=10., kp_kms=100., depth=1e-3),
         dict(kind="forest", fwhm_kms=10., kp_kms=0., depth=1e-3),
         dict(kind="forest", fwhm_kms=6., kp_kms=200., depth=1e-4),
         dict(kind="local_4819_92", fwhm_kms=6., kp_kms=200., depth=1e-3)]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_success(diagnostics):
    """Accept a per-order list or a nested fit-diagnostics mapping."""
    if isinstance(diagnostics, dict):
        if "success" in diagnostics and not diagnostics["success"]:
            raise RuntimeError(f"Fit did not converge: {diagnostics}")
        for value in diagnostics.values():
            if isinstance(value, (dict, list, tuple)):
                require_success(value)
    elif isinstance(diagnostics, (list, tuple)):
        for value in diagnostics:
            require_success(value)


def fit_pipeline(problem, target, baseline, *, flexible, max_nfev):
    q = np.asarray(baseline[0]).copy()
    z = np.asarray(baseline[1]).copy()
    diagnostics = {}
    if flexible:
        q, diagnostics["stellar"] = problem.fit_stellar(
            target=target, initial=q, max_nfev=max_nfev)
        require_success(diagnostics["stellar"])
    frozen_q = np.asarray(q).copy()
    z, diagnostics["telluric"] = problem.fit_telluric(
        q, target=target, initial=z, max_nfev=max_nfev)
    require_success(diagnostics["telluric"])
    np.testing.assert_array_equal(q, frozen_q)
    prediction = np.asarray(problem.predict(q, z))
    if not np.all(np.isfinite(prediction)) or np.min(prediction) < 0:
        raise ValueError("Non-finite or negative component prediction")
    return np.asarray(q), np.asarray(z), prediction, diagnostics


def make_signal(case, context, phase, velocity):
    reference = context["reference_component"]
    if case["kind"] == "forest":
        return np.array([
            forest_signal(v, phase, part, case["fwhm_kms"], case["kp_kms"],
                          seed=20261111+int(order))*case["depth"]/1e-3
            for v, part, order in zip(velocity, reference, context["orders"])])
    signal = np.zeros_like(reference)
    b = list(context["orders"]).index(13)
    wave_reference = np.exp(np.mean(np.log(context["wave"][b])))
    crossing_velocity = C_KMS*np.log((1e4/4819.920802)/wave_reference)
    shifts = case["kp_kms"]*np.sin(2*np.pi*phase)
    # The selected line crosses the local PHOENIX mismatch near mid-sequence.
    center = crossing_velocity-np.median(shifts)
    offset = velocity[b][None, :]-center-shifts[:, None]
    sigma = case["fwhm_kms"]/np.sqrt(8*np.log(2))
    signal[b] = -case["depth"]*reference[b]*np.exp(-.5*(offset/sigma)**2)
    return signal


def response_metrics(signal, expected, response, context):
    result = {}
    for name, frames, mask in [
            ("training", slice(0, 5), context["allowed"]),
            ("evaluation_score", slice(5, None), context["score"]),
            ("evaluation_included", slice(5, None), context["allowed"])]:
        result[name] = projection_metrics(
            signal[:, frames], expected[:, frames], response[:, frames], mask)
    result["per_order"] = {}
    for b, order in enumerate(context["orders"]):
        if not np.any(signal[b]):
            continue
        result["per_order"][str(order)] = {
            name: projection_metrics(signal[b:b+1, frames], expected[b:b+1, frames],
                                     response[b:b+1, frames], mask[b:b+1])
            for name, frames, mask in [
                ("training", slice(0, 5), context["allowed"]),
                ("evaluation_score", slice(5, None), context["score"])]}
    return result


def compare_baseline(context, prediction, reference):
    """Separate stored-fit RMS from the refined injection baseline."""
    keep = np.broadcast_to(context["score"][:, None], prediction[:, 5:].shape)
    residual = context["project"](context["y_over_c"]-prediction)[:, 5:]
    reference_residual = context["project"](context["y_over_c"]-reference)[:, 5:]
    delta = context["project"](prediction-reference)[:, 5:]
    return dict(evaluation_score_rms_percent=float(100*np.sqrt(np.mean(residual[keep]**2))),
                stored_evaluation_score_rms_percent=float(100*np.sqrt(np.mean(reference_residual[keep]**2))),
                prediction_change_score_rms_percent=float(100*np.sqrt(np.mean(delta[keep]**2))),
                prediction_change_all_pixels_max_percent=float(100*np.max(np.abs(prediction-reference))))


def write_report(result):
    lines = ["Paired planet-line response: shared stellar amplitudes", "",
             "Gain is relative to the original degree-five projection. Both uninjected pipelines were refit first.",
             "Only the four held-out frames enter the reported evaluation gain.", "",
             "Case | FWHM | Kp | Depth | PHOENIX gain | Stellar gain | Delta (percentage points)"]
    for row in result["cases"]:
        a, b = [row["models"][name]["evaluation_score"]["incremental_gain"] for name in ["phoenix", "stellar"]]
        lines.append(f"{row['kind']} | {row['fwhm_kms']:g} | {row['kp_kms']:g} | {row['depth']:g} | {100*a:.6f}% | {100*b:.6f}% | {100*(b-a):+.6f}")
        lines.append("  Order deltas: "+", ".join(f"{key}: {100*value:+.6f}" for key, value in row["paired_gain_difference"]["per_order"].items()))
    lines += ["", "Uninjected baselines:",
              "The same optimizer was rerun until two consecutive maximum component changes were below 1e-8."]
    for name, row in result["baseline"].items():
        comparison = row["comparison_to_stored_fit"]
        checks = row["null_refit_checks"]
        lines.append(f"{name}: {len(checks)} null refinements; stored RMS {comparison['stored_evaluation_score_rms_percent']:.10f}%; refined RMS {comparison['evaluation_score_rms_percent']:.10f}%.")
        lines.append(f"  Last two maximum component drifts: {[item['component_max_drift'] for item in checks[-2:]]}; final evaluation-score RMS drift: {checks[-1]['evaluation_score_rms_drift']:.6g} fractional flux.")
        lines.append(f"  Refined-minus-stored prediction: score RMS {comparison['prediction_change_score_rms_percent']:.8f}%; all-pixel maximum {comparison['prediction_change_all_pixels_max_percent']:.8f}%.")
    lines += ["", "Audit:", json.dumps(result["audit"], indent=2), "", "Limits:"]+result["limitations"]
    (OUT/"BLASE_PLANET_REPORT.txt").write_text("\n".join(lines)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=160)
    args = parser.parse_args()
    from fit_blase_stellar import Problem

    started = time.monotonic()
    source = OUT / "blase_stellar_fit.npz"
    adopted_path = OUT / "adopted_telluric_fit.npz"
    initial_adopted_hash = digest(adopted_path)
    problem = Problem()
    context = problem.context
    observed = np.asarray(context["y_over_c"])
    observed_copy = observed.copy()
    mask_copy = {key: np.asarray(context[key]).copy() for key in ["allowed", "score"]}
    calibration_copy = np.asarray(problem.calibration_mask).copy()
    np.testing.assert_array_equal(context["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    ones = np.ones(problem.n_stellar)
    np.testing.assert_array_equal(problem.initial_stellar, ones)
    identity_component = np.asarray(problem.predict(ones, problem.adopted_z))
    with np.load(source) as saved:
        for name, key in [("y_over_c", "y_over_c"), ("included_mask", "allowed"),
                          ("score_mask", "score"), ("wavelength_um", "wave")]:
            np.testing.assert_array_equal(saved[name], context[key])
        adopted_error = float(np.max(np.abs(identity_component-saved["component_adopted"])))
        np.testing.assert_allclose(identity_component, saved["component_adopted"], rtol=0, atol=1e-10)
        saved_q, saved_z = saved["q"].copy(), saved["z"].copy()
        stored_prediction = saved["component_refit"].copy()
    np.testing.assert_allclose(problem.predict(saved_q, saved_z), stored_prediction, rtol=0, atol=1e-10)
    for q in [ones, saved_q]:
        star = np.asarray(problem.stellar(q))
        if not np.all(np.isfinite(star)) or np.min(star) < 0:
            raise ValueError("Non-finite or negative stellar-only prediction")
    np.testing.assert_array_equal(problem.calibration_mask,
                                  np.min(problem.calibration_telluric[:, :5], axis=1) > .95)
    derivative_checks = []
    for block in problem.blocks:
        sl, free = block["stellar_slice"], block["free"]
        np.testing.assert_array_equal(saved_q[sl][~free], np.ones((~free).sum()))
        if not np.any(free):
            continue
        direction = np.random.default_rng(20261111+block["order"]).normal(size=free.size)
        direction[~free] = 0
        direction /= np.linalg.norm(direction)
        residual, jacobian, _ = block["calibration_functions"]
        local_q = ones[sl]
        z = problem.adopted_z[block["strength_slice"]]
        y = observed[block["index"], :5]
        step = 1e-4
        finite_difference = (np.asarray(residual(local_q+step*direction, z, y))
                             -np.asarray(residual(local_q-step*direction, z, y)))/(2*step)
        analytic = np.asarray(jacobian(local_q, z, y))@direction
        error = float(np.linalg.norm(finite_difference-analytic)/np.linalg.norm(analytic))
        assert error < 1e-5, error
        derivative_checks.append(dict(order=block["order"], relative_l2=error))
    operators = np.array([smoother_matrix(w, mask, dict(kind="polynomial", degree=5))
                          for w, mask in zip(context["wave"], context["allowed"])])
    independent = observed-np.einsum("bij,bej->bei", operators, observed)
    q5_error = float(np.max(np.abs(context["project"](observed)-independent)))
    np.testing.assert_allclose(context["project"](observed), independent, rtol=0, atol=1e-12)
    audit = dict(adopted_identity_max_error=adopted_error,
                 independent_q5_max_error=q5_error,
                 stellar_derivative_checks=derivative_checks,
                 unsupported_amplitudes_equal_one=True,
                 calibration_mask_recomputed_from_first_five=True,
                 adopted_sha256_before=initial_adopted_hash,
                 data_and_masks_unchanged=True, stellar_parameters_fixed_in_stage2=True)
    baseline = {}
    baseline_report = {}
    arrays = dict(orders=context["orders"], exposures=context["exposures"],
                  wavelength_um=context["wave"], included_mask=context["allowed"],
                  score_mask=context["score"], calibration_mask=calibration_copy,
                  common_signal_attenuation=context["reference_component"])
    for name, flexible, initial in [
            ("phoenix", False, (ones, problem.adopted_z)),
            ("stellar", True, (saved_q, saved_z))]:
        q, z, prediction, diagnostics = fit_pipeline(
            problem, observed, initial, flexible=flexible, max_nfev=args.max_nfev)
        refinements, stable_count = [], 0
        for iteration in range(10):
            next_q, next_z, next_prediction, next_diagnostics = fit_pipeline(
                problem, observed, (q, z), flexible=flexible, max_nfev=args.max_nfev)
            drift = float(np.max(np.abs(next_prediction-prediction)))
            filtered_drift = context["project"](next_prediction-prediction)
            evaluation_keep = np.broadcast_to(context["score"][:, None], filtered_drift[:, 5:].shape)
            per_order_drift = {}
            for b, order in enumerate(context["orders"]):
                values = filtered_drift[b, 5:][:, context["score"][b]]
                per_order_drift[str(order)] = dict(
                    evaluation_score_rms=float(np.sqrt(np.mean(values**2))),
                    evaluation_score_max=float(np.max(np.abs(values))),
                    q_change_max=float(np.max(np.abs(next_q[problem.blocks[b]["stellar_slice"]]
                                                      -q[problem.blocks[b]["stellar_slice"]]))))
            refinements.append(dict(iteration=iteration+1, component_max_drift=drift,
                filtered_rms_drift=float(np.sqrt(np.mean(filtered_drift**2))),
                evaluation_score_rms_drift=float(np.sqrt(np.mean(filtered_drift[:, 5:][evaluation_keep]**2))),
                per_order=per_order_drift,
                fit=next_diagnostics))
            print("Null refit", name, json.dumps({key: value for key, value in refinements[-1].items()
                                                  if key != "fit"}), flush=True)
            q, z, prediction = next_q, next_z, next_prediction
            np.savez_compressed(OUT/f"blase_planet_baseline_{name}.npz", q=q, z=z, prediction=prediction)
            stable_count = stable_count+1 if drift < 1e-8 else 0
            if stable_count >= 2:
                break
        if stable_count < 2:
            np.testing.assert_array_equal(context["y_over_c"], observed_copy)
            for key, value in mask_copy.items():
                np.testing.assert_array_equal(context[key], value)
            np.testing.assert_array_equal(problem.calibration_mask, calibration_copy)
            assert digest(adopted_path) == initial_adopted_hash
            (OUT/"blase_planet_baseline_drift.json").write_text(json.dumps(
                dict(audit=audit, model=name, initial_fit=diagnostics,
                     null_refit_checks=refinements), indent=2)+"\n")
            raise RuntimeError(f"Uninjected {name} baseline has material optimizer drift: {refinements[-1]}")
        baseline[name] = (q, z, prediction)
        baseline_report[name] = dict(fit=diagnostics,
            null_refit_checks=refinements,
            comparison_to_stored_fit=compare_baseline(
                context, prediction, identity_component if name == "phoenix" else stored_prediction),
            q_change_from_saved_max=float(np.max(np.abs(q-initial[0]))),
            z_change_from_initial_max=float(np.max(np.abs(z-initial[1]))))
        arrays.update({f"baseline_q_{name}": q, f"baseline_z_{name}": z,
                       f"baseline_prediction_{name}": prediction})
        print("Uninjected baseline", name, "complete", flush=True)
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][context["exposures"]]
    wave = context["wave"]
    velocity = C_KMS*np.log(wave/np.exp(np.mean(np.log(wave), axis=1))[:, None])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
        audit=audit, baseline=baseline_report,
        training_frames=context["exposures"][:5].tolist(),
        evaluation_frames=context["exposures"][5:].tolist(),
        paired_models={"phoenix": "Original PHOENIX fixed; refit the adopted 269 telluric strengths.",
                       "stellar": "Refit shared stellar amplitudes on telluric-clear training pixels with adopted strengths fixed, then freeze the star and refit 269 telluric strengths."},
        definition="Gain=dot(Qs, injected-minus-uninjected residual)/dot(Qs,Qs). Delta=stellar gain minus PHOENIX control gain.",
        limitations=[
            "Artificial observed-resolution Gaussian signals, not physical molecular spectra injected before IP convolution.",
            "Kp values are hypothetical controls, not measured orbital parameters.",
            "Only the five training targets change during fitting. Held-out frames are used only for response evaluation.",
            "Masks, synthetic stellar clone, line selection and priors stay frozen after their original construction.",
            "The stellar calibration and subsequent telluric refit are both repeated after injection; other earlier nuisance fits are held fixed.",
            "Gains above unity are projection effects and do not represent added information.",
            "These diagnostics cannot establish planet detectability or guarantee preservation for all spectra."], cases=[])
    for index, case in enumerate(CASES):
        case_started = time.monotonic()
        signal = make_signal(case, context, phase, velocity)
        expected = context["project"](signal)
        target = observed.copy()
        target[:, :5] += signal[:, :5]
        np.testing.assert_array_equal(target[:, 5:], observed[:, 5:])
        row = dict(**case, models={})
        shifts = case["kp_kms"]*np.sin(2*np.pi*phase)
        row["velocity_span_kms"] = dict(training=float(np.ptp(shifts[:5])),
                                         evaluation=float(np.ptp(shifts[5:])),
                                         all_frames=float(np.ptp(shifts)))
        arrays[f"signal_{index}"] = signal
        arrays[f"filtered_signal_{index}"] = expected
        for name, flexible in [("phoenix", False), ("stellar", True)]:
            q0, z0, prediction0 = baseline[name]
            q, z, prediction, diagnostics = fit_pipeline(
                problem, target, (q0, z0), flexible=flexible, max_nfev=args.max_nfev)
            response = context["project"](observed+signal-prediction)-context["project"](observed-prediction0)
            exact = expected-context["project"](prediction-prediction0)
            np.testing.assert_allclose(response, exact, rtol=1e-9, atol=1e-12)
            record = response_metrics(signal, expected, response, context)
            record.update(fit=diagnostics,
                response_identity_max_error=float(np.max(np.abs(response-exact))),
                q_change_max=float(np.max(np.abs(q-q0))),
                z_change_max=float(np.max(np.abs(z-z0))))
            row["models"][name] = record
            arrays.update({f"q_{name}_{index}": q, f"z_{name}_{index}": z,
                           f"response_{name}_{index}": response,
                           f"prediction_{name}_{index}": prediction})
        control, stellar = row["models"]["phoenix"], row["models"]["stellar"]
        row["paired_gain_difference"] = {
            key: stellar[key]["incremental_gain"]-control[key]["incremental_gain"]
            for key in ["training", "evaluation_score", "evaluation_included"]}
        row["paired_gain_difference"]["per_order"] = {
            key: stellar["per_order"][key]["evaluation_score"]["incremental_gain"]
                 -control["per_order"][key]["evaluation_score"]["incremental_gain"]
            for key in control["per_order"]}
        row["seconds"] = time.monotonic()-case_started
        result["cases"].append(row)
        print(json.dumps(dict(case=case, paired_gain_difference=row["paired_gain_difference"])), flush=True)
    np.testing.assert_array_equal(context["y_over_c"], observed_copy)
    for key, value in mask_copy.items():
        np.testing.assert_array_equal(context[key], value)
    np.testing.assert_array_equal(problem.calibration_mask, calibration_copy)
    assert digest(adopted_path) == initial_adopted_hash
    result["audit"]["adopted_sha256_after"] = digest(adopted_path)
    paths = [Path(__file__), ROOT/"fit_blase_stellar.py", ROOT/"fit_line_strength.py",
             ROOT/"check_strength_planet.py", ROOT/"check_smooth_signal.py",
             ROOT/"smooth_residuals.py", ROOT/"data/20201214_info.csv", source,
             OUT/"blase_stellar_fit.json", adopted_path]
    result["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in paths}
    result["seconds"] = time.monotonic()-started
    np.savez_compressed(OUT/"blase_planet.npz", **arrays)
    (OUT/"blase_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    (OUT/"blase_independent_audit.json").write_text(json.dumps(result["audit"], indent=2)+"\n")
    write_report(result)
    print("Paired stellar injection check complete", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
