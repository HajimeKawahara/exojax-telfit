"""Check artificial planet-line response to exact shared line-shape fitting.

The old Gaussian-forest control is reused after checking its input hashes.
An additional moving water-line pattern is refitted for both the new model
and its matched intensity-only control. These are conditional signal tests.
"""
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

import numpy as np

from check_strength_planet import projection_metrics, run_suite

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
SPLITS = ["training", "evaluation_score", "evaluation_included"]


def annotate_parameters(setting, context, baseline):
    """Replace the old suite's strength-only bound count with actual bounds."""
    line_indices = context["line_parameter_indices"]
    shape_indices = context["shape_parameter_indices"]
    upper = np.array([r["uncertainty_upper"] for r in context["records"]])
    setting["baseline_theta"] = np.asarray(baseline).tolist()
    for case in setting["cases"]:
        case.pop("active_bound_count", None)
        theta = np.asarray(case["fitted_theta"])
        change = theta-np.asarray(baseline)
        case["at_strength_positivity_bound_count"] = int(
            np.sum(1+upper*theta[line_indices] < 1e-6))
        case["fractional_strength_change_max"] = float(
            np.max(np.abs(upper*change[line_indices])))
        shapes = theta[shape_indices].reshape(-1, 2)
        shape_change = change[shape_indices].reshape(-1, 2)
        case["shape_at_bound_count"] = int(
            np.sum(np.log(2.)-np.abs(shapes[:, 0]) < 1e-5)
            + np.sum(1.-np.abs(shapes[:, 1]) < 1e-5))
        case["log_width_change_max"] = float(np.max(np.abs(shape_change[:, 0])))
        case["center_change_max_kms"] = float(np.max(np.abs(shape_change[:, 1])))


def water_signal(context, case):
    """Place Gaussian features at the 35 frozen detectable water-line centers."""
    info = np.genfromtxt(ROOT / "data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][context["exposures"]]
    shifts = case["kp_kms"]*np.sin(2*np.pi*phase)
    sigma = case["fwhm_kms"]/np.sqrt(8*np.log(2))
    records = context["shape_records"]
    normalization = max(r["original_contrast"] for r in records)
    signal = np.zeros_like(context["y_over_c"])
    for b, order in enumerate(context["orders"]):
        lines = [r for r in records if r["order"] == order]
        if not lines:
            continue
        reference = np.exp(np.mean(np.log(context["wave"][b])))
        velocity = C_KMS*np.log(context["wave"][b]/reference)
        centers = C_KMS*np.log(np.array([1e4/r["nu"] for r in lines])/reference)
        weights = np.array([r["original_contrast"] for r in lines])/normalization
        offsets = velocity[None, :, None]-centers[None, None, :]-shifts[:, None, None]
        forest = np.sum(weights[None, None, :]*np.exp(-.5*(offsets/sigma)**2), axis=-1)
        signal[b] = -case["depth"]*context["reference_component"][b]*forest
    return signal


def run_water_suite(context, config, baseline, fit_callback, predict_callback):
    """Refit a moving water pattern with sign, depth and stationary controls."""
    observed = context["y_over_c"]
    project = context["project"]
    baseline_prediction = predict_callback(baseline)
    baseline_residual = project(observed-baseline_prediction)
    cases = [dict(fwhm_kms=6., kp_kms=200., depth=1e-3),
             dict(fwhm_kms=6., kp_kms=0., depth=1e-3),
             dict(fwhm_kms=6., kp_kms=200., depth=-1e-3),
             dict(fwhm_kms=6., kp_kms=200., depth=1e-4)]
    output = dict(config=config, cases=[], pattern=dict(
        source="Frozen original-model H2O contrast > 2% selection",
        line_count=len(context["shape_records"]),
        weights="Original predicted line contrast divided by its maximum across all 35 lines",
        centers="Vacuum wavelengths from the selected raw HITRAN records",
        velocity_law="Kp*sin(2*pi*observed_phase), zero systemic velocity",
        attenuation="Same original saved transmission/stellar component for all configurations",
        limitations=["A telluric-like water pattern, not a hot-planet molecular spectrum.",
                     "Finite selected line list; features entering an order from outside it are absent.",
                     "Injection is at observed resolution, after approximate attenuation.",
                     "The 35 lines and their weights are frozen without inspecting injections."]))
    for case in cases:
        started = time.monotonic()
        signal = water_signal(context, case)
        target = observed.copy()
        target[:, :5] += signal[:, :5]
        np.testing.assert_array_equal(target[:, 5:], observed[:, 5:])
        theta, diagnostics = fit_callback(target, initial=baseline.copy())
        if not all(d["success"] for d in diagnostics):
            raise RuntimeError("A water-pattern injection fit did not converge")
        prediction = predict_callback(theta)
        response = project(observed+signal-prediction)-baseline_residual
        expected = project(signal)
        exact = expected-project(prediction-baseline_prediction)
        np.testing.assert_allclose(response, exact, rtol=1e-9, atol=1e-13)
        record = dict(**case, fitted_theta=np.asarray(theta).tolist(),
                      fit_diagnostics=diagnostics, per_order={},
                      injected_minus_uninjected_identity_max_error=float(np.max(abs(response-exact))),
                      seconds=time.monotonic()-started)
        for name, es, mask in [("training", slice(0, 5), context["allowed"]),
                               ("evaluation_score", slice(5, None), context["score"]),
                               ("evaluation_included", slice(5, None), context["allowed"])]:
            record[name] = projection_metrics(signal[:, es], expected[:, es], response[:, es], mask)
        for b, order in enumerate(context["orders"]):
            if not np.any(signal[b]):
                record["per_order"][str(order)] = dict(no_injected_signal=True)
                continue
            record["per_order"][str(order)] = {
                name: projection_metrics(signal[b:b+1, es], expected[b:b+1, es],
                                         response[b:b+1, es], mask[b:b+1])
                for name, es, mask in [("training", slice(0, 5), context["allowed"]),
                                       ("evaluation_score", slice(5, None), context["score"])]}
        output["cases"].append(record)
        print(f"Water pattern {config['name']}: Kp={case['kp_kms']:g} "
              f"depth={case['depth']:g} eval gain="
              f"{record['evaluation_score']['incremental_gain']:.8f}", flush=True)
    annotate_parameters(output, context, baseline)
    return output


def reused_control(context, saved):
    """Validate identical inputs before reusing the previous Gaussian control."""
    path = OUT / "quadratic_strength_planet.json"
    previous = json.loads(path.read_text())
    for relative in ["check_strength_planet.py", "check_smooth_signal.py",
                     "smooth_residuals.py", "data/20201214_info.csv"]:
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == previous["source_sha256"][relative]
    with np.load(OUT / "quadratic_strength_fit.npz") as old:
        for name, current in [("y_over_c", context["y_over_c"]),
                              ("wavelength_um", context["wave"]),
                              ("exposures", context["exposures"]),
                              ("included_mask", context["allowed"]),
                              ("score_mask", context["score"]),
                              ("component_baseline", context["reference_component"])]:
            np.testing.assert_allclose(old[name], current, rtol=0, atol=1e-12)
        difference = float(np.max(abs(old["component_fixed_contrast10"]
                                     -saved["component_strength_control"])))
        if difference > 1e-6:
            raise ValueError(f"Refitted control differs too much for reuse: {difference}")
    result = previous["settings"]["fixed_contrast10"]
    result["reuse"] = dict(source=str(path.relative_to(ROOT)),
                           original_config="fixed_contrast10",
                           matched_new_config="strength_control",
                           refitted_control_component_max_difference=difference)
    return result, path


def write_report(result):
    """Save paired gains and shape information without selecting any fit case."""
    (OUT / "line_shape_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Artificial planet-line response to exact line-shape fitting",
             "===========================================================", "",
             f"Residual-fit selected case: {result['selected_case']}",
             "The injection tests do not select or retune the residual-fit model.",
             "100% gain is unchanged signed projected amplitude relative to Q5 alone.",
             "It is not unchanged line shape or full-pipeline planet preservation.", "",
             "Pattern Model FWHM Kp Depth Train_gain(%) Eval_gain(%) Eval_correlation"]
    suites = [("gaussian", "adopted_control", result["adopted_gaussian_control"])]
    for name, setting in result["settings"].items():
        if "gaussian" in setting:
            suites.append(("gaussian", name, setting["gaussian"]))
        suites.append(("water", name, setting["water"]))
    for pattern, name, suite in suites:
        for case in suite["cases"]:
            lines.append(f"{pattern} {name} {case['fwhm_kms']:g} {case['kp_kms']:g} "
                         f"{case['depth']:g} {100*case['training']['incremental_gain']:.6f} "
                         f"{100*case['evaluation_score']['incremental_gain']:.6f} "
                         f"{case['evaluation_score']['response_correlation']:.8f}")
    lines += ["", "Limitations", "-----------", *result["limitations"], ""]
    (OUT / "LINE_SHAPE_PLANET_REPORT.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", help="Default: the CV-selected case")
    args = parser.parse_args()
    from fit_line_shapes import build_context, fit_shapes, predict_components

    fit_path, arrays_path = OUT/"line_shape_fit.json", OUT/"line_shape_fit.npz"
    meta = json.loads(fit_path.read_text())
    selected = meta["selection"]["selected_case"]
    requested = list(dict.fromkeys(args.case or [selected]))
    context = build_context()
    with np.load(arrays_path) as source:
        saved = {k: source[k] for k in source.files}
    np.testing.assert_allclose(saved["y_over_c"], context["y_over_c"], rtol=0, atol=1e-12)
    control, previous_path = reused_control(context, saved)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=selected,
                  requested_cases=requested, adopted_gaussian_control=control, settings={},
                  limitations=[
                      "Exact nonlinear refits vary only the shared strengths and allowed H2O widths/centers.",
                      "Original IP, TP, molecular columns, stellar model, wavelength and continuum fits remain fixed.",
                      "Only five training spectra determine injected fit parameters; evaluation data never enter the fit.",
                      "Degree-five polynomial coefficients use all included pixels, including evaluation score pixels.",
                      "Hyperparameters, line selection and contrast weights are frozen before injection.",
                      "All signals are artificial observed-resolution patterns with approximate original-model attenuation.",
                      "Kp values are hypothetical, and the water pattern is not a physical planetary atmosphere.",
                      "This does not test loss in the preceding nuisance fits or establish detection significance.",
                      "Near-unit signed gain can coexist with changed line shape; compare correlation, norm and error too.",
                      "Historical evaluation spectra have been inspected in earlier model development."])
    for name in dict.fromkeys(requested+["strength_control"]):
        config = meta["cases"][name]["config"]
        baseline = saved[f"theta_{name}"].copy()
        prediction = predict_components(context, baseline)
        np.testing.assert_allclose(prediction, saved[f"component_{name}"], rtol=0, atol=1e-11)
        fit_callback = lambda target, initial: fit_shapes(context, config, target=target, initial=initial)
        predict_callback = lambda theta: predict_components(context, theta)
        setting = dict(config=config, baseline_fit_diagnostics=meta["cases"][name]["fit"])
        if name in requested and name != "strength_control":
            setting["gaussian"] = run_suite(fit_callback, predict_callback, baseline,
                                            source_arrays=context, include_low_depth=True,
                                            prior_scale=config["prior_multiplier"])
            setting["gaussian"]["config"] = config
            annotate_parameters(setting["gaussian"], context, baseline)
            setting["gaussian"]["definitions"]["baseline"] = (
                "Uninjected exact shared intensity/width/center fit under the frozen configuration")
            setting["gaussian"]["definitions"]["fit_target"] = (
                "Only five training spectra determine all allowed injected-fit parameters")
            setting["gaussian"]["limitations"] = result["limitations"]
        setting["water"] = run_water_suite(context, config, baseline, fit_callback, predict_callback)
        result["settings"][name] = setting
        (OUT/"line_shape_planet_checkpoint.json").write_text(json.dumps(result, indent=2)+"\n")
    paths = [Path(__file__), ROOT/"fit_line_shapes.py", ROOT/"check_strength_planet.py",
             ROOT/"check_smooth_signal.py", ROOT/"smooth_residuals.py", ROOT/"data/20201214_info.csv",
             fit_path, arrays_path, OUT/"line_shape_selection.json", previous_path,
             OUT/"quadratic_strength_fit.npz", OUT/"adopted_telluric_fit.npz"]+context["source_files"]
    result["source_sha256"] = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    write_report(result)


if __name__ == "__main__":
    main()
