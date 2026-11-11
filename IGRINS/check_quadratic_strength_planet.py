"""Measure artificial planet-line response to joint IP and intensity fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_strength_planet import run_suite

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def run_setting(context, config, baseline, fit_callback, predict_callback,
                line_indices, ip_indices):
    """Refit seven paired injections with frozen prior hyperparameters."""
    result = run_suite(
        fit_callback, predict_callback, baseline, source_arrays=context,
        prior_scale=config.get("prior_scale", 1.), include_low_depth=True)
    upper = np.array([r["uncertainty_upper"] for r in context["records"]])
    result["config"] = config
    result["baseline_theta"] = np.asarray(baseline).tolist()
    result["line_parameter_indices"] = np.asarray(line_indices).tolist()
    result["ip_parameter_indices"] = np.asarray(ip_indices).tolist()
    result["definitions"]["baseline"] = (
        "Uninjected shared-intensity/IP fit with the same frozen prior and polynomial Q")
    result["definitions"]["fit_target"] = (
        "Only five training exposures enter the injected shared-intensity/IP fit")
    result["definitions"]["prior"] = (
        "Configuration and any line-visibility weights remain fixed during injections")
    result["limitations"] = [
        "Observed-resolution artificial Gaussian forests, not physical planetary molecular spectra.",
        "Kp=0/100/200 km/s are hypothetical controls, not measured orbital parameters.",
        "The original fixed transmission/stellar component approximates attenuation for every case.",
        "The experiment refits the new IP parameters and shared line strengths allowed by each configuration.",
        "TP, star, molecular columns, wavelength solution, original continuum and old epoch IP factors remain fixed.",
        "The earlier nuisance-fitting pipeline is not rerun, so these gains are not full-pipeline preservation.",
        "Near-unit projected amplitude can coexist with line-shape changes; gain above unity is not extra information.",
        "No retention value or residual-RMS improvement is a detection significance.",
        "Finite-depth nonlinear fits with active bounds need not be exactly linear in injected depth.",
    ]
    line_indices = np.asarray(line_indices, dtype=int)
    ip_indices = np.asarray(ip_indices, dtype=int)
    for case in result["cases"]:
        theta = np.asarray(case["fitted_theta"])
        delta = theta-np.asarray(baseline)
        z = theta[line_indices]
        del case["active_bound_count"]
        case["outside_original_uncertainty_count"] = int(np.sum(np.abs(z) > 1+1e-5))
        case["at_positivity_bound_count"] = int(np.sum(1+upper*z < 1e-6))
        case["line_parameter_change_max"] = float(np.max(np.abs(delta[line_indices])))
        case["fractional_line_strength_change_max"] = float(np.max(np.abs(upper*delta[line_indices])))
        case["ip_parameter_change_max"] = float(np.max(np.abs(delta[ip_indices]))) if len(ip_indices) else 0.
    return result


def summarize_setting(setting):
    """Keep amplitude and shape summaries separate."""
    moving = setting["cases"][:4]
    result = {}
    for split in ["training", "evaluation_score", "evaluation_included"]:
        gains = [100*c[split]["incremental_gain"] for c in moving]
        correlations = [c[split]["response_correlation"] for c in moving]
        result[split] = dict(
            moving_gain_percent_range=[min(gains), max(gains)],
            moving_correlation_range=[min(correlations), max(correlations)],
            stationary_gain_percent=100*setting["cases"][4][split]["incremental_gain"],
            sign_gain_difference_percentage_points=100*(
                setting["cases"][5][split]["incremental_gain"]
                - setting["cases"][3][split]["incremental_gain"]),
            low_depth_gain_difference_percentage_points=100*(
                setting["cases"][6][split]["incremental_gain"]
                - setting["cases"][3][split]["incremental_gain"]))
    order_gains = [100*c["per_order"][o]["evaluation_score"]["incremental_gain"]
                   for c in moving for o in c["per_order"]]
    result["moving_per_order_evaluation_gain_percent_range"] = [min(order_gains), max(order_gains)]
    result["max_identity_error"] = max(
        c["injected_minus_uninjected_identity_max_error"] for c in setting["cases"])
    result["all_fits_converged"] = all(
        item["success"] for c in setting["cases"] for item in c["fit_diagnostics"])
    return result


def write_outputs(result, previous):
    """Save all fitted vectors plus paired amplitude and shape diagnostics."""
    for setting in result["settings"].values():
        setting["summary"] = summarize_setting(setting)
    (OUT / "quadratic_strength_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    old = previous["settings"]["soft10"]["cases"]
    x = np.arange(len(old))
    labels = [f"FWHM {c['fwhm_kms']:g}; Kp {c['kp_kms']:g}\ndepth {c['depth']:g}" for c in old]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), constrained_layout=True, sharex=True)
    for row, split in enumerate(["training", "evaluation_score"]):
        axes[row, 0].plot(x, [100*c[split]["incremental_gain"] for c in old],
                          "o--", color=".5", label="Previous uniform soft10")
        axes[row, 1].plot(x, [c[split]["response_correlation"] for c in old],
                          "o--", color=".5", label="Previous uniform soft10")
        for name, setting in result["settings"].items():
            cases = setting["cases"]
            axes[row, 0].plot(x, [100*c[split]["incremental_gain"] for c in cases], "o-", label=name)
            axes[row, 1].plot(x, [c[split]["response_correlation"] for c in cases], "o-", label=name)
        axes[row, 0].axhline(100, color=".7", ls=":")
        axes[row, 1].axhline(1, color=".7", ls=":")
        title = "Training spectra" if row == 0 else "Evaluation scoring pixels"
        axes[row, 0].set(ylabel="Projected amplitude (%)\nrelative to polynomial only", title=title)
        axes[row, 1].set(ylabel="Response/template correlation", title=title)
    for ax in axes.flat:
        ax.legend(fontsize=7)
        ax.set(xticks=x, xticklabels=labels)
        ax.tick_params(axis="x", labelsize=6)
    fig.savefig(OUT / "quadratic_strength_planet.png", dpi=160)
    plt.close(fig)
    lines = [
        "Artificial planet-line response: quadratic IP and adaptive intensity priors",
        "========================================================================", "",
        "All comparisons use identical observed-resolution Gaussian forests and the",
        "same original transmission/stellar component for approximate attenuation.",
        "The five training exposures alone determine injected fit parameters.",
        "Prior hyperparameters and weak-line weights are frozen before injection.",
        "100% gain means unchanged signed projected amplitude relative to the same",
        "degree-five polynomial alone. It does not guarantee unchanged line shapes.", "",
        f"Selected residual-fit case: {result['selected_case']}",
        "These injection tests do not select the residual-fit configuration.", "",
        "Case FWHM Kp Depth Train_gain(%) Eval_gain(%) Eval_shape_correlation",
    ]
    for name, setting in result["settings"].items():
        for case in setting["cases"]:
            lines.append(
                f"{name} {case['fwhm_kms']:g} {case['kp_kms']:g} {case['depth']:g} "
                f"{100*case['training']['incremental_gain']:.5f} "
                f"{100*case['evaluation_score']['incremental_gain']:.5f} "
                f"{case['evaluation_score']['response_correlation']:.6f}")
        lines += ["", f"{name} summary:", json.dumps(setting["summary"], indent=2), ""]
    lines += [
        "Interpretation", "--------------",
        "A broader or more flexible telluric fit can absorb components of a planet",
        "signal that overlap its model derivatives. Compare training and evaluation",
        "responses, shapes and the stationary control instead of amplitude alone.",
        "The selected IP/strength parameters are refitted, but earlier TP, column,",
        "stellar, wavelength and continuum fits are not. These tests are conditional",
        "on the original nuisance fits and approximate post-IP injected spectra.",
        "Kp values are hypothetical. The results are neither a full planet retrieval",
        "nor a detection significance. Sign/depth checks probe local response only.",
        "Full vectors, convergence diagnostics and provenance are preserved in JSON.", "",
    ]
    (OUT / "QUADRATIC_STRENGTH_PLANET_REPORT.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs="+", required=True)
    args = parser.parse_args()
    from fit_quadratic_strength import build_context, fit_joint_strength, predict_components

    context = build_context()
    fit_path = OUT / "quadratic_strength_fit.json"
    arrays_path = OUT / "quadratic_strength_fit.npz"
    metadata = json.loads(fit_path.read_text())
    previous_path = OUT / "relaxed_strength_planet.json"
    previous = json.loads(previous_path.read_text())
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  selected_case=metadata["selected_case"],
                  injection_case_selection="Joint IP candidate and matched fixed-IP intensity prior",
                  settings={})
    with np.load(arrays_path) as saved:
        np.testing.assert_allclose(saved["y_over_c"], context["y_over_c"], rtol=0, atol=1e-12)
        for name in dict.fromkeys(args.case):
            case = metadata["cases"][name]
            config = case["config"]
            baseline = saved[f"theta_{name}"].copy()
            predicted = predict_components(context, baseline, config)
            difference = np.max(np.abs(predicted-saved[f"component_{name}"]))
            np.testing.assert_allclose(predicted, saved[f"component_{name}"], rtol=0, atol=1e-11)
            print(f"Starting injections: {name}; baseline reconstruction {difference:.3g}", flush=True)
            setting = run_setting(
                context, config, baseline,
                lambda target, initial: fit_joint_strength(
                    context, config, target=target, initial=initial),
                lambda theta: predict_components(context, theta, config),
                context["line_parameter_indices"], context["ip_parameter_indices"])
            setting["baseline_fit_diagnostics"] = case["fit"]
            setting["baseline_saved_component_max_difference"] = float(difference)
            for injected, old in zip(setting["cases"], previous["settings"]["soft10"]["cases"], strict=True):
                for key in ["fwhm_kms", "kp_kms", "depth"]:
                    assert injected[key] == old[key]
                injected["previous_uniform_soft10"] = {
                    split: old[split] for split in ["training", "evaluation_score", "evaluation_included"]}
            result["settings"][name] = setting
            # Checkpoint a finished setting before starting the second suite.
            (OUT / "quadratic_strength_planet_checkpoint.json").write_text(json.dumps(result, indent=2)+"\n")
    paths = [Path(__file__), ROOT / "fit_quadratic_strength.py", ROOT / "quadratic_ip_response.py",
             ROOT / "fit_line_strength.py", ROOT / "line_strength_support.py",
             ROOT / "check_strength_planet.py", ROOT / "check_smooth_signal.py",
             ROOT / "smooth_residuals.py", ROOT / "data/20201214_info.csv",
             fit_path, arrays_path, previous_path] + context["source_files"]
    selection_path = OUT / "quadratic_strength_selection.json"
    if selection_path.exists():
        paths.append(selection_path)
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}
    write_outputs(result, previous)


if __name__ == "__main__":
    main()
