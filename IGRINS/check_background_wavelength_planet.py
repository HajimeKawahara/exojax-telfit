"""Measure conditional planet response to shared background/wavelength refits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_smooth_signal import forest_signal
from check_strength_planet import projection_metrics, run_suite
from fit_background_wavelength import build_context, fit_case, predict_correction, project

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"


def main():
    context = build_context()
    metadata = json.loads((OUT/"background_wavelength_fit.json").read_text())
    name = metadata["selection"]["selected_case"]
    with np.load(OUT/"background_wavelength_fit.npz") as saved:
        baseline = saved[f"theta_{name}"]
        saved_correction = saved[f"correction_raw_{name}"]
    with np.load(OUT/"quadratic_strength_fit.npz") as saved:
        reference = saved["component_baseline"]
        np.testing.assert_array_equal(saved["exposures"], context["exposures"])
        np.testing.assert_array_equal(saved["wavelength_um"], context["wavelength_um"])
    checked, diagnostics = fit_case(context, name)
    np.testing.assert_allclose(checked, baseline, rtol=0, atol=1e-13)
    np.testing.assert_allclose(predict_correction(context, baseline, filtered=False), saved_correction,
                               rtol=0, atol=1e-13)

    def fit(target, initial):
        filtered = project(target-context["component"], context["wavelength_um"], context["included_mask"])
        theta, fit_diagnostics = fit_case(context, name, target=filtered)
        return theta, [fit_diagnostics]

    def predict(theta):
        return context["component"]+predict_correction(context, theta, filtered=False)

    source = dict(wave=context["wavelength_um"], allowed=context["included_mask"],
                  score=context["score_mask"], orders=context["orders"], exposures=context["exposures"],
                  y_over_c=context["y_over_c"], reference_component=reference)
    result = run_suite(fit, predict, baseline, source_arrays=source, include_low_depth=True)
    result.pop("prior_scale")
    result["selected_case"] = name
    result["selection"] = "Frozen preceding training-CV selection; injection cases do not select the model"
    result["baseline_theta"] = baseline.tolist()
    result["baseline_fit_diagnostics"] = diagnostics
    result["definitions"]["baseline"] = "Adopted fixed_contrast10 plus the uninjected selected background/wavelength correction"
    result["definitions"]["fit_target"] = "Only five training spectra determine the added shared background/wavelength coefficients"
    result["definitions"]["design"] = "Frozen physical-background and smooth wavelength-derivative response; no residual template"
    result["limitations"] = [
        "Observed-resolution artificial Gaussian forests, not physical planetary molecular spectra.",
        "The identical original component_baseline approximates attenuation as in preceding injection suites.",
        "Kp=0/100/200 km/s are hypothetical controls, not measured orbital parameters.",
        "Only the added background/wavelength coefficients are refitted; adopted line strengths and their weak-line weights are fixed.",
        "TP, stellar spectrum, columns, original IP, base wavelength solution and continuum remain fixed.",
        "The wavelength derivative design is fixed at the adopted prediction and is not recalculated after injection.",
        "The test isolates this correction stage; it is not a full-pipeline injection/recovery measurement.",
        "Gain is signed projected amplitude relative to the identical Q5-only signal, not detection significance.",
        "Shared background bounds can produce depth/sign dependence if their active sets change.",
    ]
    arrays = {key: context[key] for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask"]}
    arrays["baseline_theta"] = baseline
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][context["exposures"]]
    velocity = 299792.458*np.log(context["wavelength_um"]/np.exp(np.mean(np.log(context["wavelength_um"]), axis=1))[:, None])
    maximum_metric_difference = 0.
    for i, case in enumerate(result["cases"]):
        theta = np.asarray(case["fitted_theta"])
        case["active_bound_count"] = int(np.sum(abs(theta) >= context["bounds"]-1e-10))
        signal = np.array([forest_signal(v, phase, part, case["fwhm_kms"], case["kp_kms"],
                                        seed=20261111+int(order))*case["depth"]/1e-3
                           for v, part, order in zip(velocity, reference, context["orders"])])
        expected = project(signal, context["wavelength_um"], context["included_mask"])
        response = expected-predict_correction(context, theta-baseline)
        for split, rows, mask in [("training", slice(0, 5), context["included_mask"]),
                                  ("evaluation_score", slice(5, None), context["score_mask"]),
                                  ("evaluation_included", slice(5, None), context["included_mask"])]:
            independent = projection_metrics(signal[:, rows], expected[:, rows], response[:, rows], mask)
            for key, value in independent.items():
                delta = abs(value-case[split][key])
                maximum_metric_difference = max(maximum_metric_difference, delta)
                np.testing.assert_allclose(value, case[split][key], atol=2e-11, rtol=0)
        arrays[f"signal_{i}"] = signal
        arrays[f"expected_q5_{i}"] = expected
        arrays[f"response_{i}"] = response
        arrays[f"theta_{i}"] = theta
    moving = result["cases"][:4]
    summaries = {}
    for split in ["training", "evaluation_score", "evaluation_included"]:
        gains = [100*case[split]["incremental_gain"] for case in moving]
        shapes = [case[split]["response_correlation"] for case in moving]
        summaries[split] = dict(moving_gain_percent_range=[min(gains), max(gains)],
                                moving_correlation_range=[min(shapes), max(shapes)],
                                stationary_gain_percent=100*result["cases"][4][split]["incremental_gain"],
                                sign_gain_difference_percentage_points=100*(result["cases"][5][split]["incremental_gain"]-
                                                                           result["cases"][3][split]["incremental_gain"]),
                                low_depth_gain_difference_percentage_points=100*(result["cases"][6][split]["incremental_gain"]-
                                                                                result["cases"][3][split]["incremental_gain"]))
    per_order = [100*case["per_order"][str(order)]["evaluation_score"]["incremental_gain"]
                 for case in moving for order in context["orders"]]
    result["summary"] = dict(**summaries, moving_per_order_evaluation_gain_percent_range=[min(per_order), max(per_order)],
                             all_fits_converged=all(d["success"] for c in result["cases"] for d in c["fit_diagnostics"]),
                             independent_qr_metric_max_difference=maximum_metric_difference)
    paths = [Path(__file__), ROOT/"fit_background_wavelength.py", ROOT/"check_strength_planet.py", ROOT/"check_smooth_signal.py",
             ROOT/"smooth_residuals.py", ROOT/"data/20201214_info.csv", OUT/"background_wavelength_fit.json",
             OUT/"background_wavelength_fit.npz", OUT/"background_wavelength_selection.json", OUT/"adopted_telluric_fit.npz",
             OUT/"quadratic_strength_fit.npz"]
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    np.savez_compressed(OUT/"background_wavelength_planet.npz", **arrays)
    (OUT/"background_wavelength_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    labels = [f"FWHM {c['fwhm_kms']:g}; Kp {c['kp_kms']:g}\ndepth {c['depth']:g}" for c in result["cases"]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    for split, label in [("training", "Training"), ("evaluation_score", "Evaluation score")]:
        axes[0].plot(x, [100*c[split]["incremental_gain"] for c in result["cases"]], "o-", label=label)
        axes[1].plot(x, [c[split]["response_correlation"] for c in result["cases"]], "o-", label=label)
    for ax, reference, ylabel in [(axes[0], 100., "Projected amplitude vs Q5 only (%)"),
                                  (axes[1], 1., "Response/template correlation")]:
        ax.axhline(reference, color=".5", ls="--", lw=.8)
        ax.set(xticks=x, xticklabels=labels, ylabel=ylabel)
        ax.tick_params(axis="x", labelsize=6)
        ax.legend(fontsize=8)
    fig.savefig(OUT/"background_wavelength_planet.png", dpi=160)
    plt.close(fig)
    report = ["Artificial planet response of the conditional background/wavelength correction",
              "==========================================================================", "",
              "Frozen selected case: "+name, "",
              "Baseline: adopted weak-line-suppressed strengths and fixed IP, plus the",
              "uninjected background/wavelength correction. Refit only the added coefficients",
              "using five training epochs; predict all four evaluation epochs without fitting.",
              "The seven artificial forests match the previous suite's widths, motions, depths,",
              "seeds, and original-model attenuation. Q5 is applied to signals and responses.", "",
              "FWHM Kp Depth Training_gain(%) Evaluation_gain(%) Evaluation_shape_correlation"]
    for case in result["cases"]:
        report.append(f"{case['fwhm_kms']:g} {case['kp_kms']:g} {case['depth']:g} "
                      f"{100*case['training']['incremental_gain']:.7f} {100*case['evaluation_score']['incremental_gain']:.7f} "
                      f"{case['evaluation_score']['response_correlation']:.9f}")
    report += ["", "Summary", "-------", json.dumps(result["summary"], indent=2), "", "Limits", "------"]
    report += result["limitations"]+["", "100% amplitude is relative to Q5 only, not to the raw injected spectrum.",
                                      "Near-unit amplitude does not imply identical line shape or full-pipeline recovery.", ""]
    (OUT/"BACKGROUND_WAVELENGTH_PLANET_REPORT.txt").write_text("\n".join(report))
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
