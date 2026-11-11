"""Compare artificial planet-line responses with relaxed strength priors."""
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


def run_setting(context, prior_scale):
    """Refit the uninjected baseline and seven controlled signal injections."""
    from fit_line_strength import predict_components
    from fit_relaxed_strength import fit_relaxed

    baseline, diagnostics = fit_relaxed(context, prior_scale=prior_scale, cap=None)
    if not all(item["success"] for item in diagnostics):
        raise RuntimeError("An uninjected relaxed strength fit did not converge")
    result = run_suite(
        lambda target, initial: fit_relaxed(
            context, target=target, prior_scale=prior_scale, cap=None, initial=initial),
        lambda theta: predict_components(context, theta), baseline,
        source_arrays=context, prior_scale=prior_scale, include_low_depth=True)
    upper = np.array([record["uncertainty_upper"] for record in context["records"]])
    result["cap"] = None
    result["baseline_theta"] = np.asarray(baseline).tolist()
    result["baseline_fit_diagnostics"] = diagnostics
    result["definitions"]["bounds"] = "S >= 0 only; no upper cap on strength changes"
    for case in result["cases"]:
        theta = np.array(case["fitted_theta"])
        # The reused suite's +/-1 statistic describes the previous hard cap.
        del case["active_bound_count"]
        case["outside_original_uncertainty_count"] = int(np.sum(np.abs(theta) > 1+1e-5))
        case["at_positivity_bound_count"] = int(np.sum(1+upper*theta < 1e-6))
    return result


def add_comparison(result, previous):
    """Keep paired old/new responses for the identical artificial signals."""
    for setting in result["settings"].values():
        for case, old in zip(setting["cases"], previous["cases"], strict=True):
            for key in ("fwhm_kms", "kp_kms", "depth"):
                assert case[key] == old[key]
            case["previous_tight_fit"] = {
                split: old[split] for split in
                ("training", "evaluation_score", "evaluation_included", "per_order")}
            case["evaluation_gain_change_percentage_points"] = 100*(
                case["evaluation_score"]["incremental_gain"]
                - old["evaluation_score"]["incremental_gain"])


def verify_baselines(result):
    """Check independently refitted baselines against the saved fit experiment."""
    fit = json.loads((OUT / "relaxed_strength_fit.json").read_text())
    result["selected_case"] = fit["selected_case"]
    result["selection_criterion"] = fit["selection"]["criterion"]
    with np.load(OUT / "relaxed_strength_fit.npz") as saved:
        for setting in result["settings"].values():
            name = f"soft_prior{setting['prior_scale']:g}"
            difference = np.asarray(setting["baseline_theta"])-saved[f"theta_{name}"]
            np.testing.assert_allclose(difference, 0, rtol=0, atol=1e-9)
            setting["baseline_theta_saved_max_difference"] = float(np.max(np.abs(difference)))
            setting["selected_by_training_cv"] = name == fit["selected_case"]


def write_outputs(result, previous):
    """Save the full response audit and a compact paired comparison."""
    (OUT / "relaxed_strength_planet.json").write_text(json.dumps(result, indent=2)+"\n")
    old_cases = previous["cases"]
    labels = [f"FWHM {c['fwhm_kms']:g}; Kp {c['kp_kms']:g}\ndepth {c['depth']:g}"
              for c in old_cases]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True, sharex=True)
    for ax, split, title in zip(axes, ["training", "evaluation_score"],
                                ["Training spectra", "Held-out scoring pixels"]):
        ax.plot(x, [100*c[split]["incremental_gain"] for c in old_cases],
                "o--", color="0.45", label="Previous: hard cap 1, prior scale 1")
        for name, setting in result["settings"].items():
            ax.plot(x, [100*c[split]["incremental_gain"] for c in setting["cases"]],
                    "o-", label=f"{name}: positivity bound only")
        ax.axhline(100, color="0.7", ls=":", lw=1)
        ax.set(ylabel="Response relative to polynomial only (%)", title=title)
        ax.legend(fontsize=8)
    axes[-1].set(xticks=x, xticklabels=labels)
    axes[-1].tick_params(axis="x", labelsize=8)
    fig.savefig(OUT / "relaxed_strength_planet.png", dpi=160)
    plt.close(fig)

    lines = [
        "Artificial planet-line response with relaxed shared strength priors",
        "===================================================================", "",
        "These are paired injection/refit tests using exactly the previous seven",
        "artificial Gaussian forests. Only the five training exposures determine",
        "the shared strength parameters. The four evaluation exposures are predicted",
        "without fitting. The same degree-five residual filter is applied to both",
        "injected-minus-uninjected spectra and signal templates.", "",
        "Gain = dot(Q signal, output change) / dot(Q signal, Q signal).",
        "100% means no additional projected-amplitude change relative to polynomial",
        "filtering alone, not full-pipeline preservation or detection significance.", "",
        "FWHM  Kp  Depth  Prior scale  Train gain (%)  Evaluation gain (%)  Old gain (%)  Change (pp)",
    ]
    for name, setting in result["settings"].items():
        for case in setting["cases"]:
            lines.append(
                f"{case['fwhm_kms']:4g} {case['kp_kms']:3g} {case['depth']:7g} "
                f"{setting['prior_scale']:11g} {100*case['training']['incremental_gain']:15.5f} "
                f"{100*case['evaluation_score']['incremental_gain']:20.5f} "
                f"{100*case['previous_tight_fit']['evaluation_score']['incremental_gain']:13.5f} "
                f"{case['evaluation_gain_change_percentage_points']:12.5f}")
    lines += ["", f"Selected by conditional training CV: {result['selected_case']}",
              "The injection results do not select the prior width.", ""]
    for name, setting in result["settings"].items():
        moving = setting["cases"][:4]
        gains = [100*c["evaluation_score"]["incremental_gain"] for c in moving]
        correlations = [c["evaluation_score"]["response_correlation"] for c in moving]
        order_gains = [100*c["per_order"][o]["evaluation_score"]["incremental_gain"]
                       for c in moving for o in c["per_order"]]
        stationary = 100*setting["cases"][4]["evaluation_score"]["incremental_gain"]
        lines += [f"{name}: moving evaluation gains {min(gains):.3f}-{max(gains):.3f}%;",
                  f"  per-order gains {min(order_gains):.3f}-{max(order_gains):.3f}%;",
                  f"  pooled shape correlations {min(correlations):.5f}-{max(correlations):.5f};",
                  f"  stationary evaluation gain {stationary:.3f}%."]
    lines += ["", "Near-unit signed amplitude does not imply unchanged line shape.",
              "Relaxing the strength prior increases the ability to absorb stationary",
              "signals and the shape change of the moving controls.", "",
              "Interpretation and limitations", "------------------------------",
              "Bounds require nonnegative strengths only. Prior widths are modeling",
              "choices; HITRAN uncertainty-bin upper values are not Gaussian sigmas.",
              "Kp=0 is a stationary control; Kp=100/200 km/s are hypothetical motions.",
              "The model's TP, stellar spectrum, abundances, wavelength solution, IP",
              "and original continuum are fixed. The tests isolate the additional",
              "shared-strength-fit response and do not refit the earlier pipeline.",
              "These signals are added at observed resolution with approximate",
              "stellar/telluric attenuation, not physical planet spectra before IP.",
              "A gain above 100% is a signed projection effect, not extra information.",
              "The sign and smaller-depth controls probe local response stability.",
              "Parameter vectors, per-order responses, convergence diagnostics and",
              "input hashes are preserved in relaxed_strength_planet.json.", ""]
    (OUT / "RELAXED_STRENGTH_PLANET_REPORT.txt").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-scale", type=float, nargs="+", required=True)
    args = parser.parse_args()
    if any(scale <= 0 for scale in args.prior_scale):
        parser.error("Prior scales must be positive")
    from fit_line_strength import build_context

    context = build_context()
    previous_path = OUT / "strength_planet_injection.json"
    previous = json.loads(previous_path.read_text())
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), settings={})
    for scale in dict.fromkeys(args.prior_scale):
        result["settings"][f"soft{scale:g}"] = run_setting(context, scale)
    add_comparison(result, previous)
    verify_baselines(result)
    paths = [Path(__file__), ROOT / "check_strength_planet.py", ROOT / "fit_relaxed_strength.py",
             ROOT / "fit_line_strength.py", ROOT / "line_strength_support.py",
             ROOT / "smooth_residuals.py", ROOT / "check_smooth_signal.py",
             ROOT / "data" / "20201214_info.csv", previous_path,
             OUT / "relaxed_strength_fit.json", OUT / "relaxed_strength_fit.npz",
             OUT / "relaxed_strength_selection.json"] + context["source_files"]
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}
    write_outputs(result, previous)


if __name__ == "__main__":
    main()
