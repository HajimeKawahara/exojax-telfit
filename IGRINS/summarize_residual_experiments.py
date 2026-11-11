"""Summarize the frozen-IP residual experiments on common evaluation pixels."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def statistics(residual, mask):
    samples = residual[:, 5:]
    keep = np.broadcast_to(mask[:, None], samples.shape)
    common = np.broadcast_to(samples.mean(axis=1)[:, None], samples.shape)
    return dict(n_samples=int(keep.sum()), rms_percent=float(100*np.sqrt(np.mean(samples[keep]**2))),
                common_rms_percent=float(100*np.sqrt(np.mean(common[keep]**2))),
                temporal_rms_percent=float(100*np.sqrt(np.mean((samples-common)[keep]**2))),
                sse=float(np.sum(samples[keep]**2)))


def main():
    sources = [Path(__file__), OUT/"line_shape_fit.json", OUT/"line_shape_fit.npz",
               OUT/"vertical_profile_experiment.json", OUT/"vertical_profile_experiment.npz",
               OUT/"background_wavelength_fit.json", OUT/"background_wavelength_fit.npz",
               OUT/"background_wavelength_strength_profile.json", OUT/"background_wavelength_strength_profile.npz",
               OUT/"adopted_telluric_fit.npz", OUT/"nonstellar_spectral_selection.npz",
               OUT/"poly5_spectral_budget.json"]
    shape = json.loads((OUT/"line_shape_fit.json").read_text())
    atmosphere = json.loads((OUT/"vertical_profile_experiment.json").read_text())
    background = json.loads((OUT/"background_wavelength_fit.json").read_text())
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        adopted = {key: saved[key] for key in saved.files}
    wave, included, score = [adopted[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    with np.load(OUT/"nonstellar_spectral_selection.npz") as saved:
        category = saved["category"]

    def project(values):
        result = []
        for row, w, mask in zip(values, wave, included):
            x = 2*(np.log(w)-np.log(w).min())/np.ptp(np.log(w))-1
            basis = np.polynomial.chebyshev.chebvander(x, 5)
            q, r = np.linalg.qr(basis[mask], mode="reduced")
            coefficient = np.linalg.solve(r, q.T @ row[:, mask].T)
            result.append(row-(basis @ coefficient).T)
        return np.array(result)

    residuals = {"adopted": adopted["residual_filtered"]}
    with np.load(OUT/"line_shape_fit.npz") as saved:
        for name in shape["cases"]:
            residuals[name] = project(adopted["y_over_c"]-saved[f"component_{name}"])
        selected_theta = saved[f"theta_{shape['selection']['selected_case']}"]
    with np.load(OUT/"vertical_profile_experiment.npz") as saved:
        for name in ["abundance_consistent_self", "profile_fixed_strengths", "profile_joint_strengths"]:
            residuals[name] = project(adopted["y_over_c"]-saved[f"component_{name}"])
    background_name = background["selection"]["selected_case"]
    with np.load(OUT/"background_wavelength_fit.npz") as saved:
        residuals["background_wavelength_selected"] = saved[f"residual_{background_name}"]
    with np.load(OUT/"background_wavelength_strength_profile.npz") as saved:
        residuals["background_wavelength_joint_strengths"] = project(
            adopted["y_over_c"]-saved["component_local_joint_background_wavelength_strength"])
    baseline = statistics(residuals["adopted"], score)
    summaries = {}
    regions = json.loads((OUT/"poly5_spectral_budget.json").read_text())["region_sets"]["post_filter_training_selection"]
    for name, residual in residuals.items():
        result = statistics(residual, score)
        result["squared_residual_reduction_percent"] = 100*(1-result["sse"]/baseline["sse"])
        result["per_order"] = {str(int(order)): statistics(residual[b:b+1], score[b:b+1])
                               for b, order in enumerate(adopted["orders"])}
        result["categories"] = {label: statistics(residual, score & (category == i))
            for i, label in enumerate(["H2O-dominant", "CO2-dominant", "Mixed", "Weak absorption"])}
        result["regions"] = []
        for region in regions:
            b, (start, stop) = region["order_index"], region["array_interval"]
            mask = np.zeros_like(score)
            mask[b, start:stop] = score[b, start:stop]
            if mask.any():
                result["regions"].append(dict(order=int(adopted["orders"][b]),
                    peak_wavelength_um=region["peak_wavelength_um"], metrics=statistics(residual, mask)))
        summaries[name] = result
    corrections = []
    for record in shape["shape_records"]:
        width, center = selected_theta[[record["width_parameter_index"], record["center_parameter_index"]]]
        corrections.append(dict(order=record["order"], nu_cm1=record["nu"],
            original_contrast=record["original_contrast"], gamma_factor=float(np.exp(width)),
            center_shift_kms=float(center), width_prior_units=float(width/record["log_width_prior"]),
            center_prior_units=float(center/record["center_prior_kms"])))
    chosen = shape["selection"]["selected_case"]
    planet = {}
    for filename in ["line_shape_precision.json", "line_shape_precision_alternatives.json"]:
        path = OUT/filename
        if not path.exists():
            continue
        sources.append(path)
        injection = json.loads(path.read_text())
        for name, setting in injection["settings"].items():
            assert setting["null_precision_passed"], name
            moving = [case for case in setting["water"]["cases"]
                      if case["kp_kms"] == 200. and case["depth"] > 0]
            planet[name] = dict(null_precision_passed=True, source=filename,
                cases=[dict(depth=case["depth"], fwhm_kms=case["fwhm_kms"], kp_kms=case["kp_kms"],
                            evaluation_gain_percent=100*case["evaluation_score"]["incremental_gain"],
                            evaluation_shape_correlation=case["evaluation_score"]["response_correlation"],
                            training_gain_percent=100*case["training"]["incremental_gain"])
                       for case in moving])
    caveats = [
        "All RMS values use the same four evaluation exposures, stellar-screened score mask, and degree-five projection.",
        "Line-shape cases are fully iterated nonlinear fits of the 269 strengths and selected H2O widths/centers; n_air stays fixed.",
        "Atmospheric cases are one-step local fits followed by exact nonlinear spectra, not fully iterated retrievals.",
        "Background/wavelength cases freeze adopted strengths; their additional coefficients are training-only fits.",
        "The separate background_wavelength_joint_strengths case locally profiles strengths and evaluates their nonlinear forward model plus a frozen additive design.",
        "Cross-validation is conditional on earlier nuisance fits; these spectra have been examined in earlier experiments.",
        "A reduction in squared residual is not a unique physical attribution to HITRAN, atmosphere, or instrument.",
        "Individual width and center corrections may compensate blends and other fixed-model errors.",
        "The user-adopted fixed_contrast10 product is unchanged by these experiments.",
        "H2O-pattern gains below use tightened solver tolerances and passed zero-injection tests; they supersede default-precision water gains in line_shape_planet.json.",
        "Artificial water patterns use frozen terrestrial line weights, observed resolution and hypothetical Kp=200 km/s; they are not full planetary spectra or end-to-end preservation estimates.",
    ]
    result = dict(line_shape_selected=chosen, background_selected=background_name,
                  metrics=summaries, selected_line_corrections=corrections, planet_tests=planet, caveats=caveats,
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"residual_experiment_summary.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Residual experiments with the previous IP and weak-line suppression", "",
             "Case | Evaluation RMS (%) | Reduction in squared residual (%)"]
    for name, record in summaries.items():
        lines.append(f"{name} | {record['rms_percent']:.8f} | {record['squared_residual_reduction_percent']:.5f}")
    lines += ["", f"Training-CV line-shape selection: {chosen}",
              f"Training-CV background/wavelength selection: {background_name}", "", "Line-shape CV RMS (%):"]
    lines += [f"{name}: {entry['rms_percent']:.8f}" for name, entry in shape["selection"]["cross_validation"].items()]
    lines += ["", "Interpretation:",
              "The joint H2O width/center experiment reduces evaluation squared residual by 6.63%; width-only changes provide most of that gain.",
              "The post-Q5 1.49797082 micron hotspot improves from 2.1549% to 1.4955% local RMS; the 1.57049867 micron hotspot stays at 2.9864%.",
              "Only 35 H2O lines receive shape freedom. The CO2-rich order34 has no fitted H2O shape parameters.",
              "Two widths reach the factor-0.5 bound. Effective corrections can compensate blends, pressure shifts and other fixed-model errors; they are not direct measurements of database errors.",
              "The vertical-profile joint case reaches its lower-temperature amplitude bound and is a local candidate, not a converged nonlinear retrieval.",
              "", "Tight-precision moving H2O pattern tests (relative to Q5 alone):",
              "Case | Injected depth | Evaluation gain (%) | Shape correlation | Training gain (%)"]
    for name, setting in planet.items():
        for case in setting["cases"]:
            lines.append(f"{name} | {case['depth']:g} | {case['evaluation_gain_percent']:.6f} | "
                         f"{case['evaluation_shape_correlation']:.6f} | {case['training_gain_percent']:.6f}")
    lines += ["", "Residual-RMS reduction alone does not establish better planet-search performance. The joint model absorbs and distorts slightly more of the artificial moving H2O pattern than the adopted model. No new physical model is adopted by this experiment."]
    lines += ["", "Caveats:"]+caveats+["", "Detailed categories, local regions and fitted line corrections: residual_experiment_summary.json", ""]
    (OUT/"RESIDUAL_EXPERIMENT_REPORT.txt").write_text("\n".join(lines))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    for b, ax in enumerate(axes):
        for name, label, color in [("adopted", "Adopted fixed IP + weak-line prior", ".6"),
                                   (chosen, f"Line-shape experiment: {chosen}", "C0")]:
            mean = np.where(score[b], residuals[name][b, 5:].mean(axis=0), np.nan)
            ax.plot(wave[b], 100*mean, lw=.8, label=label, color=color)
        ax.set(xlabel="Wavelength (micron)", ylabel="Mean Q5 residual (%)", title=f"Order {adopted['orders'][b]}")
        ax.legend(fontsize=8)
    fig.savefig(OUT/"residual_experiment_comparison.png", dpi=160)
    plt.close(fig)
    print("Line-shape selected:", chosen)
    for name, row in summaries.items():
        print(name, row["rms_percent"], row["squared_residual_reduction_percent"])


if __name__ == "__main__":
    main()
