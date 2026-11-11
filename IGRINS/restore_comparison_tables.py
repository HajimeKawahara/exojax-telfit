"""Rebuild configuration and precision comparisons without refitting models."""

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
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCES = set()


def load_json(name):
    SOURCES.add(OUT/name)
    return json.loads((OUT/name).read_text())


def load_arrays(name):
    SOURCES.add(OUT/name)
    return np.load(OUT/name)


def main():
    current = load_arrays("adopted_blase_telluric_fit.npz")
    old = load_arrays("adopted_telluric_fit.npz")
    manifest = load_json("adopted_telluric_model.json")
    assert hashlib.sha256((ROOT/manifest["product"]).read_bytes()).hexdigest() == manifest["product_sha256"]
    wave, score, included, observed = [current[k] for k in
                                      ["wavelength_um", "score_mask", "included_mask", "y_over_c"]]
    for key in ["orders", "exposures", "wavelength_um", "score_mask", "included_mask", "y_over_c"]:
        np.testing.assert_array_equal(current[key], old[key])
    residuals, rows, checks = {}, {}, []

    def project(array):
        answer = []
        for w, mask, values in zip(wave, included, array):
            x = 2*(np.log(w)-np.log(w).min())/np.ptp(np.log(w))-1
            basis = np.polynomial.chebyshev.chebvander(x, 5)
            q, r = np.linalg.qr(basis[mask], mode="reduced")
            coefficients = np.linalg.solve(r, q.T@values[:, mask].T)
            answer.append(values-(basis@coefficients).T)
        return np.asarray(answer)

    def stats(residual, mask=score, order_indices=None):
        indices = np.arange(3) if order_indices is None else np.asarray(order_indices)
        values = residual[indices, 5:]
        keep = np.broadcast_to(mask[indices, None], values.shape)
        selected = values[keep]
        return dict(n_samples=int(selected.size), rms_percent=float(100*np.sqrt(np.mean(selected**2))),
                    sse=float(np.sum(selected**2)))

    def add(key, label, star, group, file, component_key, expected=None, q5=True):
        data = load_arrays(file)
        for name in ["orders", "exposures", "wavelength_um", "score_mask", "included_mask", "y_over_c"]:
            if name in data:
                np.testing.assert_array_equal(data[name], current[name])
        if component_key.startswith("residual"):
            residual = data[component_key]
        else:
            residual = observed-data[component_key]
            if q5:
                residual = project(residual)
        metric = stats(residual)
        if expected is not None:
            np.testing.assert_allclose(metric["rms_percent"], expected, atol=1e-10, rtol=0)
        residuals[key] = residual
        rows[key] = dict(case=key, label=label, stellar_model=star, group=group, q5=q5,
                         **metric, h_score=stats(residual, order_indices=[1, 2]),
                         per_order={str(order): stats(residual, order_indices=[i])
                                    for i, order in enumerate(current["orders"])},
                         source=dict(file=file, key=component_key), stored_rms_verified=expected is not None)
        checks.append(dict(case=key, matched_data_and_masks=True, n_samples=metric["n_samples"],
                           rms_percent=metric["rms_percent"]))

    legacy = "Fixed PHOENIX"
    add("reference", "Original IP + weak-line prior", legacy, "reference", "adopted_telluric_fit.npz", "component", .7746078018294396)
    add("current", "Current provisional: blase + telluric refit", "PHOENIX + blase", "current", "adopted_blase_telluric_fit.npz", "component", manifest["metrics"]["evaluation_score"]["rms_percent"])
    stage = load_json("current_blase_figures.json")
    add("stellar_only", "Blase stellar calibration only", "PHOENIX + blase", "current", "current_blase_stellar_stage.npz", "component", stage["stellar_only_stage_metrics"]["evaluation_score"]["rms_percent"])

    line = load_json("line_strength_fit.json")
    smooth = load_json("smooth_comparison.json")
    add("no_smooth", "Before smooth removal; original strengths", legacy, "historical", "line_strength_fit.npz", "component_baseline", smooth["final"]["none"]["all_included"]["continuum_rms_percent"], q5=False)
    add("q5_only", "Degree-five removal; original strengths", legacy, "historical", "line_strength_fit.npz", "component_baseline", line["baseline"]["evaluation_score"]["rms_percent"])
    strength_labels = dict(primary="HITRAN-scale capped intensity corrections",
                           half_prior_width="Intensity caps, half-width prior",
                           double_prior_width="Intensity caps, double-width prior")
    for case, setting in line["cases"].items():
        add("strength_"+case, strength_labels[case], legacy, "historical", "line_strength_fit.npz", "component_"+case, setting["metrics"]["evaluation_score"]["rms_percent"])
    relaxed = load_json("relaxed_strength_fit.json")
    for case, setting in relaxed["cases"].items():
        add("relaxed_"+case, "Intensity prior: "+case, legacy, "historical", "relaxed_strength_fit.npz", "component_"+case, setting["metrics"]["evaluation_score"]["rms_percent"])
    rows["relaxed_soft_prior10"]["label"] = "Uniform intensity-prior width x10"
    quadratic = load_json("quadratic_strength_fit.json")
    for case, setting in quadratic["cases"].items():
        add("ip_"+case, "IP/prior: "+case, legacy, "historical", "quadratic_strength_fit.npz", "component_"+case, setting["metrics"]["evaluation_score"]["rms_percent"])
    summary = load_json("residual_experiment_summary.json")
    shape_labels = dict(strength_control="Strength-only repeat control", width_prior1="H2O widths, prior x1",
                        width_prior3="H2O widths, prior x3", center_prior1="H2O centers, prior x1",
                        center_prior3="H2O centers, prior x3", joint_prior1="H2O widths + centers, prior x1",
                        joint_prior3="H2O widths + centers, prior x3")
    for case, label in shape_labels.items():
        add(case, label, legacy, "historical", "line_shape_fit.npz", "component_"+case, summary["metrics"][case]["rms_percent"])
    for case, label in [("abundance_consistent_self", "Abundance-consistent self broadening"),
                        ("profile_fixed_strengths", "Vertical profile, strengths fixed (local)"),
                        ("profile_joint_strengths", "Vertical profile + strengths (local)")]:
        add(case, label, legacy, "historical", "vertical_profile_experiment.npz", "component_"+case, summary["metrics"][case]["rms_percent"])
    bg = load_json("background_wavelength_fit.json")["selection"]["selected_case"]
    add("background_fixed", "Background/wavelength, strengths fixed (local)", legacy, "historical", "background_wavelength_fit.npz", "residual_"+bg, summary["metrics"]["background_wavelength_selected"]["rms_percent"])
    add("background_joint", "Background/wavelength + strengths (local)", legacy, "historical", "background_wavelength_strength_profile.npz", "component_local_joint_background_wavelength_strength", summary["metrics"]["background_wavelength_joint_strengths"]["rms_percent"])
    unknown = load_json("unknown_strength_fit.json")
    add("unknown30", "Eight CO2 strengths freed, 30% prior", legacy, "historical", "unknown_strength_fit.npz", "component_joint30", unknown["cases"]["joint30"]["metrics"]["evaluation_score"]["rms_percent"])
    tp = load_json("transformerpayne_elements_fit.json")
    for case, label in [("global_refit", "TransformerPayne global labels + strengths"),
                        ("elements_refit", "TransformerPayne 15 elements + strengths")]:
        add("tp_"+case, label, "TP in H; adopted blase in K", "tp", "transformerpayne_elements_fit.npz", "component_"+case, tp["metrics"][case]["evaluation"]["score"]["hybrid_all_orders"]["rms_percent"])

    for row in rows.values():
        row["squared_residual_reduction_vs_phoenix_percent"] = 100*(1-row["sse"]/rows["reference"]["sse"])
    primary = ["reference", "ip_quadratic_gaussian_contrast10", "ip_quadratic_mixture_contrast10",
               "width_prior3", "center_prior3", "joint_prior3", "profile_joint_strengths", "background_joint",
               "abundance_consistent_self", "unknown30", "stellar_only", "current", "tp_global_refit", "tp_elements_refit"]
    rows["ip_quadratic_gaussian_contrast10"]["label"] = "Quadratic Gaussian IP + strengths"
    rows["ip_quadratic_mixture_contrast10"]["label"] = "Quadratic two-Gaussian IP width + strengths"

    # Frozen original molecular classification and earlier training-selected intervals.
    category = load_arrays("nonstellar_spectral_selection.npz")["category"]
    categories = []
    for i, label in enumerate(["H2O-dominant", "CO2-dominant", "Mixed", "Weak modeled absorption"]):
        masks = score & (category == i)
        categories.append(dict(category=label, score_wavelengths=int(masks.sum()),
                               reference=stats(residuals["reference"], masks), current=stats(residuals["current"], masks)))
    for row in categories:
        for name, key in [("reference", "reference"), ("current", "current")]:
            row[name]["squared_residual_fraction_percent"] = 100*row[name]["sse"]/rows[key]["sse"]
    regions = load_json("poly5_spectral_budget.json")["region_sets"]["post_filter_training_selection"]
    hotspots, union = [], np.zeros_like(score)
    for region in regions:
        b, (start, stop) = region["order_index"], region["array_interval"]
        mask = np.zeros_like(score); mask[b, start:stop] = score[b, start:stop]
        if not mask.any():
            continue
        union |= mask
        record = dict(order=int(current["orders"][b]), wavelength_um=region["peak_wavelength_um"],
                      array_interval=[start, stop], score_wavelengths=int(mask.sum()),
                      reference=stats(residuals["reference"], mask), current=stats(residuals["current"], mask))
        for key in ["reference", "current"]:
            record[key]["squared_residual_fraction_percent"] = 100*record[key]["sse"]/rows[key]["sse"]
        hotspots.append(record)
    hotspot_total = dict(score_wavelengths=int(union.sum()),
                         reference=stats(residuals["reference"], union), current=stats(residuals["current"], union))
    for key in ["reference", "current"]:
        hotspot_total[key]["squared_residual_fraction_percent"] = 100*hotspot_total[key]["sse"]/rows[key]["sse"]

    shape = load_json("line_shape_fit.json")
    cv_rows = [dict(case=k, rms_percent=v["rms_percent"]) for k, v in shape["selection"]["cross_validation"].items()]
    precision, precision_rows = load_json("line_shape_precision.json"), []
    for name, setting in precision["settings"].items():
        assert setting["null_precision_passed"]
        for case in setting["water"]["cases"]:
            previous = case["previous_default_precision"]["evaluation_score"]
            measured = case["evaluation_score"]
            precision_rows.append(dict(case=name, kp_kms=case["kp_kms"], depth=case["depth"],
                                       default_gain_percent=100*previous["incremental_gain"],
                                       tight_gain_percent=100*measured["incremental_gain"],
                                       tight_correlation=measured["response_correlation"]))
    tightened = list(precision_rows)
    alternatives = load_json("line_shape_precision_alternatives.json")
    for name, setting in alternatives["settings"].items():
        assert setting["null_precision_passed"]
        for case in setting["water"]["cases"]:
            tightened.append(dict(case=name, kp_kms=case["kp_kms"], depth=case["depth"],
                                   tight_gain_percent=100*case["evaluation_score"]["incremental_gain"],
                                   tight_correlation=case["evaluation_score"]["response_correlation"]))
    smooth_rows = []
    data = load_arrays("smooth_results.npz")
    np.testing.assert_array_equal(data["score_mask"], score)
    np.testing.assert_array_equal(data["exposures"], current["exposures"][5:])
    for key, entry in smooth["final"].items():
        for fit_scope in ["calibration_only", "all_included"]:
            residual = data["residual_continuum"]-data[f"correction_{key}_{fit_scope}"]
            masked = np.broadcast_to(score[:, None], residual.shape)
            value = float(100*np.sqrt(np.mean(residual[masked]**2)))
            np.testing.assert_allclose(value, entry[fit_scope]["continuum_rms_percent"], atol=1e-10, rtol=0)
            smooth_rows.append(dict(case=key, fit_scope=fit_scope, rms_percent=value))

    report = ["## Configuration comparisons and numerical precision", "",
              "These tables retain the measured comparisons from the original report and add the current stellar-model experiments. Configuration evaluation RMS values use the same four evaluation frames and 2,092 scoring samples, in percent of the original continuum; H-only and training-CV values are labeled separately. Historical fixed-PHOENIX cases have **not** been rerun with the new stellar correction; their stellar-model column is part of the comparison, not an implied common baseline.", "",
              "### Physical-model and stellar-model comparisons", "",
              "Every row uses degree-five filtering. Squared-residual reductions are relative to the fixed-PHOENIX weak-line reference; they do not uniquely attribute an error to any physical cause.", "",
              "| Configuration | Stellar model | Evaluation RMS | Squared-residual reduction |",
              "| --- | --- | ---: | ---: |"]
    for key in primary:
        row = rows[key]
        report.append(f"| {row['label']} | {row['stellar_model']} | {row['rms_percent']:.6f}% | {row['squared_residual_reduction_vs_phoenix_percent']:+.2f}% |")
    report += ["", "The original IP experiments, H2O-width/center experiments, profile/background changes and abundance-consistent self-broadening remain visible here. The line-shape cases fit 35 selected H2O features and the original 269 strengths; CO2 shapes and temperature exponents are fixed. The vertical-profile and background/strength cases are local proposals, not converged flexible-atmosphere retrievals. Prior x3 means three times the adopted line-shape regularization scale. None of these historical physical alternatives is silently combined with the current blasé model.", "",
               "### TransformerPayne: compare the covered H orders separately", "",
               "The public checkpoint covers the H windows but not order13 near 2.07 microns. Its three-order numbers are therefore **hybrids**: TransformerPayne in orders34/40 and the adopted blasé model in order13. The 15-element experiment improves its tied-abundance control but does not beat the current model.", "",
               "| Stellar case | H-only score RMS (1,204 samples) | Three-order/hybrid RMS (2,092 samples) | Order13 | Order34 | Order40 |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for key in ["reference", "current", "tp_global_refit", "tp_elements_refit"]:
        row = rows[key]
        report.append(f"| {row['label']} | {row['h_score']['rms_percent']:.6f}% | {row['rms_percent']:.6f}% | " + " | ".join(f"{row['per_order'][str(o)]['rms_percent']:.6f}%" for o in current["orders"])+" |")
    report += ["", "The TransformerPayne fits calibrate six shared global/kinematic labels, then add fifteen independent elemental abundances in the expanded case; 119 H-order telluric strengths are refitted afterward. Order13 remains unchanged. These are exploratory conditional labels, not calibrated stellar abundances. The current provisional selection remains blasé.", "",
               "### Smooth-removal and strength-prior progression", "",
               "This restores the earlier progression, including the case without smooth removal. It is not a sequence of additive improvements that can all be applied again to the current fit.", "",
               "| Configuration | Stellar model | Evaluation RMS | Smooth operation |",
               "| --- | --- | ---: | --- |"]
    for key in ["no_smooth", "q5_only", "strength_primary", "relaxed_soft_prior10", "reference", "current"]:
        row = rows[key]
        report.append(f"| {row['label']} | {row['stellar_model']} | {row['rms_percent']:.6f}% | {'Q5' if row['q5'] else 'None'} |")
    report += ["", "The uniformly widened intensity-prior case is retained as a comparison; weak-line suppression was chosen deliberately even though it slightly increases RMS. The primary capped-intensity fit treats the HITRAN interval scale as a bound. Those intervals are not calibrated Gaussian standard deviations.", "",
               "<details>", "<summary>All saved intensity-prior and IP-prior settings</summary>", "",
               "All cases in this expanded table use fixed PHOENIX and Q5. Case names retain their original reproducible identifiers.", "",
               "| Experiment | Setting | Evaluation RMS |", "| --- | --- | ---: |"]
    for prefix, label, cases in [("strength_", "HITRAN capped strengths", line["cases"]),
                                  ("relaxed_", "Relaxed strengths", relaxed["cases"]),
                                  ("ip_", "IP and weak-line prior", quadratic["cases"])]:
        for case in cases:
            row = rows[prefix+case]
            report.append(f"| {label} | `{case}` | {row['rms_percent']:.6f}% |")
    report += ["", "</details>", "", "<details>", "<summary>All saved polynomial and GP smooth-removal settings</summary>", "",
               "These are original-strength, fixed-PHOENIX controls before the later intensity corrections. Calibration-only and all-included columns describe which pixels estimate the smooth correction; both are scored on the same held-out-frame scoring pixels.", "",
               "| Smooth model | Fit on calibration pixels only | Fit on all included pixels |", "| --- | ---: | ---: |"]
    for key in smooth["final"]:
        values = [r["rms_percent"] for r in smooth_rows if r["case"] == key]
        report.append(f"| `{key}` | {values[0]:.6f}% | {values[1]:.6f}% |")
    report += ["", "GP names give correlation length in km/s and amplitude in continuum units. Lower RMS alone does not justify a more aggressive smooth filter; planet templates must undergo the same operation.", "", "</details>", "",
               "### H2O line-shape freedom and training-frame validation", "",
               "All cases here keep PHOENIX fixed and refit the original strengths. The prior-x1 and prior-x3 alternatives are retained. The selected historical line-shape case was `joint_prior3`; this remains an experiment, not the current adopted stellar model.", "",
               "| Line-shape case | Training-frame CV RMS | Evaluation RMS |", "| --- | ---: | ---: |"]
    for row in cv_rows:
        report.append(f"| `{row['case']}` | {row['rms_percent']:.6f}% | {rows[row['case']]['rms_percent']:.6f}% |")
    report += ["", "### Numerical precision: default versus tightened optimization", "",
               "This is a numerical-accuracy test with the **same physical model, data, priors and masks**, not a new physical configuration. The original solver used ftol=1e-8, xtol=1e-9, gtol=1e-7, max_nfev=150. Tightened tolerances were ftol=xtol=1e-12, gtol=1e-8, max_nfev=500, with a zero-injection check. The older default-precision weak-water response is superseded by the tightened result.", "",
               "| Fixed-PHOENIX model | Hypothetical Kp (km/s) | Injected depth | Default response | Tight response | Tight shape correlation |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in precision_rows:
        report.append(f"| `{row['case']}` | {row['kp_kms']:g} | {row['depth']:g} | {row['default_gain_percent']:.6f}% | {row['tight_gain_percent']:.6f}% | {row['tight_correlation']:.6f} |")
    report += ["", "Negative injected depth is the sign-reversed control. These artificial H2O-weighted patterns differ from the Gaussian forests used in the current blasé checks, so their absolute response coefficients must not be ranked as if they were the same injected planet spectrum.", "",
               "| Tight-precision moving-H2O case | Depth | Evaluation response | Shape correlation |", "| --- | ---: | ---: | ---: |"]
    for row in tightened:
        if row["kp_kms"] == 200 and row["depth"] > 0:
            report.append(f"| `{row['case']}` | {row['depth']:g} | {row['tight_gain_percent']:.6f}% | {row['tight_correlation']:.6f} |")
    report += ["", "These response coefficients are relative to Q5 alone. A lower fitting RMS is not sufficient if additional freedom absorbs or distorts the injected pattern.", "",
               "### Where the residual lies: frozen molecular classification", "",
               "The same original-model molecular classification is used for both columns. This measures where squared residual accumulates; it does not assign its physical cause to a particular molecule or database.", "",
               "| Region | Scoring wavelengths | Fixed-PHOENIX fraction of squared residual | Current blasé fraction | Current local RMS |",
               "| --- | ---: | ---: | ---: | ---: |"]
    for row in categories:
        report.append(f"| {row['category']} | {row['score_wavelengths']} | {row['reference']['squared_residual_fraction_percent']:.2f}% | {row['current']['squared_residual_fraction_percent']:.2f}% | {row['current']['rms_percent']:.6f}% |")
    report += ["", "### Earlier selected residual intervals, updated with the current model", "",
               "The interval definitions remain frozen from the earlier training analysis. No pixels are removed by this table.", "",
               "| Order | Earlier peak (micron) | Scoring wavelengths | Fixed-PHOENIX local RMS | Current local RMS | Current share of total squared residual |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in hotspots:
        report.append(f"| {row['order']} | {row['wavelength_um']:.9f} | {row['score_wavelengths']} | {row['reference']['rms_percent']:.6f}% | {row['current']['rms_percent']:.6f}% | {row['current']['squared_residual_fraction_percent']:.2f}% |")
    report += ["", f"Together these {hotspot_total['score_wavelengths']} scoring wavelengths contain {hotspot_total['reference']['squared_residual_fraction_percent']:.2f}% of the reference squared residual and **{hotspot_total['current']['squared_residual_fraction_percent']:.2f}%** of the current squared residual. Their statistical concentration is not a unique line-list/atmosphere/instrument attribution.", ""]
    (OUT/"restored_comparison_tables.txt").write_text("\n".join(report))

    # Restore the original horizontal RMS/reduction layout, with explicit stellar-model groups.
    colors = {"reference": "#2166ac", "historical": "#999999", "current": "#238b45", "tp": "#d98c20"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 10.8), sharey=True, gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.37, right=.97, top=.84, bottom=.15, wspace=.35)
    fig.suptitle("Measured configurations: residual RMS and squared-residual reduction", y=.97, fontsize=15, weight="bold")
    fig.text(.5, .933, "Same 523 scoring wavelengths x 4 evaluation frames; Q5 in every row", ha="center", fontsize=11)
    fig.legend(handles=[Patch(color=colors[k], label=v) for k, v in [("reference", "Fixed-PHOENIX reference"), ("historical", "Fixed-PHOENIX experiments"), ("current", "PHOENIX + blase"), ("tp", "TP in H; adopted blase in K")]], loc="upper center", bbox_to_anchor=(.52, .912), ncol=2, frameon=False, fontsize=10)
    for i, key in enumerate(primary):
        row = rows[key]; color = colors[row["group"]]
        value, reduction = row["rms_percent"], row["squared_residual_reduction_vs_phoenix_percent"]
        axes[0].barh(i, value, color=color, height=.65)
        axes[1].barh(i, reduction, color=color, height=.65)
        axes[0].text(value+.012, i, f"{value:.4f}", va="center", fontsize=9)
        axes[1].text(reduction+(1 if reduction >= 0 else -1), i, f"{reduction:+.2f}", va="center", ha="left" if reduction >= 0 else "right", fontsize=9)
    axes[0].set_yticks(np.arange(len(primary)), [rows[k]["label"] for k in primary], fontsize=10)
    axes[0].invert_yaxis(); axes[0].set_xlim(0, 1.07); axes[1].set_xlim(-60, 22)
    axes[0].set_xlabel("Evaluation RMS (% of original continuum)")
    axes[1].set_xlabel("Squared-residual reduction vs PHOENIX (%)")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="x", color="#dddddd", lw=.6)
        ax.set_axisbelow(True); ax.tick_params(axis="y", length=0)
    axes[1].axvline(0, color="#555555", lw=.7)
    fig.text(.5, .065, "Historical physical experiments have NOT been rerun with the new stellar model.\nTransformerPayne changes H orders34/40 only; K order13 keeps the current blase model.\nProfile/background proposals are local fits. Lower RMS alone does not establish planet-signal preservation.", ha="center", fontsize=10.5)
    figure = OUT/"restored_configuration_comparison.png"
    fig.savefig(figure, dpi=170, facecolor="white"); plt.close(fig)

    original_report = OUT/"PREMASK_ISSUE_REPORT.txt"
    SOURCES.add(original_report); SOURCES.add(Path(__file__)); SOURCES.add(ROOT/"fit_line_shapes.py")
    inventory = [line for line in original_report.read_text().splitlines() if line.startswith("|")]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), no_refits_performed=True,
                  adopted_product=manifest["product"], adopted_product_sha256=manifest["product_sha256"],
                  units="Percent of original fitted continuum; same 2092 evaluation scoring samples",
                  original_report_table_rows=inventory, original_comparison_cases_retained=True,
                  rows=rows, primary_comparison_cases=primary, line_shape_cross_validation=cv_rows,
                  solver_precision=precision["solver"], default_vs_tight_precision=precision_rows,
                  all_tightened_water_controls=tightened, smooth_controls=smooth_rows,
                  molecular_categories=categories, frozen_hotspots=hotspots, hotspot_union=hotspot_total,
                  validation=checks, figure=figure.name,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(SOURCES)},
                  figure_sha256=hashlib.sha256(figure.read_bytes()).hexdigest())
    (OUT/"restored_comparison_tables.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(configurations=len(rows), primary_cases=len(primary), original_table_rows=len(inventory),
                          current_rms=rows["current"]["rms_percent"], current_hotspot_fraction=hotspot_total["current"]["squared_residual_fraction_percent"],
                          figures=[figure.name]), indent=2))


if __name__ == "__main__":
    main()
