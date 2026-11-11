"""Summarize smooth-filter RMS and artificial planet-line response together."""
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
DISPLAY = ["none", "poly3", "poly5", "gp_l480_a0.01", "gp_l240_a0.01",
           "gp_l120_a0.01", "gp_l60_a0.01", "gp_l30_a0.01"]
LABELS = ["None", "Polynomial 3", "Polynomial 5", "GP 480 km/s", "GP 240 km/s",
          "GP 120 km/s", "GP 60 km/s", "GP 30 km/s"]


def main():
    comparison = json.loads((OUT / "smooth_comparison.json").read_text())
    injection = json.loads((OUT / "smooth_signal_retention.json").read_text())
    selection = json.loads((OUT / "smooth_selection.json").read_text())
    settings = {item["name"]: item for item in injection["settings"]}
    qualified = [name for name in selection["development_rank"]
                 if settings[name]["qualification"]["passed"]]
    conservative = qualified[0]
    baseline = comparison["final"]["none"]["all_included"]["continuum_rms_percent"]
    rows = []
    for name, label in zip(DISPLAY, LABELS):
        metrics = comparison["final"][name]["all_included"]
        protocol = settings[name]["protocols"]["all_included"]
        isolated = protocol["isolated"]["10.0"]["pooled"]
        forests = [item for item in protocol["forests"] if item["fwhm_kms"] == 10]
        gains = [item["mean_removed_gain"] for item in forests]
        rows.append(dict(name=name, label=label, rms_percent=metrics["continuum_rms_percent"],
                         fixed_rms_percent=metrics["fixed_rms_percent"],
                         rms_reduction_percent=100*(1-metrics["continuum_rms_percent"]/baseline),
                         squared_error_reduction_percent=100*metrics["squared_error_reduction_fraction"],
                         blocked_rms_percent=comparison["final"][name]["calibration_only"]["continuum_rms_percent"],
                         isolated_raw_median=isolated["raw_gain"]["median"],
                         isolated_raw_p05=isolated["raw_gain"]["p05"],
                         isolated_contrast_median=isolated["contrast_gain"]["median"],
                         isolated_contrast_p05=isolated["contrast_gain"]["p05"],
                         forest_mean_removed_min=min(gains), forest_mean_removed_max=max(gains)))
    result = dict(comparison=rows, strict_raw_gate_qualified=qualified,
                  strict_raw_gate_selected=conservative,
                  interpretation="Filtering tradeoff, not a new atmospheric fit or a measured planet detection significance",
                  source_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [OUT / "smooth_comparison.json", OUT / "smooth_signal_retention.json",
                                              OUT / "smooth_selection.json", Path(__file__)]})
    (OUT / "smooth_planet_tradeoff.json").write_text(json.dumps(result, indent=2)+"\n")

    text = ["Smooth nuisance removal for faint planet-line analysis", "="*53, "",
            "The current physical telluric/stellar model is frozen. Each order and",
            "exposure receives an additive smooth correction in original fitted-continuum",
            "units: r=(data-model)/C, g=H r, corrected model=model+C*g, residual=r-g.",
            "The pre-existing continuum is cubic and multiplicative. This additive",
            "correction is a different nuisance model; no atmospheric refit is claimed.", "",
            "Scope: the same three orders, four exposures and 2092 scoring samples as",
            "the preceding residual assessment. The original mask is unchanged.",
            "Production filtering uses all 1467 included wavelengths per exposure,",
            "including the scoring pixels. Its RMS is a descriptive post-filter RMS,",
            "not independent predictive accuracy. The disjoint-block prediction is",
            "reported separately. Original continuum units are fixed for all comparisons.", "",
            "Main production-filter results", "------------------------------",
            "Model              RMS (%)  RMS reduction (%)  Mean-removed forest gain (%)"]
    for row in rows:
        text.append(f"{row['label']:18s} {row['rms_percent']:8.5f} {row['rms_reduction_percent']:18.2f}"
                    f" {100*row['forest_mean_removed_min']:10.2f}-{100*row['forest_mean_removed_max']:.2f}")
    text += ["", "Forest gain uses hypothetical moving Gaussian absorption-line patterns",
             "with FWHM 10 km/s, Kp=0/100/200 km/s and the actual training exposure",
             "phases. These are not measured WASP-77Ab orbital parameters or a physical",
             "planetary molecular template. Gains are signed matched-template responses",
             "after subtracting each order/exposure mean. A constant removed as intended",
             "should not be interpreted as a narrow-line loss. Finite-width and blended",
             "line power can still be lost. Tests at 6 and 20 km/s are saved separately.", "",
             "Polynomial degree five or the 480 km/s GP reduces RMS by about 15%,",
             "and squared error by about 27%-28%. Their 10 km/s synthetic forest",
             "mean-removed gains are about 95%-98%; for 20 km/s they can fall to 91%.",
             "The 120 km/s GP reduces RMS by about 22%, but its 10 km/s forest gain",
             "is 88%-92%. A 30 km/s GP reaches 0.663% RMS while preserving only",
             "56%-59% of that forest contrast, and is not a narrow-line-neutral filter.", "",
             "GP settings", "-----------",
             "K(v,v')=a^2 exp[-(v-v')^2/(2*l^2)], v=c*ln(wavelength/wref).",
             "The displayed GPs use a=0.01 and diagonal regularization sigma=0.0045",
             "in continuum units; l is in km/s. The complete grid also tests a=0.03.",
             "This regularization is not a full calibrated observational covariance.",
             "Kernel length alone does not determine retained line power: amplitude,",
             "pixel density, mask fragmentation and boundaries also matter.", "",
             "Raw amplitude and local contrast are different diagnostics", "---------------------------------------------------------",
             "Model              Isolated raw median/p05 (%)    Local contrast median/p05 (%)"]
    for row in rows:
        text.append(f"{row['label']:18s} {100*row['isolated_raw_median']:9.3f}/{100*row['isolated_raw_p05']:<9.3f}"
                    f" {100*row['isolated_contrast_median']:14.3f}/{100*row['isolated_contrast_p05']:.3f}")
    text += ["", "These isolated-line tests use FWHM10 km/s and positions sampled across",
             "the included mask, with a +/-35 km/s local mean removed only for the",
             "contrast metric. Raw matched gain also penalizes the removed broad pedestal.",
             "The predeclared conservative raw gate was median>=95%, p05>=90%.",
             f"It passes {qualified}; development-only ranking selects '{conservative}'.",
             "This gate has not been relaxed after seeing results. It supports no",
             "automatic smoothing choice under that strict raw-amplitude criterion.",
             "The production tradeoff table instead shows the scientifically relevant",
             "options when low-frequency modes may be discarded, as requested.", "",
             "Independent wavelength-block prediction", "---------------------------------------"]
    for row in rows:
        text.append(f"{row['label']:18s} {row['blocked_rms_percent']:.6f}%")
    text += ["", "Here only calibration blocks determine each exposure's smooth correction.",
             "The lowest-development-RMS unconstrained choice is the 480 km/s, a=.01",
             "GP; its four-exposure RMS changes 1.091696% -> 1.072299%, a modest",
             "1.8% improvement. Its previous-flux-scale RMS is slightly worse,",
             "0.946304% -> 0.950388%, because the weighting differs. The production",
             "RMS reduction should therefore not be described as an improvement in",
             "the atmosphere's predictive model. The blocks span about 160 km/s and",
             "this prediction task differs from filtering observed pixels directly.", "",
             "Application to a planet model", "-----------------------------",
             "Use exactly the same fixed operator on the data residual and every",
             "Doppler-shifted planetary template, in the same continuum units:",
             "  from smooth_residuals import smoother_matrix",
             "  config = {'kind': 'polynomial', 'degree': 5}",
             "  H = smoother_matrix(wavelength_um, included_mask, config)",
             "  residual_filtered = residual - H @ residual",
             "  planet_filtered = planet - H @ planet",
             "Apply this separately per order/exposure. Never correlate filtered data",
             "against an unfiltered template when estimating line amplitude. Matching",
             "the filter accounts for the response but does not recreate discarded",
             "information. Residual covariance changes as well. These RMS values do",
             "not establish a detection-significance improvement.", "",
             "The injection test covers only the new fixed smoothing stage. Existing",
             "telluric/continuum fitting was not repeated after injection. Artificial",
             "forests are injected at observed resolution with approximate transmission",
             "weighting, not before IP convolution. End-to-end recovery with the actual",
             "planet model remains necessary for a measured planet amplitude.", "",
             "Verification and files", "----------------------",
             "Independent QR/direct-GP-solve arithmetic reproduces all 15 candidates",
             "and both modes, including normalization and masking. Held-out score",
             "perturbations leave calibration-only corrections unchanged. Explicit",
             "injected-minus-uninjected differences match (I-H)s to 1.1e-15.",
             "All edits and artifacts remain within Git-ignored IGRINS/.", "",
             "Reproduce in order:", "  python IGRINS/smooth_residuals.py",
             "  python IGRINS/check_smooth_signal.py", "  python IGRINS/audit_smooth_residuals.py",
             "  python IGRINS/summarize_smoothing.py", "",
             "Outputs under IGRINS/results/:", "  smooth_planet_tradeoff.json / .png",
             "  smooth_comparison.json / .csv / .png", "  smooth_results.npz (all corrections and original residuals)",
             "  smooth_signal_retention.json / .csv / .png", "  smooth_independent_audit.json"]
    (OUT / "SMOOTH_PLANET_REPORT.txt").write_text("\n".join(text)+"\n")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x, [row["rms_percent"] for row in rows], color="steelblue")
    axes[0].axhline(baseline, color="0.4", ls="--", lw=1)
    axes[0].set(ylabel="Continuum RMS after filtering (%)", title="All-included smoothing; scoring pixels participate")
    low = np.array([100*row["forest_mean_removed_min"] for row in rows])
    high = np.array([100*row["forest_mean_removed_max"] for row in rows])
    axes[1].errorbar(x, (low+high)/2, yerr=(high-low)/2, fmt="o", capsize=4)
    axes[1].axhline(95, color="0.4", ls="--", lw=1)
    axes[1].set(ylabel="Synthetic forest mean-removed gain (%)", ylim=(40, 102),
                title="FWHM 10 km/s; hypothetical Kp 0/100/200 km/s")
    for ax in axes:
        ax.set_xticks(x, LABELS, rotation=55, ha="right", fontsize=9)
    fig.savefig(OUT / "smooth_planet_tradeoff.png", dpi=160)
    plt.close(fig)
    print("Strict raw gate selected:", conservative)
    print("Wrote SMOOTH_PLANET_REPORT.txt and smooth_planet_tradeoff.{json,png}")


if __name__ == "__main__":
    main()
