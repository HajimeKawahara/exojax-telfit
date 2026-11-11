"""Present the completed stellar control without changing its fitting source."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
LABELS = {
    "phoenix": "PHOENIX adopted",
    "initial": "TP initial labels",
    "stellar_only": "TP stellar calibration",
    "joint": "TP calibrated star + strength refit",
}
COLORS = dict(phoenix="#2166ac", initial="#999999", stellar_only="#e08214", joint="#b2182b")


def main():
    report_path = OUT / "transformerpayne_stellar_control.json"
    report = json.loads(report_path.read_text())
    for filename, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == expected, filename
    with np.load(OUT / "transformerpayne_stellar_control.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    frame = int(np.flatnonzero(data["exposures"] == 30)[0])
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    for pair, b in enumerate([1, 2]):
        x = 1e4 / data["wavelength_um"][b]
        ax, rx = axes[2*pair:2*pair+2]
        excluded = ~data["included_mask"][b]
        edges = np.r_[x[0]-(x[1]-x[0])/2, (x[1:]+x[:-1])/2, x[-1]+(x[-1]-x[-2])/2]
        starts = np.flatnonzero(np.diff(np.r_[False, excluded, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, excluded, False].astype(int)) == -1)
        for axis in [ax, rx]:
            for lo, hi in zip(starts, ends):
                axis.axvspan(edges[lo], edges[hi], color="#eeeeee", zorder=0)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.set_xlim(x.min(), x.max())
            axis.grid(axis="y", alpha=.2)
        ax.plot(x, data["y_over_c"][b, frame], color="black", lw=.6, label="Observed")
        for name in ["phoenix", "joint"]:
            fitted = data["y_over_c"]-data[f"residual_{name}"]
            ax.plot(x, fitted[b, frame], color=COLORS[name], lw=.9, label=LABELS[name])
            star_label = "PHOENIX star alone" if name == "phoenix" else "TP calibrated star alone"
            ax.plot(x, data[f"stellar_only_{name}"][b, frame], color=COLORS[name], lw=1, ls="--", label=star_label)
        for name, label in LABELS.items():
            residual = np.where(data["included_mask"][b], data[f"residual_{name}"][b, frame], np.nan)
            rx.plot(x, 100*residual, color=COLORS[name], lw=.7, label=label)
        ax.set(ylabel="Flux / original continuum", title=f"Order {data['orders'][b]}: evaluation frame 30", ylim=(-.05, 1.15))
        rx.set(ylabel="Residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)")
        rx.axhline(0, color="0.4", lw=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles+[Patch(facecolor="#eeeeee")], labels+["Original exclusions"], fontsize=8, ncol=2, loc="lower left")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("TransformerPayne stellar calibration, then a separate telluric-strength refit\nStar: fixed training transmission >95%; comparison: original Q5 projection and evaluation pixels", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "transformerpayne_stellar_control.png", dpi=170)
    plt.close(fig)

    lines = [
        "TransformerPayne stellar-calibration control", "",
        "The model improves after optimizing the star, but does not beat the adopted PHOENIX model.",
        "No model adoption or planet injection/recovery test is performed.", "",
        "Protocol:",
        "1. Select pixels with adopted telluric-only transmission >0.95 in all five training frames.",
        "   Fit six shared stellar parameters with adopted telluric strengths fixed; profile Q5 on these pixels.",
        "2. Freeze the calibrated star and refit 119 telluric strengths on the original included training pixels.",
        "3. Compare all cases on the same four evaluation frames using the ORIGINAL Q5 projection and score mask.",
        "   Evaluation flux does not select pixels or fit the stellar/strength parameters.",
        "   Order13 remains PHOENIX; TransformerPayne covers only H orders34/40 here.", "",
        "Calibration pixels / strong-stellar pixels (PHOENIX depth >2%):",
    ]
    for row in report["mask_counts"]:
        lines.append(f"Order {row['order']}: {row['calibration_pixels']} / {row['strong_stellar_calibration_pixels']}; "
                     f"{row['newly_used_pixels']} pixels were outside the original training mask.")
    lines += ["", "RMS (% of original continuum):", "Case | H score | Order34 | Order40 | Hybrid3 | H full-window | H strong-stellar"]
    for name, label in LABELS.items():
        entry = report["metrics"][name]["evaluation"]
        score = entry["score"]
        values = [score["h_orders"]["rms_percent"], score["per_order"]["34"]["rms_percent"],
                  score["per_order"]["40"]["rms_percent"], score["hybrid_all_orders"]["rms_percent"],
                  entry["full_window"]["h_orders"]["rms_percent"], entry["strong_stellar_diagnostic"]["h_orders"]["rms_percent"]]
        lines.append(label + " | " + " | ".join(f"{value:.6f}" for value in values))
    lines += ["", "Effective fitted stellar parameters:"]
    for name, value in report["fitted_stellar"].items():
        lines.append(f"{name}: {value:.8g}")
    lines += ["", "Convergence:"]
    for name, fit in report["fits"].items():
        free_bounds = [i for i in fit["active_bounds"] if (i < 6 if name == "stellar" else i >= 6)]
        fixed_bounds = [i for i in fit["active_bounds"] if i not in free_bounds]
        lines.append(f"{name}: {fit['message']} nfev={fit['nfev']}; optimality={fit['optimality']:.6g}; "
                     f"objective={fit['objective']:.8f}; active free-parameter bounds={free_bounds}; "
                     f"fixed parameters already at bounds={fixed_bounds}.")
    lines += [
        "Stellar logg and extra Gaussian width reach their allowed upper/lower bounds (5 and 0.3 km/s).",
        "The stellar stage stops by small parameter changes, not by a small gradient; its parameters are not a precise stellar retrieval.",
        "The two stages use different training masks, so their objective values must not be compared directly.", "",
        "Plain explanation:",
        "The original telluric-focused mask removes many stellar lines. To give the stellar model useful information,",
        "this control calibrates it where the adopted telluric model transmits at least 95% of the light.",
        "It then holds that star fixed while adjusting telluric line strengths. The residual falls substantially",
        "from the initial TransformerPayne prediction, especially in order40, but the PHOENIX model still fits better.",
        "The remaining gap can reflect stellar-model mismatch and fixed broadening/nuisance assumptions; this experiment",
        "does not isolate a single cause or establish physically reliable stellar parameters.", "",
        "Figure: transformerpayne_stellar_control.png. Upper panels include stellar-only curves.",
        "Lower panels show original included pixels; gray regions are original exclusions.",
        "All plotted residuals use the original Q5 projection. Calibration pixels are used only for stage1 training.", "",
        "Additional interpretation limits:",
    ] + report["caveats"]
    sensitivity_path = OUT / "transformerpayne_stellar_control_sensitivity.json"
    if sensitivity_path.exists():
        sensitivity = json.loads(sensitivity_path.read_text())
        improved = [row for row in sensitivity["trials"] if row["objective_change"] < 0]
        lines += ["", "Training-only finite-step sensitivity check:",
                  "Each parameter was perturbed separately at 10 K / 0.01 dex / 0.01 dex / 0.02 km/s / 0.02 km/s / 0.02 km/s; only feasible directions were tested.",
                  "All tested physical steps increased the training objective; no larger fit was triggered." if not improved else
                  "At least one tested step decreased the training objective; inspect the sensitivity JSON before interpreting convergence."]
        for row in sensitivity["trials"]:
            lines.append(f"{row['parameter']} {row['delta']:+g}: objective change {row['objective_change']:+.8f} ({100*row['relative_change']:+.8f}%).")
        lines.append(sensitivity["explanation"])
    (OUT / "TRANSFORMERPAYNE_STELLAR_CONTROL_REPORT.txt").write_text("\n".join(lines)+"\n")
    report["presentation"] = dict(source=str(Path(__file__).relative_to(ROOT)),
                                  sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                  note="Plot labels and English prose regenerated from saved fit arrays; fitting source and fitted parameters unchanged.")
    report_path.write_text(json.dumps(report, indent=2)+"\n")
    print("Control figure and report regenerated; original fitting source hashes verified.")


if __name__ == "__main__":
    main()
