"""Plot the adopted IGRINS fit before any additional wavenumber masking."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCES = [
    "adopted_telluric_fit.npz",
    "adopted_telluric_model.json",
    "residual_experiment_summary.json",
    "poly5_spectral_budget.json",
    "quadratic_strength_fit.json",
    "premask_stellar_component.npz",
    "premask_stellar_component.json",
]
BLUE = "#2166ac"
RED = "#b2182b"
GREEN = "#238b45"
GRAY = "#777777"
COLORS = ["#4c78a8", "#f58518", "#54a24b", "#b279a2"]


def read_json(name):
    return json.loads((OUT / name).read_text())


def shade_excluded(ax, x, included):
    """Shade pre-existing exclusions, using pixel edges without filling gaps."""
    edges = np.r_[x[0] - (x[1] - x[0]) / 2, (x[:-1] + x[1:]) / 2,
                  x[-1] + (x[-1] - x[-2]) / 2]
    starts = np.flatnonzero(np.diff(np.r_[False, ~included, False].astype(int)) == 1)
    ends = np.flatnonzero(np.diff(np.r_[False, ~included, False].astype(int)) == -1)
    for start, end in zip(starts, ends):
        ax.axvspan(edges[start], edges[end], color="#ececec", zorder=0)


def clean_axis(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=170, facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11, "legend.fontsize": 10,
                         "font.family": "DejaVu Sans"})
    a = np.load(OUT / SOURCES[0])
    manifest = read_json(SOURCES[1])
    summary = read_json(SOURCES[2])
    regions = read_json(SOURCES[3])["region_sets"]["post_filter_training_selection"]
    quadratic = read_json(SOURCES[4])
    assert manifest["case"] == "fixed_contrast10"
    assert hashlib.sha256((OUT / SOURCES[0]).read_bytes()).hexdigest() == manifest["product_sha256"]
    orders = a["orders"]
    evaluation = [int(np.flatnonzero(a["exposures"] == e)[0]) for e in [10, 30, 50, 70]]
    frame = int(np.flatnonzero(a["exposures"] == 30)[0])
    with np.load(OUT / "premask_stellar_component.npz") as stellar:
        np.testing.assert_array_equal(stellar["orders"], orders)
        np.testing.assert_array_equal(stellar["wavelength_um"], a["wavelength_um"])
        assert int(stellar["exposure"]) == int(a["exposures"][frame])
        stellar_only = stellar["stellar_only"].copy()
    assert stellar_only.shape == a["wavelength_um"].shape
    assert np.all(np.isfinite(stellar_only))
    included, score = a["included_mask"], a["score_mask"]
    nu = 1e4 / a["wavelength_um"]
    residual = a["residual_filtered"][:, evaluation, :]
    smooth = a["residual_unfiltered"] - a["residual_filtered"]
    fitted = a["component"] + smooth
    np.testing.assert_allclose(a["y_over_c"] - fitted, a["residual_filtered"], atol=3e-16)
    direct_rms = 100 * np.sqrt(np.mean(np.concatenate([r[:, s].ravel() for r, s in zip(residual, score)]) ** 2))
    np.testing.assert_allclose(direct_rms, manifest["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12)
    stats = summary["metrics"]["adopted"]

    # A full-window view makes both the absorption fit and selection visible.
    fig = plt.figure(figsize=(12, 11.5))
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 1, 2, 1, 2, 1],
                          left=0.09, right=0.98, bottom=0.085, top=0.86, hspace=0.38)
    fig.suptitle("WASP-77Ab / IGRINS: adopted spectral fit before additional masking",
                 y=0.98, fontsize=15, weight="bold")
    fig.text(0.5, 0.945, "Evaluation frame 30  |  fixed two-Gaussian IP + regularized line strengths + degree-five smooth correction",
             ha="center", fontsize=10)
    legend = [Line2D([], [], color="#303030", lw=1, label="Observed spectrum"),
              Line2D([], [], color=RED, lw=1.4, label="Adopted model + smooth correction"),
              Line2D([], [], color=GREEN, lw=1.4, ls="--", label="Stellar model alone (after IP)"),
              Line2D([], [], color=BLUE, marker=".", ls="", label="Score pixels"),
              Patch(facecolor="#ececec", label="Pre-existing exclusions")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.928), ncol=3, frameon=False)
    for i, order in enumerate(orders):
        x, inc, sc = nu[i], included[i], score[i]
        ax = fig.add_subplot(gs[2 * i])
        rx = fig.add_subplot(gs[2 * i + 1], sharex=ax)
        for axis in [ax, rx]:
            shade_excluded(axis, x, inc)
            clean_axis(axis)
            axis.set_xlim(x.min(), x.max())
        ax.plot(x, a["y_over_c"][i, frame], color="#303030", lw=0.7)
        ax.plot(x, fitted[i, frame], color=RED, lw=1.1, alpha=0.9)
        ax.plot(x, stellar_only[i], color=GREEN, lw=1.2, ls="--", zorder=4)
        ax.set_ylabel("Flux / continuum")
        ax.set_ylim(-0.05, 1.14)
        ax.set_title(f"Order {order}  |  four-frame score RMS = {stats['per_order'][str(order)]['rms_percent']:.3f}%", loc="left")
        ax.tick_params(labelbottom=False)
        r = 100 * a["residual_filtered"][i, frame]
        rx.axhline(0, color="#444444", lw=0.7)
        rx.plot(x, np.where(inc, r, np.nan), color=GRAY, lw=0.6)
        rx.scatter(x[sc], r[sc], color=BLUE, s=6, zorder=3)
        rx.set_ylim(-6, 6)
        rx.set_yticks([-4, 0, 4])
        rx.set_ylabel("Residual (%)")
        if i == len(orders) - 1:
            rx.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.text(0.5, 0.014, "Green: stellar spectrum with the same velocity and instrumental response, without telluric absorption.\n"
             "Residual = observed minus total model, in percent of the original continuum. No new pixels are removed.",
             ha="center", fontsize=10)
    save(fig, "premask_spectral_fit")

    # Repeatability in time is shown without subtracting the time mean.
    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.35, 1.35],
                          left=0.08, right=0.98, bottom=0.11, top=0.90,
                          hspace=0.70, wspace=0.20)
    fig.suptitle("Residual structure repeats across evaluation frames", y=0.985,
                 fontsize=15, weight="bold")
    fig.text(0.5, 0.957, "Adopted model, after degree-five filtering; the time mean is shown for diagnosis only",
             ha="center", fontsize=11)
    legend = [Line2D([], [], color=c, lw=1.2, label=f"Frame {e}") for c, e in zip(COLORS, [10, 30, 50, 70])]
    legend.append(Line2D([], [], color="black", lw=1.8, label="Four-frame mean"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.944), ncol=5, frameon=False)
    max_residual = float(np.max([np.max(np.abs(r[:, inc])) for r, inc in zip(residual * 100, included)]))
    ylim = max(6, np.ceil(max_residual))
    for i, order in enumerate(orders):
        ax = fig.add_subplot(gs[i, :])
        x, inc = nu[i], included[i]
        shade_excluded(ax, x, inc)
        for j, color in enumerate(COLORS):
            ax.plot(x, np.where(inc, 100 * residual[i, j], np.nan), color=color, lw=0.65, alpha=0.75)
        ax.plot(x, np.where(inc, 100 * residual[i].mean(axis=0), np.nan), color="black", lw=1.1)
        ax.axhline(0, color="#555555", lw=0.6)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel("Residual (%)")
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_title(f"Order {order}  |  score RMS {stats['per_order'][str(order)]['rms_percent']:.3f}%", loc="left")
        clean_axis(ax)
    hotspot_records = []
    for k, region in enumerate(stats["regions"]):
        i = int(np.flatnonzero(orders == region["order"])[0])
        center = 1e4 / region["peak_wavelength_um"]
        original = min((v for v in regions if v["order"] == region["order"]),
                       key=lambda v: abs(v["peak_wavelength_um"] - region["peak_wavelength_um"]))
        lo, hi = original["array_interval"]
        interval = np.zeros(900, dtype=bool)
        interval[lo:hi] = True
        selected = interval & score[i]
        local_rms = float(100 * np.sqrt(np.mean(residual[i][:, selected] ** 2)))
        np.testing.assert_allclose(local_rms, region["metrics"]["rms_percent"], atol=1e-12)
        ax = fig.add_subplot(gs[3 + k // 2, k % 2])
        x, inc = nu[i], included[i]
        near = abs(x - center) < 0.40
        shade_excluded(ax, x, inc)
        ax.axvspan(min(x[lo:hi]), max(x[lo:hi]), color="#ffdc73", alpha=0.55, zorder=1)
        for j, color in enumerate(COLORS):
            ax.plot(x, np.where(inc, 100 * residual[i, j], np.nan), color=color, marker=".", markersize=3, lw=0.8, alpha=0.8)
        ax.plot(x, np.where(inc, 100 * residual[i].mean(axis=0), np.nan), color="black", lw=1.8)
        ax.axhline(0, color="#555555", lw=0.7)
        ax.set_xlim(max(x.min(), center - 0.4), min(x.max(), center + 0.4))
        lim = max(4, np.ceil(100 * np.max(np.abs(residual[i][:, near & inc]))))
        ax.set_ylim(-lim, lim)
        ax.set_ylabel("Residual (%)")
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_title(f"Order {region['order']}, {region['peak_wavelength_um']:.6f} $\\mu$m\nEarlier selected interval: RMS {local_rms:.2f}%", loc="left")
        clean_axis(ax)
        hotspot_records.append({"order": int(region["order"]), "peak_wavelength_um": region["peak_wavelength_um"],
                                "peak_wavenumber_cm1": center, "array_interval": [lo, hi],
                                "score_pixels": int(selected.sum()), "local_score_rms_percent": local_rms})
    fig.text(0.5, 0.025, "Yellow: intervals selected in earlier training analysis; all remain included. Gray: pre-existing exclusions.\n"
             "A repeatable residual can have several causes; repeatability alone does not identify an error in the line list.",
             ha="center", fontsize=10)
    save(fig, "premask_residuals")

    cases = ["adopted", "width_prior3", "center_prior3", "joint_prior3", "profile_joint_strengths",
             "background_wavelength_joint_strengths", "quadratic_gaussian_contrast10"]
    labels = ["Adopted: original IP + line strengths", "Air-broadening widths: 35 H$_2$O lines",
              "Line centers: 35 H$_2$O lines", "Widths + centers: 35 H$_2$O lines",
              "Vertical atmosphere + strengths (local)", "Background / wavelength + strengths (local)",
              "Variable Gaussian IP + strengths"]
    rms = [summary["metrics"][k]["rms_percent"] for k in cases[:-1]]
    rms.append(quadratic["cases"][cases[-1]]["metrics"]["evaluation_score"]["rms_percent"])
    reduction = 100 * (1 - (np.array(rms) / rms[0]) ** 2)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.8), gridspec_kw={"width_ratios": [1.3, 1]}, sharey=True)
    fig.subplots_adjust(left=0.34, right=0.97, top=0.80, bottom=0.19, wspace=0.32)
    fig.suptitle("What changes the remaining residual?", y=0.965, fontsize=15, weight="bold")
    fig.text(0.5, 0.91, "Same 523 score wavelengths × 4 evaluation frames; degree-five filtering in every case",
             ha="center", fontsize=10.5)
    y = np.arange(len(cases))
    colors = [BLUE] + ["#999999"] * (len(cases) - 1)
    axes[0].barh(y, rms, color=colors, height=0.65)
    axes[1].barh(y, reduction, color=colors, height=0.65)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 0.9)
    axes[1].set_xlim(-1.5, 8.5)
    axes[0].set_xlabel("Evaluation RMS (% of continuum)")
    axes[1].set_xlabel("Squared residual reduction (%)")
    for i, (v, red) in enumerate(zip(rms, reduction)):
        axes[0].text(v + 0.012, i, f"{v:.4f}", va="center", fontsize=10)
        axes[1].text(max(0, red) + 0.16, i, f"{red:+.2f}" if i else "0.00", va="center", fontsize=10)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#dddddd", lw=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
    axes[1].axvline(0, color="#444444", lw=0.7)
    fig.text(0.5, 0.075, "Blue is adopted; gray cases are experiments. Width/center freedom gives the largest tested gain,\n"
             "but also changes injected planet-line patterns. A lower residual RMS does not by itself justify adoption.",
             ha="center", fontsize=10.5)
    save(fig, "premask_experiments")

    verification = {
        "case": manifest["case"], "additional_wavenumber_mask_applied": False,
        "units": "Percent of the original fitted continuum",
        "evaluation_frames": [10, 30, 50, 70], "spectral_figure_frame": 30,
        "stellar_overlay": "PHOENIX alone, with adopted rotation, stellar broadening, frame velocity, IP and wavelength sampling; no telluric absorption or smooth residual correction",
        "stellar_only_range": [float(stellar_only.min()), float(stellar_only.max())],
        "stellar_overlay_clipped_points": int(np.sum((stellar_only < -0.05) | (stellar_only > 1.14))),
        "included_wavelength_pixels": int(included.sum()), "score_wavelength_pixels": int(score.sum()),
        "evaluation_score_samples": int(score.sum()) * 4,
        "evaluation_score_rms_percent": float(direct_rms),
        "model_plus_smooth_residual_max_error": float(np.max(np.abs(a["y_over_c"] - fitted - a["residual_filtered"]))),
        "spectral_figure_residual_axis_limit_percent": 6,
        "spectral_figure_clipped_included_residual_points": int(sum(np.sum(np.abs(a["residual_filtered"][i, frame, inc]) * 100 > 6) for i, inc in enumerate(included))),
        "residual_figure_axis_limit_percent": float(ylim),
        "max_included_evaluation_residual_percent": max_residual,
        "hotspots": hotspot_records,
        "experiment_rms_percent": dict(zip(cases, [float(v) for v in rms])),
        "source_sha256": {name: hashlib.sha256((OUT / name).read_bytes()).hexdigest() for name in SOURCES},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "figure_sha256": {f"{stem}.png": hashlib.sha256((OUT / f"{stem}.png").read_bytes()).hexdigest()
                          for stem in ["premask_spectral_fit", "premask_residuals", "premask_experiments"]},
    }
    (OUT / "premask_figures_validation.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps({"rms_percent": direct_rms, "figures": list(verification["figure_sha256"])}))


if __name__ == "__main__":
    main()
