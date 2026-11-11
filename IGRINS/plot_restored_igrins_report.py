"""Restore the original report layout using the current manifest-selected fit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from plot_premask_report import BLUE, COLORS, GRAY, GREEN, RED, clean_axis, save, shade_excluded

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rms(values):
    return float(100*np.sqrt(np.mean(np.asarray(values)**2)))


def main():
    plt.rcParams.update({"font.size": 11, "axes.titlesize": 11,
                         "axes.labelsize": 11, "legend.fontsize": 10,
                         "font.family": "DejaVu Sans"})
    manifest_path = OUT/"adopted_telluric_model.json"
    manifest = json.loads(manifest_path.read_text())
    product = ROOT/manifest["product"]
    assert digest(product) == manifest["product_sha256"]
    with np.load(product) as saved:
        a = {key: saved[key] for key in saved.files}
    regions_path = OUT/"poly5_spectral_budget.json"
    regions = json.loads(regions_path.read_text())["region_sets"]["post_filter_training_selection"]
    orders, included, score = a["orders"], a["included_mask"], a["score_mask"]
    evaluation_frames = manifest["scope"]["evaluation_exposures"]
    evaluation = [int(np.flatnonzero(a["exposures"] == e)[0]) for e in evaluation_frames]
    frame = int(np.flatnonzero(a["exposures"] == 30)[0])
    assert evaluation_frames == [10, 30, 50, 70]
    assert len(orders) == 3 and a["stellar_only"].shape == a["component"].shape
    stellar_only = a["stellar_only"][:, frame]
    nu = 1e4/a["wavelength_um"]
    residual = a["residual_filtered"][:, evaluation]
    smooth = a["residual_unfiltered"]-a["residual_filtered"]
    fitted = a["component"]+smooth
    np.testing.assert_allclose(a["y_over_c"]-fitted, a["residual_filtered"], atol=3e-16, rtol=0)
    projected = np.empty_like(a["component"])
    for b, (wave, mask) in enumerate(zip(a["wavelength_um"], included)):
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        raw = a["y_over_c"][b]-a["component"][b]
        projected[b] = raw-(raw[:, mask]@np.linalg.pinv(basis[mask]).T)@basis.T
    q5_error = float(np.max(np.abs(projected-a["residual_filtered"])))
    assert q5_error < 1e-11
    direct_rms = rms(np.concatenate([row[:, mask].ravel() for row, mask in zip(residual, score)]))
    np.testing.assert_allclose(direct_rms, manifest["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12)
    per_order = {str(order): rms(residual[b][:, score[b]]) for b, order in enumerate(orders)}
    for order in orders:
        np.testing.assert_allclose(per_order[str(order)], manifest["metrics"]["evaluation_score"]["per_order"][str(order)], atol=1e-12)

    fig = plt.figure(figsize=(12, 11.5))
    gs = fig.add_gridspec(6, 1, height_ratios=[2, 1, 2, 1, 2, 1],
                          left=.09, right=.98, bottom=.085, top=.86, hspace=.38)
    fig.suptitle("WASP-77Ab / IGRINS: current adopted spectral fit", y=.98,
                 fontsize=15, weight="bold")
    fig.text(.5, .945, "Evaluation frame 30  |  shared stellar amplitudes + regularized telluric strengths + degree-five correction",
             ha="center", fontsize=10)
    legend = [Line2D([], [], color="#303030", lw=1, label="Observed spectrum"),
              Line2D([], [], color=RED, lw=1.4, label="Current model + smooth correction"),
              Line2D([], [], color=GREEN, lw=1.4, ls="--", label="Current stellar model alone (after IP)"),
              Line2D([], [], color=BLUE, marker=".", ls="", label="Score pixels"),
              Patch(facecolor="#ececec", label="Original exclusions")]
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .928), ncol=3, frameon=False)
    for b, order in enumerate(orders):
        x, inc, sc = nu[b], included[b], score[b]
        ax = fig.add_subplot(gs[2*b])
        rx = fig.add_subplot(gs[2*b+1], sharex=ax)
        for axis in [ax, rx]:
            shade_excluded(axis, x, inc)
            clean_axis(axis)
            axis.set_xlim(x.min(), x.max())
        ax.plot(x, a["y_over_c"][b, frame], color="#303030", lw=.7)
        ax.plot(x, fitted[b, frame], color=RED, lw=1.1, alpha=.9)
        ax.plot(x, stellar_only[b], color=GREEN, lw=1.2, ls="--", zorder=4)
        ax.set_ylabel("Flux / continuum")
        ax.set_ylim(-.05, 1.14)
        ax.set_title(f"Order {order}  |  four-frame score RMS = {per_order[str(order)]:.3f}%", loc="left")
        ax.tick_params(labelbottom=False)
        values = 100*a["residual_filtered"][b, frame]
        rx.axhline(0, color="#444444", lw=.7)
        rx.plot(x, np.where(inc, values, np.nan), color=GRAY, lw=.6)
        rx.scatter(x[sc], values[sc], color=BLUE, s=6, zorder=3)
        rx.set_ylim(-6, 6)
        rx.set_yticks([-4, 0, 4])
        rx.set_ylabel("Residual (%)")
        if b == len(orders)-1:
            rx.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.text(.5, .014, "Green: current PHOENIX + blase stellar component with the same velocity and IP, without telluric absorption.\n"
             "Residual = observed minus total model, in percent of the original continuum. Original masks and IP are retained.",
             ha="center", fontsize=10)
    save(fig, "restored_current_spectral_fit")

    fig = plt.figure(figsize=(12, 13))
    gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1.35, 1.35],
                          left=.08, right=.98, bottom=.11, top=.90,
                          hspace=.70, wspace=.20)
    fig.suptitle("Current residual structure across evaluation frames", y=.985,
                 fontsize=15, weight="bold")
    fig.text(.5, .957, "Current adopted model, after degree-five filtering; the time mean is shown for diagnosis only",
             ha="center", fontsize=11)
    legend = [Line2D([], [], color=color, lw=1.2, label=f"Frame {e}") for color, e in zip(COLORS, evaluation_frames)]
    legend.append(Line2D([], [], color="black", lw=1.8, label="Four-frame mean"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .944), ncol=5, frameon=False)
    max_residual = float(max(np.max(np.abs(row[:, inc])) for row, inc in zip(residual*100, included)))
    ylim = max(6, np.ceil(max_residual))
    for b, order in enumerate(orders):
        ax = fig.add_subplot(gs[b, :])
        x, inc = nu[b], included[b]
        shade_excluded(ax, x, inc)
        for j, color in enumerate(COLORS):
            ax.plot(x, np.where(inc, 100*residual[b, j], np.nan), color=color, lw=.65, alpha=.75)
        ax.plot(x, np.where(inc, 100*residual[b].mean(axis=0), np.nan), color="black", lw=1.1)
        ax.axhline(0, color="#555555", lw=.6)
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel("Residual (%)")
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_title(f"Order {order}  |  score RMS {per_order[str(order)]:.3f}%", loc="left")
        clean_axis(ax)
    hotspot_records = []
    for original in regions:
        b = int(np.flatnonzero(orders == original["order"])[0])
        lo, hi = original["array_interval"]
        interval = np.zeros(len(nu[b]), dtype=bool)
        interval[lo:hi] = True
        selected = interval & score[b]
        if not selected.any():
            continue
        k = len(hotspot_records)
        if k >= 4:
            raise ValueError("The historical four-window selection has changed")
        center = 1e4/original["peak_wavelength_um"]
        local_rms = rms(residual[b][:, selected])
        ax = fig.add_subplot(gs[3+k//2, k%2])
        x, inc = nu[b], included[b]
        near = abs(x-center) < .40
        shade_excluded(ax, x, inc)
        ax.axvspan(min(x[lo:hi]), max(x[lo:hi]), color="#ffdc73", alpha=.55, zorder=1)
        for j, color in enumerate(COLORS):
            ax.plot(x, np.where(inc, 100*residual[b, j], np.nan), color=color,
                    marker=".", markersize=3, lw=.8, alpha=.8)
        ax.plot(x, np.where(inc, 100*residual[b].mean(axis=0), np.nan), color="black", lw=1.8)
        ax.axhline(0, color="#555555", lw=.7)
        ax.set_xlim(max(x.min(), center-.4), min(x.max(), center+.4))
        limit = max(4, np.ceil(100*np.max(np.abs(residual[b][:, near & inc]))))
        ax.set_ylim(-limit, limit)
        ax.set_ylabel("Residual (%)")
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
        ax.set_title(f"Order {orders[b]}, {original['peak_wavelength_um']:.6f} $\\mu$m\nEarlier selected interval: RMS {local_rms:.2f}%", loc="left")
        clean_axis(ax)
        hotspot_records.append(dict(order=int(orders[b]), peak_wavelength_um=original["peak_wavelength_um"],
                                    peak_wavenumber_cm1=center, array_interval=[lo, hi],
                                    score_pixels=int(selected.sum()), evaluation_samples=int(selected.sum())*len(evaluation),
                                    local_score_rms_percent=local_rms, axis_limit_percent=float(limit),
                                    clipped_included_points=int(np.sum(np.abs(residual[b][:, near & inc])*100 > limit))))
    assert len(hotspot_records) == 4
    fig.text(.5, .025, "Yellow: the same intervals selected in the earlier training analysis; no new selection or masking. Gray: original exclusions.\n"
             "Every RMS is recomputed from the current adopted predictions. Repeatability alone does not identify a line-list error.",
             ha="center", fontsize=10)
    save(fig, "restored_current_residuals")

    historical = [OUT/f"premask_{name}.png" for name in ["spectral_fit", "residuals", "experiments"]]
    sources = [manifest_path, product, regions_path, ROOT/"plot_premask_report.py"]
    verification = dict(created_utc=datetime.now(timezone.utc).isoformat(), case=manifest["case"],
        current_product=manifest["product"], additional_wavenumber_mask_applied=False,
        evaluation_frames=evaluation_frames, spectral_figure_frame=30,
        stellar_overlay="Current adopted stellar_only array; same velocities, broadening and IP, without telluric absorption or Q5 correction",
        stellar_only_range=[float(stellar_only.min()), float(stellar_only.max())],
        upper_panel_axis_limits=[-.05, 1.14],
        upper_panel_clipped_points={name: int(np.sum((value < -.05) | (value > 1.14)))
                                    for name, value in [("observed", a["y_over_c"][:, frame]), ("model_plus_q5", fitted[:, frame]), ("stellar_only", stellar_only)]},
        included_wavelength_pixels=int(included.sum()), score_wavelength_pixels=int(score.sum()),
        evaluation_score_samples=int(score.sum())*len(evaluation), evaluation_score_rms_percent=direct_rms,
        per_order_score_rms_percent=per_order, independent_q5_max_error=q5_error,
        model_plus_smooth_residual_max_error=float(np.max(np.abs(a["y_over_c"]-fitted-a["residual_filtered"]))),
        spectral_figure_residual_axis_limit_percent=6,
        spectral_figure_clipped_included_residual_points=int(sum(np.sum(np.abs(a["residual_filtered"][b, frame, inc])*100 > 6) for b, inc in enumerate(included))),
        residual_figure_axis_limit_percent=float(ylim), max_included_evaluation_residual_percent=max_residual,
        residual_figure_clipped_included_points=int(sum(np.sum(np.abs(row[:, inc])*100 > ylim) for row, inc in zip(residual, included))),
        hotspots=hotspot_records,
        source_sha256={str(path.relative_to(ROOT)): digest(path) for path in sources},
        historical_figure_sha256={str(path.relative_to(ROOT)): digest(path) for path in historical},
        script_sha256=digest(Path(__file__)),
        figure_sha256={f"{stem}.png": digest(OUT/f"{stem}.png") for stem in ["restored_current_spectral_fit", "restored_current_residuals"]})
    (OUT/"restored_current_figures_validation.json").write_text(json.dumps(verification, indent=2)+"\n")
    print(json.dumps(dict(rms_percent=direct_rms, figures=verification["figure_sha256"], hotspots=hotspot_records)))


if __name__ == "__main__":
    main()
