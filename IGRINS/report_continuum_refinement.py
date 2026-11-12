"""Compare frozen-model continuum revisions in one unchanged flux scale."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
import numpy as np

ROOT = Path(__file__).resolve().parent
EVALUATION = [10, 30, 50, 70]
SPECTRAL_KEYS = ["observed_fixed_scale", "continuum_fixed_scale", "full_model",
                 "stellar_only", "telluric_only", "residual_continuum_q5",
                 "residual_fixed_scale_q5_independent", "observed_continuum",
                 "smooth_continuum_q5"]


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 ** 2), b""):
            checksum.update(block)
    return checksum.hexdigest()


def load_product(path):
    with np.load(path, allow_pickle=False) as saved:
        keys = ["orders", "exposures", "wavelength_um", "wavenumber_cm1", *SPECTRAL_KEYS]
        arrays = {key: saved[key] for key in keys}
        for key in ["continuum_active_lower_bounds", "continuum_coefficient_floor"]:
            if key in saved:
                arrays[key] = saved[key]
    np.testing.assert_array_equal(arrays["orders"], np.arange(44))
    np.testing.assert_array_equal(arrays["exposures"], np.arange(79))
    for key in ["wavelength_um", "wavenumber_cm1"]:
        assert arrays[key].shape == (44, 1848), key
    for key in SPECTRAL_KEYS:
        assert arrays[key].shape == (44, 79, 1848), key
        if key not in {"residual_continuum_q5", "observed_continuum"}:
            assert np.isfinite(arrays[key]).all(), key
    assert (np.diff(arrays["wavenumber_cm1"], axis=1) > 0).all()
    return arrays


def audit_q5(arrays):
    """Reconstruct the independent fixed-scale projection for every frame."""
    largest = 0.
    for order in range(44):
        log_wave = np.log(arrays["wavelength_um"][order])
        x = 2 * (log_wave - log_wave.min()) / np.ptp(log_wave) - 1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, _ = np.linalg.qr(basis, mode="reduced")
        residual = (arrays["observed_fixed_scale"][order]
                    - arrays["full_model"][order] * arrays["continuum_fixed_scale"][order])
        filtered = residual - (residual @ q) @ q.T
        error = float(np.max(np.abs(filtered - arrays["residual_fixed_scale_q5_independent"][order])))
        largest = max(largest, error)
    if largest > 1e-10:
        raise ValueError(f"Fixed-scale Q5 reconstruction failed: {largest:.3g}")
    return largest


def metrics(values):
    finite = np.isfinite(values)
    return dict(rms_percent=float(100 * np.sqrt(np.mean(values[finite] ** 2)))
                if finite.any() else None, finite_samples=int(finite.sum()),
                undefined_samples=int(values.size - finite.sum()))


def comparison(before, after):
    old, new = metrics(before), metrics(after)
    improvement = (100 * (1 - new["rms_percent"] / old["rms_percent"])
                   if old["rms_percent"] and new["rms_percent"] is not None else None)
    return dict(before=old, after=new, rms_reduction_percent=improvement)


def order_metrics(before, after, order, edge_pixels):
    floor = float(np.asarray(after.get("continuum_coefficient_floor", 1e-8)))
    continuum = after["continuum_fixed_scale"][order]
    row = dict(cube_index=order,
               wavelength_min_um=float(after["wavelength_um"][order].min()),
               wavelength_max_um=float(after["wavelength_um"][order].max()),
               minimum_continuum_fixed_scale=float(continuum.min()),
               continuum_floor=floor,
               near_floor_samples=int(np.count_nonzero(continuum <= 1.01 * floor)),
               near_floor_frames=np.flatnonzero(np.any(continuum <= 1.01 * floor, axis=1)).tolist(),
               metrics={})
    if "continuum_active_lower_bounds" in after:
        active = after["continuum_active_lower_bounds"][order]
        row["active_coefficient_bounds_per_frame"] = active.sum(axis=-1).tolist()
        row["active_coefficient_bound_frames"] = np.flatnonzero(active.any(axis=-1)).tolist()
    for name, key in [("fixed_scale_q5", "residual_fixed_scale_q5_independent"),
                      ("continuum_q5", "residual_continuum_q5")]:
        result = row["metrics"].setdefault(name, {})
        for frame_name, frames in [("all_frames", np.arange(79)), ("evaluation_frames", EVALUATION)]:
            result[frame_name] = {}
            for region, selection in [("all_pixels", slice(None)),
                                      ("right_edge", slice(-edge_pixels, None)),
                                      ("interior", slice(edge_pixels, -edge_pixels))]:
                result[frame_name][region] = comparison(
                    before[key][order, frames, selection], after[key][order, frames, selection])
    return row


def style_axis(axis):
    axis.grid(alpha=.15)
    axis.spines[["top", "right"]].set_visible(False)


def symlog_axis(axis, threshold):
    axis.set_yscale("symlog", linthresh=threshold)
    locator = SymmetricalLogLocator(base=10, linthresh=threshold)
    locator.set_params(numticks=7)
    axis.yaxis.set_major_locator(locator)
    def label(value, position):
        if value == 0:
            return "0"
        if abs(value) < threshold:
            return ""
        sign = "-" if value < 0 else ""
        return rf"${sign}10^{{{int(round(np.log10(abs(value))))}}}$"
    axis.yaxis.set_major_formatter(FuncFormatter(label))


def make_figure(before, after, row, frame, edge_pixels):
    order = row["cube_index"]
    nu = after["wavenumber_cm1"][order]
    continuum = after["continuum_fixed_scale"][order, frame]
    observed = after["observed_fixed_scale"][order, frame]
    old_residual = before["residual_fixed_scale_q5_independent"][order, frame]
    new_residual = after["residual_fixed_scale_q5_independent"][order, frame]
    old_prediction, new_prediction = observed - old_residual, observed - new_residual
    fig, axes = plt.subplots(4, 1, figsize=(14, 12),
                             gridspec_kw=dict(height_ratios=[2.1, 1.1, 1.5, 1.1]))
    fig.subplots_adjust(left=.085, right=.98, top=.90, bottom=.13, hspace=.48)
    label = f"Stored cube index {order}" + (" (user Order 7)" if order == 5 else "")
    fig.suptitle(f"WASP-77Ab / IGRINS: {label} — continuum revision", fontsize=15, weight="bold", y=.973)
    m = row["metrics"]["fixed_scale_q5"]["evaluation_frames"]
    fig.text(.5, .939, f"Frame {frame} | {row['wavelength_min_um']:.8f}–{row['wavelength_max_um']:.8f} microns | "
             f"evaluation RMS {m['all_pixels']['before']['rms_percent']:.3f}% → "
             f"{m['all_pixels']['after']['rms_percent']:.3f}% of frame p95 flux", ha="center")
    series = [(observed, "black", "-", .7, "Observed"),
              (old_prediction, "#377eb8", "-", .9, "Previous model + Q5"),
              (new_prediction, "#c53932", "-", .9, "Refined model + Q5"),
              (after["stellar_only"][order, frame] * continuum, "#238b45", "--", .8, "Stellar only × new continuum"),
              (after["telluric_only"][order, frame] * continuum, "#d58b19", ":", .8, "Telluric only × new continuum")]
    for values, color, linestyle, width, text in series:
        axes[0].plot(nu, values, color=color, ls=linestyle, lw=width, label=text)
    axes[0].set(ylabel="Flux / frame p95", xlim=(nu[0], nu[-1]))
    axes[0].legend(fontsize=8, ncol=3, loc="best")
    for residual, color, text in [(old_residual, "#377eb8", "Previous"),
                                  (new_residual, "#c53932", "Refined")]:
        axes[1].plot(nu, 100 * residual, color=color, lw=.7, label=text)
    axes[1].set(ylabel="Residual (% of p95)", xlim=(nu[0], nu[-1]),
                xlabel=r"Wavenumber (cm$^{-1}$)")
    axes[1].legend(fontsize=8, ncol=2, loc="best")
    edge = slice(-edge_pixels, None)
    for values, color, linestyle, width, text in series[:3]:
        axes[2].plot(nu[edge], values[edge], color=color, ls=linestyle, lw=width, label=text)
    axes[2].set(ylabel="Flux / frame p95", xlim=(nu[-edge_pixels], nu[-1]),
                title=f"Rightmost {edge_pixels} pixels: increasing wavenumber, decreasing wavelength")
    axes[2].legend(fontsize=8, ncol=3, loc="best")
    axes[3].plot(nu[edge], 100 * old_residual[edge], color="#377eb8", lw=.8, label="Previous")
    axes[3].plot(nu[edge], 100 * new_residual[edge], color="#c53932", lw=.8, label="Refined")
    axes[3].set(ylabel="Residual (% of p95)", xlim=(nu[-edge_pixels], nu[-1]),
                xlabel=r"Wavenumber (cm$^{-1}$)")
    for axis in axes:
        style_axis(axis)
    for axis in [axes[1], axes[3]]:
        axis.axhline(0, color="black", lw=.5)
    for axis in axes[:2]:
        axis.axvspan(nu[-edge_pixels], nu[-1], color=".8", alpha=.2)
    flag = (f"Continuum reaches its numerical floor in {len(row['near_floor_frames'])} frames; "
            "these points remain included." if row["near_floor_samples"] else
            f"Minimum new continuum: {row['minimum_continuum_fixed_scale']:.3g} of frame p95 flux.")
    if row.get("active_coefficient_bound_frames"):
        flag += f" Coefficient bounds active in {len(row['active_coefficient_bound_frames'])} frames."
    fig.text(.5, .035,
             "Every panel uses the same fixed observational scale. Q5 is fitted directly in that scale. No continuum division in these plots.\n"
             "Physical, stellar and line-strength parameters are frozen; only the smooth continuum is revised. No injection or enuma input.\n"
             + flag, ha="center", fontsize=8)
    return fig


def make_normalized_figure(after, row, frame):
    """Show only the current fit in the original three-panel layout."""
    from summarize_all_orders import bounds, shade

    order = row["cube_index"]
    nu = after["wavenumber_cm1"][order]
    pixel = np.arange(len(nu))
    score = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    strong = np.min(after["telluric_only"][order, [0, 19, 39, 59, 78]], axis=0) <= .5
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw=dict(height_ratios=[2.4, 1.1, 1.2]))
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.15, hspace=.22)
    label = f"Stored cube index {order}" + (" (user Order 7)" if order == 5 else "")
    fig.suptitle(f"WASP-77Ab / IGRINS: {label}", fontsize=15, weight="bold", y=.975)
    metric = row["metrics"]["continuum_q5"]["evaluation_frames"]["all_pixels"]["after"]
    fig.text(.5, .935,
             f"{row['wavelength_min_um']:.8f}–{row['wavelength_max_um']:.8f} microns | "
             f"all {len(nu)} stored pixels | matched-frame RMS {metric['rms_percent']:.4f}%",
             ha="center")
    values = [after["observed_continuum"][order, frame],
              after["full_model"][order, frame] + after["smooth_continuum_q5"][order, frame],
              after["stellar_only"][order, frame], after["telluric_only"][order, frame]]
    for y, color, style, width, text in zip(
            values, ["black", "#c53932", "#238b45", "#d58b19"],
            ["-", "-", "--", ":"], [.65, 1., 1., .9],
            [f"Observed, frame {frame}", "Current model + Q5", "Stellar model alone", "Telluric alone"]):
        axes[0].plot(nu, y, color=color, ls=style, lw=width, label=text)
    axes[0].set(ylabel="Flux / continuum", ylim=bounds(values))
    axes[0].legend(fontsize=9, ncol=2, loc="lower left")
    residual = 100 * after["residual_continuum_q5"][order, frame]
    axes[1].plot(nu, residual, color=".65", lw=.65)
    axes[1].scatter(nu[score], residual[score], s=5, color="#2166ac", zorder=3)
    axes[1].set(ylabel=f"Frame {frame} residual (%)", ylim=bounds([residual]))
    evaluation = 100 * after["residual_continuum_q5"][order, EVALUATION]
    for exposure, y, color in zip(EVALUATION, evaluation,
                                   ["#1b9e77", "#7570b3", "#d95f02", "#666666"]):
        axes[2].plot(nu, y, color=color, lw=.6, alpha=.75, label=f"Frame {exposure}")
    axes[2].plot(nu, evaluation.mean(axis=0), color="black", lw=1.1, label="Mean")
    axes[2].set(ylabel="Evaluation residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)",
                ylim=bounds([evaluation]))
    axes[2].legend(fontsize=8, ncol=5, loc="lower left")
    log_axes = []
    for index, (axis, y, threshold, trigger) in enumerate(zip(
            axes, [np.asarray(values), residual, evaluation], [1., 5., 5.], [5., 100., 100.])):
        if np.max(np.abs(y)) > trigger:
            symlog_axis(axis, threshold)
            log_axes.append(index + 1)
        shade(axis, nu, strong)
        axis.set_xlim(nu[0], nu[-1])
        style_axis(axis)
    for axis in axes[1:]:
        axis.axhline(0, color="black", lw=.5)
    flag = (f"Continuum near its numerical floor at {row['near_floor_samples']} samples across all frames. "
            if row["near_floor_samples"] else "")
    if log_axes:
        flag += f"Symmetric-log panels {log_axes}; thresholds: flux 1, residual 5%. No finite samples clipped."
    fig.text(.5, .035,
             "Blue points: original score subset; gray-only points also included. Yellow: strong modeled telluric absorption, included.\n"
             "Current continuum and degree-five residual projection use all pixels. No injection or enuma input.\n"
             + flag, ha="center", fontsize=8)
    return fig


def main(args):
    before, after = load_product(args.before), load_product(args.after)
    for key in ["orders", "exposures", "wavelength_um", "wavenumber_cm1",
                "observed_fixed_scale", "full_model", "stellar_only", "telluric_only"]:
        np.testing.assert_array_equal(before[key], after[key], err_msg=f"Frozen array changed: {key}")
    if np.any(after["continuum_fixed_scale"] <= 0):
        raise ValueError("Refined continuum is not strictly positive")
    if not 0 <= args.frame < 79 or not 1 <= args.edge_pixels < 924:
        raise ValueError("Invalid frame or edge size")
    audits = dict(before_q5_max_error=audit_q5(before), after_q5_max_error=audit_q5(after))
    rows = [order_metrics(before, after, order, args.edge_pixels) for order in range(44)]
    pooled = {}
    for scope, frames in [("all_frames", np.arange(79)), ("evaluation_frames", EVALUATION)]:
        pooled[scope] = {}
        for region, selection in [("all_pixels", slice(None)),
                                  ("right_edge", slice(-args.edge_pixels, None)),
                                  ("interior", slice(args.edge_pixels, -args.edge_pixels))]:
            key = "residual_fixed_scale_q5_independent"
            pooled[scope][region] = comparison(before[key][:, frames, selection], after[key][:, frames, selection])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  before=dict(path=str(args.before), sha256=digest(args.before)),
                  after=dict(path=str(args.after), sha256=digest(args.after)),
                  report_script_sha256=digest(__file__), shape=[44, 79, 1848],
                  figure_frame=args.frame, evaluation_frames=EVALUATION,
                  normalized_figures=args.normalized,
                  normalized_figure_format="Current fit only: spectrum, frame residual, evaluation residuals",
                  right_edge_pixels=args.edge_pixels,
                  right_edge_definition="Last stored pixels: increasing wavenumber, decreasing wavelength",
                  units=dict(fixed_scale_q5="Percent of each frame's unchanged all-pixel p95 observed flux",
                             continuum_q5="Percent of each product's fitted continuum; denominators differ"),
                  frozen_physical_models=True, artificial_injection=False, enuma_used=False,
                  comparison_note="Models in figures equal observed_fixed_scale minus independent fixed-scale Q5 residual. Stellar and telluric curves are each scaled by the new continuum.",
                  limitation="Only continuum terms are revised. Shared physical fits are not reoptimized, and planet-signal recovery is not established.",
                  audit=audits, pooled_fixed_scale_q5=pooled, rows=rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.prefix
    prefix.with_name(prefix.name + "_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = ["cube_index", "all_frames_before_percent", "all_frames_after_percent",
               "evaluation_before_percent", "evaluation_after_percent", "evaluation_right_edge_before_percent",
               "evaluation_right_edge_after_percent", "minimum_continuum_fixed_scale", "near_floor_samples"]
    with prefix.with_name(prefix.name + "_metrics.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            m = row["metrics"]["fixed_scale_q5"]
            flat = {key: row[key] for key in ["cube_index", "minimum_continuum_fixed_scale", "near_floor_samples"]}
            for name, scope, region in [("all_frames", "all_frames", "all_pixels"),
                                         ("evaluation", "evaluation_frames", "all_pixels"),
                                         ("evaluation_right_edge", "evaluation_frames", "right_edge")]:
                for stage in ["before", "after"]:
                    flat[f"{name}_{stage}_percent"] = m[scope][region][stage]["rms_percent"]
            writer.writerow(flat)
    with PdfPages(prefix.with_name(prefix.name + "_spectral_fits.pdf")) as pdf:
        for row in rows:
            figure = make_figure(before, after, row, args.frame, args.edge_pixels)
            pdf.savefig(figure)
            if not args.no_png:
                figure.savefig(prefix.with_name(f"{prefix.name}_o{row['cube_index']:02d}_spectral_fit.png"), dpi=140)
            plt.close(figure)
    if args.normalized:
        with PdfPages(prefix.with_name(prefix.name + "_normalized_spectral_fits.pdf")) as pdf:
            for row in rows:
                figure = make_normalized_figure(after, row, args.frame)
                pdf.savefig(figure)
                if not args.no_png:
                    figure.savefig(prefix.with_name(f"{prefix.name}_normalized_o{row['cube_index']:02d}_spectral_fit.png"), dpi=140)
                plt.close(figure)
    print(json.dumps(dict(output_prefix=str(prefix), audit=audits, pooled_fixed_scale_q5=pooled), indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, default=ROOT / "results/allorders_ccf_no_injection.npz")
    parser.add_argument("--after", type=Path, default=ROOT / "results/allorders_ccf_continuum_refined.npz")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--prefix", default="allorders_continuum_refined")
    parser.add_argument("--frame", type=int, default=30)
    parser.add_argument("--edge-pixels", type=int, default=128)
    parser.add_argument("--no-png", action="store_true")
    parser.add_argument("--normalized", action="store_true",
                        help="Also show the current normalized fit in the original three-panel format")
    main(parser.parse_args())
