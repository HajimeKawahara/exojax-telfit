"""Report the full released Order 7 band without absorption-depth exclusions."""

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


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
STAGES = ("baseline", "stellar", "refit")
LABELS = ("Fixed PHOENIX + Q5", "Stellar calibration + Q5", "Stellar + telluric strengths + Q5")
BLUE, RED, GREEN, ORANGE = "#2166ac", "#c53932", "#238b45", "#d58b19"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):
    values = np.asarray(values)
    return float(100 * np.sqrt(np.mean(values**2))) if values.size else None


def q5(raw, wave):
    """Independently remove six polynomial modes using a QR decomposition."""
    x = 2 * (np.log(wave) - np.log(wave).min()) / np.ptp(np.log(wave)) - 1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    orthogonal, triangular = np.linalg.qr(basis, mode="reduced")
    coefficients = np.linalg.solve(triangular, orthogonal.T @ raw.T)
    smooth = (basis @ coefficients).T
    return raw - smooth, smooth


def limits(values, fraction=.08):
    values = np.concatenate([np.asarray(v).ravel() for v in values])
    lo, hi = float(values.min()), float(values.max())
    padding = max(fraction * (hi - lo), .01)
    return lo - padding, hi + padding


def shade(ax, wave, mask):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    edges = np.r_[wave[0] - (wave[1] - wave[0]) / 2,
                  (wave[:-1] + wave[1:]) / 2,
                  wave[-1] + (wave[-1] - wave[-2]) / 2]
    for group in groups:
        if len(group):
            ax.axvspan(edges[group[0]], edges[group[-1] + 1], color="#ffe5aa", alpha=.55, zorder=0)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT / "order7_stellar_fit.npz"
    metadata = json.loads(source.with_suffix(".json").read_text())
    scope = json.loads((OUT / "order7_scope_audit.json").read_text())
    physical = json.loads((OUT / "order7_physical_e10.json").read_text())
    species_diagnostic = json.loads((OUT / "order7_physical_species.json").read_text())
    precision_path = OUT / "order7_numerical_precision.json"
    precision = json.loads(precision_path.read_text()) if precision_path.exists() else None
    independent_path = OUT / "order7_independent_audit.json"
    independent = json.loads(independent_path.read_text())
    assert independent["passed"]
    with np.load(source) as saved:
        a = {key: saved[key] for key in saved.files}
    assert sha256(source) == metadata["product_sha256"]
    np.testing.assert_array_equal(a["orders"], [5])
    np.testing.assert_array_equal(a["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    assert a["included_mask"].shape == (1, 1848) and a["included_mask"].all()
    wave = a["wavelength_um"][0]
    nu = 1e4 / wave
    score = a["score_mask"][0]
    pixel = np.arange(len(wave))
    gate = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    np.testing.assert_array_equal(score, gate)
    assert score.sum() == 773
    observed = a["y_over_c"][0]
    residuals, fitted, metrics = {}, {}, {}
    for stage in STAGES:
        raw = observed - a[f"component_{stage}"][0]
        residual, smooth = q5(raw, wave)
        residuals[stage] = residual
        fitted[stage] = a[f"component_{stage}"][0] + smooth
        metrics[stage] = dict(training_all_percent=rms(residual[:5]),
                              evaluation_score_percent=rms(residual[5:, score]),
                              evaluation_all_percent=rms(residual[5:]),
                              evaluation_per_frame_all_percent=[rms(row) for row in residual[5:]])
    raw_baseline = observed - a["component_baseline"][0]
    metrics["baseline_unfiltered"] = dict(training_all_percent=rms(raw_baseline[:5]),
        evaluation_score_percent=rms(raw_baseline[5:, score]),
        evaluation_all_percent=rms(raw_baseline[5:]))
    stellar = a["stellar_only"][0]
    telluric = a["telluric_only"][0]
    frozen_telluric = np.min(a["telluric_only_baseline"][0, :5], axis=0)
    frozen_star = np.min(a["stellar_only_baseline"][0, :5], axis=0)
    edges = [.15, .50, .80, .95]
    bin_ids = np.digitize(frozen_telluric, edges, right=True)
    bin_labels = ["T <= 0.15", "0.15 < T <= 0.50", "0.50 < T <= 0.80",
                  "0.80 < T <= 0.95", "T > 0.95"]

    def region(name, mask):
        return dict(name=name, wavelength_pixels=int(mask.sum()),
                    evaluation_samples=int(mask.sum()) * 4,
                    score_pixels=int((mask & score).sum()),
                    baseline_rms_percent=rms(residuals["baseline"][5:, mask]),
                    final_rms_percent=rms(residuals["refit"][5:, mask]))

    bins = [region(name, bin_ids == i) for i, name in enumerate(bin_labels)]
    stellar_regions = [region(f"Stellar depth > {100 * depth:g}%", 1 - frozen_star > depth)
                       for depth in [.02, .10]]
    strong_telluric = frozen_telluric <= .5
    frame = int(np.flatnonzero(a["exposures"] == 30)[0])
    residual = residuals["refit"]
    final_metrics = metrics["refit"]
    plt.rcParams.update({"font.size": 11, "font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1, 1.2]})
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.16, hspace=.27)
    fig.suptitle("WASP-77Ab / IGRINS Order 7: strong absorption included", y=.98,
                 fontsize=16, weight="bold")
    fig.text(.5, .944, "Released cube index 5 | all 1,848 stored pixels | evaluation frame 30",
             ha="center", fontsize=11)
    handles = [Line2D([], [], color="black", lw=1, label="Observed"),
               Line2D([], [], color=RED, lw=1.3, label="Final model + Q5 smooth correction"),
               Line2D([], [], color=GREEN, ls="--", lw=1.3, label="Stellar model alone"),
               Line2D([], [], color=ORANGE, ls=":", lw=1.2, label="Telluric transmission alone"),
               Patch(facecolor="#ffe5aa", label="Strong telluric absorption: included"),
               Line2D([], [], color=BLUE, marker=".", ls="", label="Score subset")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .925),
               ncol=3, frameon=False, fontsize=10)
    top_values = [observed[frame], fitted["refit"][frame], stellar[frame], telluric[frame]]
    upper_limits = limits(top_values)
    axes[0].plot(nu, observed[frame], color="black", lw=.65)
    axes[0].plot(nu, fitted["refit"][frame], color=RED, lw=1.05)
    axes[0].plot(nu, stellar[frame], color=GREEN, ls="--", lw=1.1)
    axes[0].plot(nu, telluric[frame], color=ORANGE, ls=":", lw=.9)
    axes[0].set(ylim=upper_limits, ylabel="Flux / continuum")
    axes[0].set_title(f"Four-frame RMS: all included {final_metrics['evaluation_all_percent']:.3f}%"
                      f" | score subset {final_metrics['evaluation_score_percent']:.3f}%", loc="left")
    values = residual[frame] * 100
    axes[1].plot(nu, values, color=".65", lw=.65)
    axes[1].scatter(nu[score], values[score], s=5, color=BLUE, zorder=3)
    frame_residual_limits = limits([values])
    axes[1].set(ylabel="Frame 30\nresidual (%)", ylim=frame_residual_limits)
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#666666"]
    for e, row, color in zip([10, 30, 50, 70], residual[5:], colors):
        axes[2].plot(nu, row * 100, lw=.6, alpha=.75, color=color, label=f"Frame {e}")
    axes[2].plot(nu, residual[5:].mean(axis=0) * 100, color="black", lw=1.1,
                 label="Four-frame mean")
    evaluation_residual_limits = limits([residual[5:] * 100])
    axes[2].set(ylabel="Evaluation\nresidual (%)", xlabel=r"Wavenumber (cm$^{-1}$)",
                ylim=evaluation_residual_limits)
    axes[2].legend(ncol=5, fontsize=8, loc="upper right", framealpha=.85)
    for ax in axes:
        shade(ax, nu, strong_telluric)
        ax.set_xlim(nu.min(), nu.max())
        clean(ax)
    for ax in axes[1:]:
        ax.axhline(0, color="black", lw=.5)
    fig.text(.5, .022, "Yellow bands mark frozen reference transmission <= 0.50; they are fitted and evaluated. No absorption-depth exclusions.\n"
             "Stellar/telluric curves include the instrumental response separately. Residuals use all-pixel degree-five filtering.",
             ha="center", fontsize=10)
    fig.savefig(OUT / "order7_spectral_fit.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [1, 1.25]})
    fig.subplots_adjust(left=.085, right=.98, top=.88, bottom=.14, hspace=.45)
    fig.suptitle("Order 7: fit quality across absorption depths", fontsize=15, weight="bold", y=.98)
    fig.text(.5, .941, "Depth classes are frozen from the initial model on five training frames; every stored pixel is retained", ha="center", fontsize=10)
    axes[0].plot(nu, frozen_telluric, color=ORANGE, lw=.9, label="Minimum reference telluric transmission")
    axes[0].plot(nu, frozen_star, color=GREEN, lw=.9, ls="--", label="Minimum reference stellar flux")
    axes[0].set(xlim=(nu.min(), nu.max()), ylim=limits([frozen_telluric, frozen_star]),
                xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Continuum fraction")
    axes[0].legend(loc="lower left", fontsize=9)
    positions = np.arange(len(bins))
    for offset, stage, color, label in [(-.18, "baseline", ".6", LABELS[0]),
                                       (.18, "refit", BLUE, LABELS[2])]:
        values = [r[f"{'baseline' if stage == 'baseline' else 'final'}_rms_percent"] for r in bins]
        axes[1].bar(positions + offset, [v if v is not None else 0 for v in values],
                    width=.36, color=color, label=label)
    axes[1].set_xticks(positions, [f"{name}\n{row['wavelength_pixels']} wavelengths" for name, row in zip(bin_labels, bins)], fontsize=9)
    axes[1].set_ylabel("Evaluation RMS (% continuum)")
    axes[1].legend(fontsize=9)
    noise = scope["metadata_noise_proxy"]["pooled_percent"]
    axes[1].axhline(noise, color=".35", ls=":", lw=1, label="Metadata noise proxy")
    for i, row in enumerate(bins):
        if row["wavelength_pixels"] == 0:
            axes[1].text(i, .02, "No pixels", transform=axes[1].get_xaxis_transform(), ha="center", fontsize=9)
    for ax in axes:
        clean(ax)
    fig.text(.5, .035, "All depth-bin RMS values use all four evaluation frames and all wavelengths in the bin. Empty bins are not measurements.\n"
             f"Dotted line: {noise:.3f}% global metadata S/N proxy; no per-pixel photon variance is provided.", ha="center", fontsize=9)
    fig.savefig(OUT / "order7_absorption_depth.png", dpi=170)
    plt.close(fig)

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2, left=.08, right=.98, bottom=.18, top=.84,
                          height_ratios=[1, 1.15], hspace=.45, wspace=.24)
    fig.suptitle("Order 7: residuals across the four evaluation frames", y=.98,
                 fontsize=15, weight="bold")
    fig.text(.5, .943, "All 1,848 pixels retained | no residual or absorption-depth exclusions", ha="center")
    legend = [Line2D([], [], color=color, lw=1, label=f"Frame {e}")
              for color, e in zip(colors, [10, 30, 50, 70])]
    legend.append(Line2D([], [], color="black", lw=1.5, label="Four-frame mean"))
    fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .92), ncol=5, frameon=False)
    windows = [("Full stored order", np.ones(len(wave), dtype=bool), None)]
    for label, reference in [("Deepest reference telluric absorption", frozen_telluric),
                             ("Deepest reference stellar absorption", frozen_star)]:
        center = float(nu[np.argmin(reference)])
        windows.append((label, abs(nu - center) <= .60, center))
    zoom_records = []
    for i, (label, window, center) in enumerate(windows):
        ax = fig.add_subplot(gs[0, :] if i == 0 else gs[1, i-1])
        for color, row in zip(colors, residual[5:]):
            ax.plot(nu, row * 100, color=color, lw=.65, alpha=.8,
                    marker="." if i else None, markersize=3)
        ax.plot(nu, residual[5:].mean(axis=0) * 100, color="black", lw=1.35)
        bounds = limits([residual[5:, window] * 100])
        ax.set(xlim=(nu[window].min(), nu[window].max()), ylim=bounds,
               xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Residual (% continuum)")
        ax.set_title(label + (f"\n{center:.4f} cm$^{{-1}}$" if center is not None else ""), loc="left", fontsize=11)
        ax.axhline(0, color="black", lw=.5)
        shade(ax, nu, strong_telluric)
        clean(ax)
        selected = residual[5:, window] * 100
        zoom_records.append(dict(label=label, center_cm1=center,
            wavelength_pixels=int(window.sum()), final_rms_percent=rms(residual[5:, window]),
            axis_limits_percent=list(bounds),
            clipped_points=int(np.sum((selected < bounds[0]) | (selected > bounds[1])))))
    fig.text(.5, .025, "Zoom centers are selected from the frozen baseline models in the training frames, not from evaluation residuals.\n"
             "Yellow: reference telluric transmission <= 0.50. The four-frame mean is diagnostic only; it is not subtracted.", ha="center", fontsize=10)
    fig.savefig(OUT / "order7_residuals.png", dpi=170)
    plt.close(fig)

    comparison_stages = ("baseline_unfiltered",) + STAGES
    comparison_labels = ("Physical PHOENIX fit, before Q5",) + LABELS
    positions = np.arange(len(comparison_stages))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.3, 1]})
    fig.subplots_adjust(left=.34, right=.98, bottom=.21, top=.78, wspace=.25)
    fig.suptitle("Order 7: matched fitting stages", y=.97, fontsize=15, weight="bold")
    fig.text(.5, .91, "Identical 1,848 wavelengths and four evaluation frames in every row", ha="center", fontsize=11)
    all_rms = np.array([metrics[stage]["evaluation_all_percent"] for stage in comparison_stages])
    score_rms = np.array([metrics[stage]["evaluation_score_percent"] for stage in comparison_stages])
    axes[0].barh(positions - .16, all_rms, height=.30, color=BLUE, label="All included (primary)")
    axes[0].barh(positions + .16, score_rms, height=.30, color="#8db6d9", label="Score subset")
    axes[0].set_yticks(positions, comparison_labels, fontsize=10)
    axes[0].set_xlabel("Evaluation RMS (% continuum)")
    axes[0].set_xlim(0, 1.23 * max(all_rms.max(), score_rms.max()))
    axes[0].invert_yaxis()
    axes[0].legend(fontsize=9, loc="lower right")
    reference_rms = metrics["baseline"]["evaluation_all_percent"]
    reductions = 100 * (1 - all_rms**2 / reference_rms**2)
    axes[1].barh(positions, reductions, height=.50, color=[".65", ".65", GREEN, GREEN])
    axes[1].set_yticks(positions, [""] * len(positions))
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=".3", lw=.8)
    axes[1].set_xlabel("Squared-residual reduction (%)\nrelative to PHOENIX + Q5, all pixels")
    left, right = min(0, reductions.min()), max(0, reductions.max())
    pad = max(8., .22 * (right - left))
    axes[1].set_xlim(left - pad, right + pad)
    for i, (all_value, score_value, reduction) in enumerate(zip(all_rms, score_rms, reductions)):
        axes[0].text(all_value + .015, i-.16, f"{all_value:.3f}", va="center", fontsize=9)
        axes[0].text(score_value + .015, i+.16, f"{score_value:.3f}", va="center", fontsize=9)
        axes[1].text(reduction + (1 if reduction >= 0 else -1), i, f"{reduction:+.2f}",
                     va="center", ha="left" if reduction >= 0 else "right", fontsize=9)
    for ax in axes:
        clean(ax)
    fig.text(.5, .035, "Q5 removes a smooth degree-five component in each frame. Lower residual RMS alone does not establish planet-signal preservation.", ha="center", fontsize=9)
    fig.savefig(OUT / "order7_comparison.png", dpi=170)
    plt.close(fig)

    def percent(value):
        return f"{value:.6f}%" if value is not None else "n/a"

    free = dict(stellar_amplitudes=metadata["stellar_free_features"],
                telluric_strengths=metadata["free_strength_count"],
                telluric_strengths_by_species=metadata["free_strength_species"],
                stellar_calibration_pixels=metadata["stellar_calibration_pixels"])
    species_counts = ", ".join(f"{name}: {count}" for name, count in free["telluric_strengths_by_species"].items())
    stellar_max = float(stellar.max())
    stellar_above_continuum = int(np.sum(stellar > 1.001))
    lines = [
        f"The current **Order 7** fit reaches **{final_metrics['evaluation_all_percent']:.4f}% residual RMS across the complete stored order**. "
        f"All **1,848 wavelengths** are fitted and evaluated, including strong stellar and terrestrial absorption. "
        f"The secondary fixed score subset gives {final_metrics['evaluation_score_percent']:.4f}% RMS.", "",
        "The goal is to remove Earth's atmospheric absorption while preserving the much weaker, Doppler-shifting planetary spectrum. "
        "This report concerns only the newly requested Order 7 band.", "",
        "### Data and evaluation", "",
        "The data are the 2020-12-14 Gemini South/IGRINS observations of WASP-77A: "
        "[public data](https://zenodo.org/records/10382053), "
        "[observations and reduction](https://arxiv.org/html/2312.13069v1#S2). "
        "User Order 7 maps by wavelength to **released cube index 5**. The requested "
        "2.26448634–2.29592734 micron band is available as **2.26622167–2.29458395 microns** "
        "after the archive's edge trimming. Missing edge flux is not synthesized.", "",
        "Shared corrections use five training frames: **0, 19, 39, 59, 78**. "
        "Evaluation uses four other frames: **10, 30, 50, 70**. Indices are zero-based.", "",
        "The **primary all-included evaluation uses 1,848 × 4 = 7,392 samples**. "
        "A secondary fixed geometric score subset uses 773 × 4 = 3,092 samples. "
        "That subset consists of alternating wavelength blocks inherited from the nuisance-calibration convention; "
        "it does not exclude absorption by depth. No new stellar-depth, telluric-depth, residual or edge mask is applied.", "",
        "### Spectral fit", "",
        "![Observed spectrum, full-order fit, stellar and telluric components, and residuals](ASSET_URL/order7_spectral_fit.png)", "",
        "**Upper panel:** frame 30 observations (black), final model plus degree-five smooth correction (red), "
        "stellar-only model (green dashed) and telluric-only transmission (orange dotted). "
        "**Middle panel:** residuals for the same frame; blue points mark the secondary score subset. "
        "**Lower panel:** the four evaluation residuals and their mean. "
        "Yellow bands mark strong reference telluric absorption (transmission ≤ 0.50), which remains included. "
        "Plot limits contain all displayed samples.", "",
        "### Matched configuration comparison", "",
        "Every row below uses the same wavelengths, frames and original fitted continuum. "
        "RMS is in percent of that continuum. Q5 removes a degree-five polynomial from each order/frame residual.", "",
        "| Stage | Training all | Evaluation all included (primary) | Evaluation score subset | Photon-noise estimate† |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for stage, label in zip(comparison_stages, comparison_labels):
        m = metrics[stage]
        lines.append(f"| {label} | {percent(m['training_all_percent'])} | {percent(m['evaluation_all_percent'])} | {percent(m['evaluation_score_percent'])} | ≈{noise:.3f}% |")
    lines += ["", "† The reference noise scale is `sqrt(mean((100 / Med. SNR)^2))`, using the four evaluation frames. "
        "It assumes photon-dominated S/N. One median S/N is supplied per frame, so this is not an order-specific "
        "or deep-line photon-noise measurement. Per-pixel variance, electron counts and gain are unavailable. "
        "The value is before Q5, which changes the covariance; it is not divided by √4 because the RMS pools individual frames.", "",
        "![Matched fitting stages, residual RMS and squared-residual reduction](ASSET_URL/order7_comparison.png)", "",
        "The before/after-Q5 comparison measures how much smooth mismatch is removed. "
        "The next two rows isolate the changes from calibrating the stellar amplitudes and then refitting telluric strengths. "
        "These gains are conditional on the earlier physical and nuisance fits.", "",
        "### Remaining residuals and strong absorption", "",
        "![Four-frame residuals and model-selected strong-absorption close-ups](ASSET_URL/order7_residuals.png)", "",
        "The two close-ups are centered on the deepest reference telluric and stellar features. "
        "Their centers are selected from the initial models in the training frames, without using evaluation residuals. "
        "The time mean is displayed for diagnosis and is not subtracted.", "",
        "| Frozen reference region | Wavelengths | Evaluation samples | PHOENIX + Q5 RMS | Final RMS |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for row in bins + stellar_regions:
        lines.append(f"| {row['name']} | {row['wavelength_pixels']} | {row['evaluation_samples']} | {percent(row['baseline_rms_percent'])} | {percent(row['final_rms_percent'])} |")
    lines += ["", "Telluric classes use the minimum initial telluric-only transmission across the five training frames. "
        "Stellar classes use the analogous initial stellar-only model; the >10% stellar set is contained within the >2% set. "
        "These fixed classes locate residuals and do not uniquely identify their physical cause. Empty bins are reported as n/a.", "",
        "The **physical initialization** identifies methane as the main strong telluric absorber in this band. "
        f"In frame 30, isolated CH4 absorption reaches {species_diagnostic['per_species']['CH4']['maximum_absorption_percent']:.2f}%, "
        f"and CH4 dominates all {species_diagnostic['combined_pixels_below_0_5']} pixels with combined transmission below 0.50. "
        f"The isolated H2O and N2O maxima are {species_diagnostic['per_species']['H2O']['maximum_absorption_percent']:.2f}% "
        f"and {species_diagnostic['per_species']['N2O']['maximum_absorption_percent']:.2f}%, at different wavelengths. "
        "These modeled absorption depths explain the band content; they do not assign the residual to a unique molecule. "
        "Separately convolved species transmissions do not multiply exactly to the combined convolved spectrum.", "",
        "![Residual RMS by frozen absorption-depth class](ASSET_URL/order7_absorption_depth.png)", "",
        "### Forward model and free parameters", "",
        "The model multiplies the Doppler-shifted stellar spectrum by terrestrial transmission before applying the IP. "
        "It includes cached **HITRAN2024 H2O, CH4, N2O, CO and CO2** data, pressure shifts, and a **20-layer** profile based on dated "
        "NCEP/NCAR reanalysis for **2020-12-14**, with explicit upper-atmosphere extensions. "
        "The IP is a symmetric two-Gaussian mixture. The stellar reference is PHOENIX at 5,600 K, "
        "log g = 4.5 and solar metallicity, with rotation fixed at 4 km/s.", "",
        "| Stage / quantity | Treatment |",
        "| --- | --- |",
        f"| Physical initialization | {len(physical['training_fit']['optimized_parameters'])} nonlinear parameters: IP core width and wing fraction; wavelength offset/stretch; four dry-species scales; five water amplitudes; four relative epoch wavelength offsets and four width changes |",
        f"| Stellar kinematics | Reference RV {physical['parameters']['stellar_rv_reference_kms']:.6f} km/s and extra Gaussian width (sigma) {physical['parameters']['stellar_extra_sigma_kms']:.6f} km/s fixed from earlier calibration; relative frame velocities follow the metadata |",
        f"| Stellar amplitudes | {free['stellar_amplitudes']} shared feature amplitudes fitted on {free['stellar_calibration_pixels']} telluric-clear wavelengths (reference T > 0.95 in all training frames) |",
        f"| Telluric line strengths | {free['telluric_strengths']} shared corrections, refitted with the calibrated star fixed ({species_counts}) |",
        "| Stellar feature centers and widths; telluric centers and widths | Fixed during the amplitude-correction stages |",
        "| Atmospheric profile, physical columns, wavelength solution, IP and original continua | Fixed during the amplitude-correction stages |",
        "| Smooth residual | Six degree-five polynomial coefficients per frame, solved on all 1,848 pixels |", "",
        "The actual [blasé JAX clone](https://github.com/gully/blase/blob/6f6817b0c4095c104a10b2940375fc1e16cb80ca/src/blase/jaxemulator.py) "
        "supplies an amplitude correction anchored to PHOENIX: "
        "`PHOENIX × clone(adjusted amplitudes) / clone(reference amplitudes)`. "
        "Telluric strengths remain nonnegative and use HITRAN-derived uncertainty scales with contrast-dependent "
        "regularization that suppresses changes to weak lines. These are regularized fits, not calibrated uncertainty distributions.", "",
        "Most individual line corrections in this band are for **methane (CH4)**, followed by N2O and H2O. "
        "The line counts describe model freedom and do not assign residual power to a molecule. "
        "CO and CO2 are included in the physical opacity but have no free individual line strengths in this correction stage.", "",
        "### Validation and limits", "",
        "The four evaluation frames had their water amplitude, wavelength offset, IP width and original continuum "
        "calibrated on separate spatial blocks beforehand. Q5 is also fitted separately on each evaluation frame. "
        "This is conditional evaluation rather than a fully untouched exposure test. "
        "All displayed comparisons use the same retained data.", "",
        "**CO and CO2 scale factors reached their lower bounds in the physical initialization.** "
        "This band does not constrain those factors sufficiently to interpret them as abundance measurements. "
        "Strong-line residuals can combine line-list, instrumental, stellar and extraction errors. "
        "The larger wavelength coverage does not recover information lost where terrestrial transmission approaches zero.", "",
        "No planetary spectrum or time-common residual template is subtracted. This new band and fitting mask "
        "still need dedicated planet-injection checks; a smaller fitting RMS alone does not establish preservation "
        "of faint planetary lines. A shifted planetary template must undergo the same Q5 operation.", "",
        "### Reproducibility", "",
        f"Report date (UTC): {datetime.now(timezone.utc).date().isoformat()}. "
        "The active-analysis manifest is `IGRINS/results/current_order7_model.json`; "
        "the saved fit is `IGRINS/results/order7_stellar_fit.npz`. Product SHA256: "
        f"`{metadata['product_sha256']}`. "
        "Fitting entry points are `IGRINS/run_order7.py` and `IGRINS/fit_order7_stellar.py`; "
        "figures and this report are regenerated by `IGRINS/report_order7.py`. "
        "The report independently reconstructs all-pixel Q5 residuals and records checks in "
        "`results/order7_report_audit.json`. The scope/coordinate audit is `results/order7_scope_audit.json`. "
        "All fitting code, observational data and full arrays remain in Git-ignored `IGRINS/`."]
    if stellar_above_continuum:
        index = lines.index("### Reproducibility")
        lines[index:index] = [
            f"The empirical stellar-only correction reaches {stellar_max:.5f} of the continuum; "
            f"{stellar_above_continuum} of its nine-frame pixel samples exceed 1.001. "
            "These local overfills limit its interpretation as a physical stellar atmosphere. "
            "They remain included in the figures and evaluation.", ""]
    largest = independent["largest_intensity_correction"]
    index = lines.index("### Reproducibility")
    lines[index:index] = [
        f"The empirical strength fit also drives **{independent['near_zero_intensity_count']} telluric intensities effectively to zero**. "
        f"Its largest correction is {largest['line']['species']} at {largest['line']['nu']:.3f} cm^-1: "
        f"**{largest['multiplier']:.2f} times** the HITRAN strength, or {largest['assumed_prior_widths']:.2f} assumed prior widths. "
        f"Its reference spectral contrast is {100 * largest['reference_contrast']:.3f}%, so it is not a very weak line; "
        f"its HITRAN strength-error code is {largest['line']['strength_code']} "
        f"({largest['line']['uncertainty_label']}). This large fitted adjustment is an effective "
        "correction that may absorb other model errors, not a measurement of a HITRAN line-strength error.", ""]
    index = lines.index("### Remaining residuals and strong absorption")
    numerical_lines = ["### Numerical precision check", ""]
    if precision is not None:
        numerical_lines += [
            "The following control holds the stellar amplitudes, physical model, data and priors fixed "
            "and polishes a coarse near-optimum checkpoint with a tighter inner linear solve. "
            "The coarse run was interrupted and is not claimed to be converged. **Both rows precede "
            "the final null refinements**, so they are numerical controls rather than the final model rows above.", "",
            "| Numerical stage | LSMR atol/btol | LSMR maxiter | All-included evaluation RMS | Score-subset evaluation RMS |",
            "| --- | ---: | ---: | ---: | ---: |"]
        for case, label in [("coarse", "Coarse checkpoint; 70 evaluations, interrupted"),
                            ("tight", "Tight polish; 7 evaluations, converged")]:
            row = precision["residual_metrics"][case]
            settings = precision["solver_settings"][case]
            numerical_lines.append(f"| {label} | {settings['atol']:.0e} / {settings['btol']:.0e} | {settings['maxiter']} | {row['evaluation_all_rms_percent']:.9f}% | {row['evaluation_score_rms_percent']:.9f}% |")
        numerical_lines += ["",
            "Both use outer `ftol = xtol = 1e-9`, `gtol = 1e-7`. "
            f"The Q5-filtered predictions differ by **{precision['q5_evaluation_all_difference_rms_percent']:.6f}% RMS** "
            f"on all evaluation pixels; the maximum unfiltered change is {precision['max_absolute_component_difference_percent']:.6f}% "
            "of the continuum. Individual weak or blended line coefficients can move more than the combined spectrum. "
            "This check measures numerical solution sensitivity, not planet-signal retention.", ""]
    else:
        numerical_lines += [
            "The historical coarse/tight solver checkpoints are not present in this workspace. "
            "No matched numerical-precision comparison is claimed for this run.", ""]
    refinements = metadata.get("null_refinements", [])
    final_drift = refinements[-1]["component_max_drift"] if refinements else None
    final_solver = metadata["fits"]["telluric"][-1]
    numerical_lines += [
        f"The final telluric fit used **{final_solver.get('method', 'TRF')}** "
        "with the same regularized objective, bounds and free parameters. "
        "Objective and derivative checks are recorded in the fit audit.", ""]
    if final_drift is None:
        numerical_lines += ["No null refits were requested; repeated-fit numerical stability was not evaluated.", ""]
    elif not metadata["numerically_stabilized"]:
        numerical_lines += [
            f"The saved final fit includes {len(metadata['null_refinements'])} null refits. "
            f"Its last maximum model-flux change is {final_drift:.3e} in continuum units "
            f"({100 * final_drift:.9f}%). This is small relative to the measured residuals, "
            "but the strict criterion of two consecutive changes below 1e-8 was not met. "
            "All final optimizers terminated successfully; perfect numerical stabilization is not claimed.", ""]
    else:
        numerical_lines += [
            f"The final saved solution meets the repeated-null-refit criterion; its last maximum "
            f"model-flux change is {final_drift:.3e} in continuum units.", ""]
    lines[index:index] = numerical_lines
    enuma_overlay = None
    if (ROOT / "series_model.npz").exists():
        from overlay_enuma import main as make_enuma_overlay, report_lines
        enuma_overlay = make_enuma_overlay()
        start = lines.index("### Spectral fit") + 1
        stop = lines.index("### Matched configuration comparison")
        lines[start:stop] = report_lines(enuma_overlay)
    (OUT / "ORDER7_REPORT.txt").write_text("\n".join(lines) + "\n")
    projection_errors = {}
    for stage in STAGES:
        key = f"residual_{stage}"
        projection_errors[stage] = float(np.max(abs(residuals[stage] - a[key][0])))
        assert projection_errors[stage] < 1e-10
        for our_key, saved_key in [("evaluation_all_percent", "evaluation_included"),
                                   ("evaluation_score_percent", "evaluation_score")]:
            np.testing.assert_allclose(metrics[stage][our_key],
                metadata["metrics"][stage][saved_key]["rms_percent"], rtol=0, atol=1e-10)
    frame_values = residual[frame] * 100
    clipped_residuals = int(np.sum((frame_values < frame_residual_limits[0]) |
                                  (frame_values > frame_residual_limits[1])))
    all_values = residual[5:] * 100
    clipped_residuals += int(np.sum((all_values < evaluation_residual_limits[0]) |
                                   (all_values > evaluation_residual_limits[1])))
    assert clipped_residuals == 0
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), metrics=metrics,
                 transmission_bins=bins, stellar_depth_regions=stellar_regions,
                 all_stored_pixels_included=True, included_pixels=1848, score_pixels=773,
                 no_absorption_depth_mask=True, q5_independent_max_errors=projection_errors,
                 spectrum_frame=30, upper_axis_limits=list(upper_limits),
                 upper_panel_clipped_points=int(sum(np.sum((v < upper_limits[0]) | (v > upper_limits[1])) for v in top_values)),
                 residual_panels_clipped_points=clipped_residuals,
                 model_selected_residual_windows=zoom_records,
                 metadata_noise_proxy_percent=noise, free_parameters=free,
                 stellar_only_maximum=stellar_max,
                 stellar_only_samples_above_1p001=stellar_above_continuum,
                 numerically_stabilized=metadata["numerically_stabilized"],
                 intermediate_precision_control=precision,
                 source_sha256={str(p.relative_to(ROOT)): sha256(p) for p in [source, source.with_suffix('.json'), OUT/'order7_scope_audit.json', OUT/'order7_physical_e10.json', OUT/'order7_physical_species.json', precision_path, independent_path] if p.exists()},
                 script_sha256=sha256(Path(__file__)),
                 output_sha256={name: sha256(OUT/name) for name in ['ORDER7_REPORT.txt', 'order7_spectral_fit.png', 'order7_absorption_depth.png', 'order7_residuals.png', 'order7_comparison.png']})
    if enuma_overlay is not None:
        audit["enuma_comparison"] = enuma_overlay
        audit["source_sha256"].update(enuma_overlay["input_sha256"])
        audit["source_sha256"]["overlay_enuma.py"] = enuma_overlay["script_sha256"]
        figure = enuma_overlay["figures"][0]
        audit["upper_axis_limits"] = figure["upper_axis_limits"]
        audit["upper_panel_clipped_points"] = figure["upper_panel_clipped_points"]
        audit["residual_panels_clipped_points"] = figure["clipped_residual_points"]
        audit["output_sha256"]["order7_spectral_fit_enuma_4383.png"] = enuma_overlay["figures"][1]["sha256"]
    (OUT / "order7_report_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(dict(metrics=metrics, bins=bins, stellar_regions=stellar_regions)))


if __name__ == "__main__":
    main()
