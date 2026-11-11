"""Render the saved blase stellar experiment without rerunning the fit."""

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
COLORS = dict(adopted="tab:blue", stellar="tab:orange", refit="tab:green")
LABELS = dict(adopted="Adopted PHOENIX", stellar="Stellar amplitudes fitted",
              refit="Stellar amplitudes + telluric refit")
CASES = tuple(COLORS)
COMMIT = "6f6817b0c4095c104a10b2940375fc1e16cb80ca"


def corrected_model(data, case):
    """Add the profiled Q5 polynomial to the physical component for display."""
    component = data[f"component_{case}"]
    adjusted = np.empty_like(component)
    for b, (wave, mask) in enumerate(zip(data["wavelength_um"], data["included_mask"])):
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        delta = data["y_over_c"][b]-component[b]
        coefficients = delta[:, mask]@np.linalg.pinv(basis[mask]).T
        adjusted[b] = component[b]+coefficients@basis.T
    np.testing.assert_allclose(data["y_over_c"]-adjusted, data[f"residual_{case}"],
                               atol=1e-11, rtol=0)
    return adjusted


def shade_mask(ax, x, mask):
    """Shade exactly the excluded pixel bins, including isolated pixels."""
    order = np.argsort(x)
    centers, keep = x[order], mask[order]
    edges = np.r_[centers[0]-(centers[1]-centers[0])/2,
                  (centers[:-1]+centers[1:])/2,
                  centers[-1]+(centers[-1]-centers[-2])/2]
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~keep, False]))
    for left, right in boundaries.reshape(-1, 2):
        ax.axvspan(edges[left], edges[right], color="0.8", alpha=.3, lw=0, zorder=-2)


def draw_order(axes, data, adjusted, b, window=None):
    nu = 1e4/data["wavelength_um"][b]
    order = np.argsort(nu)
    x = nu[order]
    allowed, score = data["included_mask"][b], data["score_mask"][b]
    star_ax, spectrum_ax, residual_ax = axes
    for key, label, color in [("adopted", "PHOENIX stellar component", COLORS["adopted"]),
                               ("corrected", "Corrected stellar component", COLORS["stellar"])]:
        values = np.mean(data[f"stellar_only_{key}"][b, 5:], axis=0)
        star_ax.plot(x, values[order], color=color, lw=1.15, label=label)
    star_ax.axhline(1., color="0.55", lw=.6)
    calibration = data["calibration_mask"][b][order]
    star_ax.plot(x[calibration], np.full(calibration.sum(), .025), "|", color="teal", ms=4,
                 alpha=.65, transform=star_ax.get_xaxis_transform(), label="Stellar calibration pixels")
    observed = np.mean(data["y_over_c"][b, 5:], axis=0)
    spectrum_ax.plot(x, observed[order], color="0.15", lw=.8, label="Observed / original continuum")
    for case in CASES:
        mean_model = np.mean(adjusted[case][b, 5:], axis=0)
        mean_residual = 100*np.mean(data[f"residual_{case}"][b, 5:], axis=0)
        spectrum_ax.plot(x, mean_model[order], color=COLORS[case], lw=.95, label=LABELS[case])
        # Faint full-window curves preserve diagnostics inside the original exclusions.
        residual_ax.plot(x, mean_residual[order], color=COLORS[case], lw=.65, alpha=.25)
        residual_ax.plot(x, np.where(allowed, mean_residual, np.nan)[order],
                         color=COLORS[case], lw=1.05, label=LABELS[case])
    residual_ax.axhline(0, color="0.35", lw=.65)
    selected_x = x[score[order]]
    residual_ax.plot(selected_x, np.full(len(selected_x), .035), "|", color="black", ms=4,
                     transform=residual_ax.get_xaxis_transform(), label="Original score pixels")
    for ax in axes:
        shade_mask(ax, nu, allowed)
        ax.grid(alpha=.13)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.set_xlim(window if window is not None else (x[0], x[-1]))
    if window is not None:
        selection = (nu >= window[0]) & (nu <= window[1])
        for ax, arrays in [(star_ax, [data[f"stellar_only_{key}"][b, 5:] for key in ["adopted", "corrected"]]),
                           (spectrum_ax, [data["y_over_c"][b, 5:]]+[adjusted[key][b, 5:] for key in CASES])]:
            values = np.concatenate([np.mean(value, axis=0)[selection] for value in arrays])
            margin = max(.003, .12*np.ptp(values))
            ax.set_ylim(values.min()-margin, values.max()+margin)
        values = np.concatenate([100*np.mean(data[f"residual_{case}"][b, 5:], axis=0)[selection]
                                 for case in CASES])
        margin = max(.15, .12*np.ptp(values))
        residual_ax.set_ylim(values.min()-margin, values.max()+margin)
    star_ax.set_ylabel("Stellar-only flux")
    spectrum_ax.set_ylabel("Continuum-scaled flux")
    residual_ax.set_ylabel("Mean residual (%)")
    residual_ax.set_xlabel(r"Observed wavenumber (cm$^{-1}$)")


def plots(data, result, adjusted):
    fig, axes = plt.subplots(3, 3, figsize=(18, 10), sharex="col")
    for b, order in enumerate(data["orders"]):
        draw_order(axes[:, b], data, adjusted, b)
        old = result["metrics"]["adopted"]["evaluation_score"]["per_order"][str(order)]
        new = result["metrics"]["refit"]["evaluation_score"]["per_order"][str(order)]
        axes[0, b].set_title(f"Order {order}; score RMS {old:.3f}% to {new:.3f}%")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    axes[1, 0].legend(loc="lower left", fontsize=7)
    axes[2, 0].legend(handles=[Patch(facecolor=".8", alpha=.3, label="Original exclusions"),
                              Line2D([], [], color="k", marker="|", ls="", label="Original score pixels")],
                      loc="lower left", fontsize=7)
    fig.suptitle("Shared stellar amplitudes with actual blase JAX profiles\n"
                 "Curves: mean of four evaluation frames. RMS: all individual evaluation samples; unchanged masks and Q5.",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT/"blase_stellar_fit.png", dpi=170)
    plt.close(fig)

    b = list(data["orders"]).index(13)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    draw_order(axes, data, adjusted, b, (4819., 4820.7))
    for ax in axes:
        ax.axvline(4819.920802, color=".4", ls=":", lw=.7)
    axes[0].legend(loc="lower left", fontsize=9)
    axes[1].legend(loc="lower left", fontsize=9)
    axes[2].legend(handles=[Patch(facecolor=".8", alpha=.3, label="Original exclusions"),
                           Line2D([], [], color="k", marker="|", ls="", label="Original score pixels")],
                   fontsize=9)
    old = result["metrics"]["adopted"]["diagnostic"]["stellar_4819_92"]["rms_percent"]
    new = result["metrics"]["refit"]["diagnostic"]["stellar_4819_92"]["rms_percent"]
    fig.suptitle(f"4819.92 cm$^{{-1}}$ residual: local score RMS {old:.3f}% to {new:.3f}%\n"
                 "Local RMS uses 4819.74-4820.10 cm$^{-1}$; curves show the four-frame mean.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT/"blase_stellar_4820.png", dpi=180)
    plt.close(fig)

    b, _, pixel = np.unravel_index(data["stellar_only_corrected"].argmax(),
                                   data["stellar_only_corrected"].shape)
    center = 1e4/data["wavelength_um"][b, pixel]
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    draw_order(axes, data, adjusted, b, (center-.45, center+.45))
    axes[0].legend(loc="lower left", fontsize=8)
    axes[1].legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Stellar overshoot diagnostic: order {data['orders'][b]}, {center:.4f} cm$^{{-1}}$\n"
                 "Largest corrected stellar flux; gray pixels remain excluded from the primary score.", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .93))
    fig.savefig(OUT/"blase_stellar_overshoot.png", dpi=170)
    plt.close(fig)


def report(data, result):
    manifest = json.loads((OUT/"adopted_telluric_model.json").read_text())
    selected = manifest.get("case") == "blase_stellar_refit_stabilized"
    status = ("Status: provisionally adopted by user instruction; stabilized uninjected baseline selected."
              if selected else "Status: saved experiment; see CURRENT_MODEL.txt for the current selection.")
    metrics = result["metrics"]
    old = metrics["adopted"]["evaluation_score"]["rms_percent"]
    star = metrics["stellar"]["evaluation_score"]["rms_percent"]
    new = metrics["refit"]["evaluation_score"]["rms_percent"]
    local_old = metrics["adopted"]["diagnostic"]["stellar_4819_92"]["rms_percent"]
    local_new = metrics["refit"]["diagnostic"]["stellar_4819_92"]["rms_percent"]
    nfree = sum(row["free_lines"] for row in result["mask_counts"])
    lines = ["Shared stellar-amplitude experiment with blase", "",
             f"Generated UTC: {datetime.now(timezone.utc).isoformat()}",
             f"Fit UTC: {result['created_utc']}", status, "",
             f"On the unchanged evaluation score pixels, RMS is {old:.6f}% with the adopted PHOENIX model,",
             f"{star:.6f}% after fitting stellar amplitudes alone, and {new:.6f}% after refitting the existing telluric strengths.",
             f"The final change is {new-old:+.6f} percentage points ({100*(new/old-1):+.2f}% relative RMS).",
             f"Around 4819.92 cm^-1, the original local score RMS changes from {local_old:.6f}% to {local_new:.6f}%.", "",
             "What was changed", "",
             "The adopted PHOENIX spectrum is kept as the reference. A small set of absorption features is represented",
             "with the actual upstream blase JAX pseudo-Voigt implementation. These are mathematical absorption",
             "features; a feature may blend several transitions and is not an identified atomic line or oscillator strength.",
             "The clone amplitudes and widths are first fitted to noiseless PHOENIX. Observations then adjust only",
             "selected amplitudes, shared by every exposure in the stellar rest frame. Centers and widths remain fixed.", "",
             "This amplitude-only experiment cannot add a missing stellar feature or move a misplaced line center.",
             "Those possibilities remain separate from the correction tested here.", "",
             "The native stellar spectrum is PHOENIX * clone(q) / clone(q=1). At q=1 it exactly recovers PHOENIX,",
             "so a better numerical clone cannot by itself masquerade as a better data fit. Rotation, extra broadening",
             "and stellar RV are applied next. The stellar spectrum is multiplied by the telluric transmission,",
             "and the original instrumental profile is applied to that product.",
             f"Maximum adopted-component reconstruction error: {result['reconstruction_error']:.3e}.", "",
             "Training and regularization", "",
             "Only exposures 0, 19, 39, 59 and 78 train the new amplitudes. Exposures 10, 30, 50 and 70 evaluate them.",
             "Stellar calibration uses pixels whose adopted telluric-only transmission exceeds 0.95 in all five training",
             "frames. Strong stellar lines can enter this calibration even when the original telluric-fit mask excluded them.",
             "A line is free only if its model response is measurable in these pixels. Its maximum response to a unit",
             "amplitude change must exceed 0.001, and the calibration/full-window response L2 ratio must exceed 0.25.",
             f"This selects {nfree} shared amplitudes, without selecting lines from observed residuals.",
             "Amplitude ratio q has a prior centered at 1 with width 0.5*(0.1+0.9*d^2/(d^2+0.005^2)), where d is",
             "the frozen model response on calibration pixels. Weak features therefore have tighter regularization.",
             "The bounds are 0 <= q <= min(3, the positive-profile limit). This prior is empirical shrinkage, not",
             "a HITRAN error bar, a stellar abundance measurement, or a calibrated posterior uncertainty.", "",
             "Order | Calibration pixels | Strong stellar calibration | Newly used pixels | Original included | Score overlap | Free / candidate stellar features"]
    for row in result["mask_counts"]:
        lines.append(f"{row['order']} | {row['calibration_pixels']} | "
                     f"{row.get('strong_stellar_calibration_pixels', 'not saved')} | {row.get('newly_used_pixels', 'not saved')} | "
                     f"{row['original_included_pixels']} | "
                     f"{row['original_score_overlap']} | {row['free_lines']} / {row['candidate_lines']}")
    lines += ["", "Noiseless PHOENIX clone diagnostics", "",
              "Order | Native clone RMS (%) | Maximum absolute mismatch (%) | Converged | Evaluations"]
    for row in result["clone_metadata"]:
        fit = row.get("optimization", {})
        lines.append(f"{row.get('order')} | {row.get('clone_rms_percent', float('nan')):.6f} | "
                     f"{row.get('clone_max_absolute_percent', float('nan')):.6f} | "
                     f"{fit.get('success', 'not saved')} | {fit.get('nfev', 'not saved')}")
    lines += ["", "These clone mismatches are retained in the anchored PHOENIX reference; they are not directly added",
              "to the observed model. They limit how faithfully a fitted feature response represents a physical line.",
              "Numerical convergence of the clone does not establish that its line decomposition is unique.", "",
              "Two separate fitting stages", "",
              "Stage 1 holds the adopted 269 telluric line-strength corrections fixed and learns only stellar amplitudes.",
              "Its degree-five polynomial is profiled on the stellar-calibration pixels.",
              "Stage 2 freezes those stellar amplitudes and refits the same 269 telluric corrections on the original",
              "included pixels, retaining the adopted weak-line prior widths. The extra unreported-intensity lines from",
              "the preceding experiment are deliberately absent here, so this comparison isolates the stellar change.",
              "The atmospheric profile, columns, wavelength solution, old two-Gaussian IP, rotation and original continua stay fixed.", "",
              "All reported residuals use the original included-mask degree-five projection and the original score mask.",
              "There is no new residual-based exclusion. This is conditional evaluation: the data and masks have been",
              "explored previously, and a polynomial is fitted separately to each evaluation frame.", "",
              "Residual RMS (% of original continuum units)", "",
              "Case | Training included | Evaluation score | Evaluation included | Strong stellar diagnostic | Full window | Local 4819.92"]
    for name in CASES:
        m = metrics[name]
        values = [m["training"]["rms_percent"], m["evaluation_score"]["rms_percent"],
                  m["evaluation_included"]["rms_percent"], m["diagnostic"]["strong_stellar"]["rms_percent"],
                  m["diagnostic"]["full_window"]["rms_percent"], m["diagnostic"]["stellar_4819_92"]["rms_percent"]]
        lines.append(name+" | "+" | ".join(f"{value:.6f}" for value in values))
    lines += ["", "Case | Order 13 evaluation score | Order 34 | Order 40"]
    for name in CASES:
        lines.append(name+" | "+" | ".join(f"{metrics[name]['evaluation_score']['per_order'][str(order)]:.6f}"
                                             for order in data["orders"]))
    lines += ["", "The strong-stellar diagnostic uses >2% absorption in the original PHOENIX stellar-only model.",
              "The full-window diagnostic includes original excluded pixels and is not the primary score.",
              "The local 4819.92 diagnostic uses original score pixels in 4819.74-4820.10 cm^-1.",
              "RMS values include every individual evaluation sample, not only the mean curve displayed in the figures.", ""]
    b = list(data["orders"]).index(13)
    pixel = int(np.argmin(np.abs(1e4/data["wavelength_um"][b]-4819.920802)))
    lines += ["At the pixel nearest 4819.920802 cm^-1, the four-frame mean residual (%) is: "+
              ", ".join(f"{case} {100*np.mean(data[f'residual_{case}'][b, 5:, pixel]):+.6f}" for case in CASES)+".",
              "The corresponding stellar-only absorption depth (%) is: "+
              ", ".join(f"{case} {100*(1-np.mean(data[f'stellar_only_{case}'][b, 5:, pixel])):.6f}"
                        for case in ["adopted", "corrected"])+".",
              "This pixel belongs to the telluric-clear stellar-calibration mask. The data prefer an almost absent",
              "stellar feature here. This supports a PHOENIX feature that is too strong, but does not identify an",
              "individual faulty atomic transition; the fitted clone features are blends in the stellar rest frame.", "",
              "Stellar-flux overshoot diagnostic (all nine exposures, after the response)", "",
              "Order | PHOENIX maximum flux | Corrected maximum flux | Max excess over local PHOENIX envelope (%) | Samples exceeding envelope by >0.1%"]
    overshoot = []
    largest_flux = None
    for b, order in enumerate(data["orders"]):
        reference = data["stellar_only_adopted"][b]
        corrected = data["stellar_only_corrected"][b]
        padded = np.pad(reference, ((0, 0), (15, 15)), mode="edge")
        envelope = np.lib.stride_tricks.sliding_window_view(padded, 31, axis=-1).max(axis=-1)
        excess = corrected-envelope
        frame, pixel = np.unravel_index(corrected.argmax(), corrected.shape)
        if largest_flux is None or corrected[frame, pixel] > largest_flux[0]:
            largest_flux = (float(corrected[frame, pixel]), float(envelope[frame, pixel]),
                            float(1e4/data["wavelength_um"][b, pixel]),
                            bool(data["included_mask"][b, pixel]), bool(data["score_mask"][b, pixel]))
        lines.append(f"{order} | {reference.max():.6f} | {corrected.max():.6f} | "
                     f"{100*excess.max():+.6f} | {int(np.sum(excess > .001))}")
        for name, mask in [("score", data["score_mask"][b]), ("included", data["included_mask"][b])]:
            values = excess[5:, mask]
            overshoot.append((int(order), name, int(np.sum(values > .001)), values.size, float(100*values.max())))
    lines += ["", "Order | Evaluation mask | Samples above envelope by >0.1% | Total samples | Max envelope excess (%)"]
    for order, name, count, size, maximum in overshoot:
        lines.append(f"{order} | {name} | {count} | {size} | {maximum:+.6f}")
    lines += ["", "The local envelope is a 31-pixel running maximum of the original stellar component. This is a diagnostic,",
              "not a fitted constraint or a new mask. Flux above one alone is not evidence for emission because the",
              "normalized PHOENIX reference itself can exceed one. A narrow excess above its surrounding envelope",
              "can indicate an imperfect clone response when an amplitude approaches zero; it requires visual inspection.",
              f"The largest corrected stellar flux is {largest_flux[0]:.6f} near observed {largest_flux[2]:.6f} cm^-1,",
              f"with local PHOENIX envelope {largest_flux[1]:.6f}. Original included={largest_flux[3]}, score={largest_flux[4]}.",
              "The local 4819.92 improvement does not show a comparable above-unity peak. Overfilling elsewhere",
              "is a limitation of interpreting an anchored phenomenological correction as a physical stellar spectrum.", "",
              "Largest amplitude changes (selected features; stellar rest frame)", "",
              "Order | Rest center (cm^-1) | Amplitude ratio | Assumed prior width"]
    largest = sorted((row for row in result["line_records"] if row["fitted"]),
                     key=lambda row: abs(row["amplitude_ratio"]-1), reverse=True)[:20]
    for row in largest:
        lines.append(f"{row['order']} | {row['center_rest_cm1']:.6f} | {row['amplitude_ratio']:.6f} | {row['prior_width']:.6f}")
    lines += ["", "Planet-line preservation", ""]
    planet_path = OUT/"blase_planet.json"
    if planet_path.exists():
        planet = json.loads(planet_path.read_text())
        lines += ["Paired artificial-line tests rerun the complete stellar-calibration and telluric-refit pipeline after injection.",
                  "The control keeps the original PHOENIX star and refits its telluric strengths. Both uninjected baselines",
                  "are also refitted. Gains below are measured on the four evaluation frames relative to Q5-filtered injected",
                  "signals. They are response coefficients, not a fraction of all available planetary information.",
                  "A gain above 100% can result from the fitted response projected onto the template; it does not mean",
                  "that the data gained physical signal or that total planet throughput exceeds unity.", "",
                  "Signal | FWHM km/s | Kp km/s | Depth | PHOENIX gain (%) | New gain (%) | Change (percentage points)"]
        for row in planet["cases"]:
            a, b = [row["models"][name]["evaluation_score"]["incremental_gain"] for name in ["phoenix", "stellar"]]
            lines.append(f"{row['kind']} | {row['fwhm_kms']:g} | {row['kp_kms']:g} | {row['depth']:g} | "
                         f"{100*a:.6f} | {100*b:.6f} | {100*(b-a):+.6f}")
        moving = [row for row in planet["cases"] if row["kind"] == "forest" and row["kp_kms"] > 0]
        stationary = next(row for row in planet["cases"] if row["kp_kms"] == 0)
        local = next(row for row in planet["cases"] if row["kind"] == "local_4819_92")
        changes = [100*row["paired_gain_difference"]["evaluation_score"] for row in moving]
        lines += ["", f"Across the moving-forest controls, the extra stellar flexibility changes pooled response by {min(changes):+.6f}",
                  f"to {max(changes):+.6f} percentage points. The 0.01%-depth case reproduces the 0.1%-depth response change.",
                  f"The stationary forest changes by {100*stationary['paired_gain_difference']['evaluation_score']:+.6f} percentage points,",
                  f"while the local moving line crossing 4819.92 changes by {100*local['paired_gain_difference']['evaluation_score']:+.6f} points.",
                  "The stationary loss matters: a shared stellar correction can absorb features that barely move."]
        null_checks = planet["baseline"]["stellar"]["null_refit_checks"]
        with np.load(OUT/"blase_planet.npz") as injected:
            baseline_component = injected["baseline_prediction_stellar"]
            residual = np.empty_like(baseline_component)
            for b, (wave, mask) in enumerate(zip(data["wavelength_um"], data["included_mask"])):
                x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
                basis = np.polynomial.chebyshev.chebvander(x, 5)
                delta = data["y_over_c"][b]-baseline_component[b]
                residual[b] = delta-(delta[:, mask]@np.linalg.pinv(basis[mask]).T)@basis.T
            keep = np.broadcast_to(data["score_mask"][:, None], residual[:, 5:].shape)
            refined_rms = 100*np.sqrt(np.mean(residual[:, 5:][keep]**2))
        lines += ["", "The main table and figures retain the original saved fit. Injection tests use a numerically stabilized",
                  f"uninjected baseline after {len(null_checks)} identical warm refinements, because the clipped profiles can produce stopping drift.",
                  f"Main evaluation RMS: {new:.9f}%; stabilized injection-baseline RMS: {refined_rms:.9f}%",
                  f"(difference {refined_rms-new:+.9f} percentage points). These are different saved parameter states.",
                  "The last two null-refit maximum flux changes are "+
                  ", ".join(f"{row['component_max_drift']:.3e}" for row in null_checks[-2:])+".",
                  "Every injected response is compared with its own stabilized uninjected baseline."]
        lines += ["", "These tests use artificial Gaussian features at observed resolution, not a physical planet spectrum before",
                  "the IP. Stationary features are intrinsically degenerate with a shared stellar/telluric correction.",
                  "A small set of moving-line tests cannot establish preservation for every molecule, orbital velocity or epoch.",
                  "See BLASE_PLANET_REPORT.txt and blase_independent_audit.json for the paired controls and audit."]
    else:
        lines += ["The paired injection check has not produced its result file yet. No planet-preservation claim is made here.",
                  "Regenerate this report after check_blase_planet.py finishes."]
    lines += ["", "Reading the figures", "",
              "blase_stellar_fit.png covers all three windows; blase_stellar_4820.png shows the 4819-4820.7 cm^-1 interval.",
              "blase_stellar_overshoot.png displays the region of the largest corrected stellar flux as a limitation diagnostic.",
              "The upper panel shows only the stellar component after the same broadening, RV and IP as the data.",
              "The middle panel adds the profiled degree-five polynomial to each physical model so it can be compared",
              "directly with the observed/original-continuum flux. The lower panel is exactly the resulting Q5 residual.",
              "Gray shading marks the original exclusions. Faint residual curves retain diagnostics there; opaque curves",
              "show the included pixels. Black ticks identify the original scoring subset. Teal ticks in the top panel",
              "identify stellar-calibration pixels. Curves are means of four evaluation frames.", "",
              "Interpretation limits", ""]+result["limitations"]
    lines += ["", "Source and reproducibility", "",
              f"Actual blase JAX source: https://github.com/gully/blase/blob/{COMMIT}/src/blase/jaxemulator.py",
              "Method paper: https://arxiv.org/abs/2210.01827",
              "The unmodified upstream source and MIT license are in IGRINS/vendor/blase. The thin local adapter fixes",
              "sparse support indexing on the instance and anchors the clone to the original PHOENIX spectrum.",
              "Fit (command used on this Linux host):",
              "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 JAX_ENABLE_X64=true taskset -c 0-15 python IGRINS/fit_blase_stellar.py",
              "Injection/audit:",
              "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 JAX_ENABLE_X64=true taskset -c 0-15 python IGRINS/check_blase_planet.py --max-nfev 160",
              "Report only: OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python IGRINS/report_blase_stellar.py",
              "taskset controls Linux CPU affinity; omit or adapt it on other machines.",
              "Fitting code and arrays remain inside the Git-ignored IGRINS directory; derived figures and the summary were published to Issue832.",
              "Current selection is recorded in CURRENT_MODEL.txt and adopted_telluric_model.json.", "",
              "Saved clone metadata:", json.dumps(result["clone_metadata"], indent=2), "",
              "Input product SHA256:"]
    products = ["blase_stellar_fit.json", "blase_stellar_fit.npz", "adopted_telluric_fit.npz"]
    if planet_path.exists():
        products += ["blase_planet.json", "blase_planet.npz"]
    if selected:
        lines += ["Current adopted product: "+manifest["product"],
                  f"Current adopted evaluation score RMS: {manifest['metrics']['evaluation_score']['rms_percent']:.9f}%",
                  "Published update: https://github.com/HajimeKawahara/exojax/issues/832#issuecomment-6793133393"]
        products += ["adopted_telluric_model.json", Path(manifest["product"]).name]
    for filename in products:
        lines.append(filename+": "+hashlib.sha256((OUT/filename).read_bytes()).hexdigest())
    (OUT/"BLASE_STELLAR_REPORT.txt").write_text("\n".join(lines)+"\n")


def main():
    result = json.loads((OUT/"blase_stellar_fit.json").read_text())
    with np.load(OUT/"blase_stellar_fit.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    adjusted = {case: corrected_model(data, case) for case in CASES}
    plots(data, result, adjusted)
    report(data, result)
    print("Wrote BLASE_STELLAR_REPORT.txt and three blase_stellar PNG figures")


if __name__ == "__main__":
    main()
