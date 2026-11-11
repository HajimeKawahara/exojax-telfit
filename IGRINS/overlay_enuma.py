"""Overlay supplied enuma spectra with the current Order 7 ExoJAX fit."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

from report_order7 import BLUE, GREEN, ORANGE, RED, clean, limits, q5, rms, shade, sha256

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PURPLE, TEAL = "#8e44ad", "#008b8b"


def compare():
    """Match stored pixels and frames, then compare on the original continuum."""
    paths = [ROOT / "series_model.npz", OUT / "order7_stellar_fit.npz",
             ROOT / "data/cube14v3.pic"]
    hashes = {str(p.relative_to(ROOT)): sha256(p) for p in paths}
    with np.load(paths[0]) as saved:
        supplied = {key: saved[key] for key in saved.files}
    with np.load(paths[1]) as saved:
        current = {key: saved[key] for key in saved.files}
    with paths[2].open("rb") as stream:
        cube_wave, cube = pickle.load(stream)
    wave = current["wavelength_um"][0]
    exposures = current["exposures"]
    assert current["orders"].tolist() == [5]
    assert current["included_mask"].all()
    np.testing.assert_allclose(wave, cube_wave[5, ::-1], rtol=0, atol=1e-12)
    order = int(np.argmin(np.max(abs(supplied["wave"]/1000-wave[::-1]), axis=1)))
    assert order == 38
    rounded_nm = (cube_wave[5]*1000).astype(np.float32).astype(float)
    quantization_error = float(np.max(abs(supplied["wave"][order]-rounded_nm)))
    assert quantization_error < 1e-7
    indices = []
    for frame in exposures:
        matches = np.flatnonzero(supplied["frame"] == frame)
        assert len(matches) == 1
        indices.append(int(matches[0]))
    model_supplied = supplied["flux_model"][indices, order, ::-1]
    transmission = supplied["flux_telluric"][indices, order, ::-1]
    observed, continuum = current["y_over_c"][0], current["continuum"][0]
    raw_data = cube[5, exposures, ::-1]
    source_scale = np.median(raw_data/(observed*continuum), axis=1)
    np.testing.assert_allclose(raw_data/source_scale[:, None], observed*continuum,
                               rtol=1e-13, atol=1e-13)
    p95 = np.percentile(raw_data, 95, axis=1)
    scale = p95/source_scale
    enuma = model_supplied*scale[:, None]/continuum
    raw_residual = observed-enuma
    additive_residual, additive_smooth = q5(raw_residual, wave)
    x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    coefficients = np.array([np.linalg.lstsq(model[:, None]*basis, row, rcond=None)[0]
                             for model, row in zip(enuma, observed)])
    blaze = coefficients @ basis.T
    assert np.isfinite(blaze).all() and (blaze > 0).all()
    corrected = enuma*blaze
    corrected_residual = observed-corrected
    exojax_residual, exojax_smooth = q5(observed-current["component"][0], wave)
    np.testing.assert_allclose(exojax_residual, current["residual_filtered"][0],
                               rtol=0, atol=1e-10)
    nu = 1e4/wave
    local = (nu >= 4382.4) & (nu <= 4383.5)
    score = current["score_mask"][0]
    metrics = {}
    for name, residual in [("exojax_current_q5", exojax_residual),
                           ("enuma_supplied", raw_residual),
                           ("enuma_multiplicative_degree5", corrected_residual),
                           ("enuma_additive_q5", additive_residual)]:
        metrics[name] = dict(four_frame_all_rms_percent=rms(residual[5:]),
                             four_frame_score_rms_percent=rms(residual[5:, score]),
                             four_frame_local_4383_rms_percent=rms(residual[5:, local]),
                             per_frame_all_rms_percent=[rms(row) for row in residual],
                             frame30_all_rms_percent=rms(residual[6]))
    # A second plausible normalization tests the small evaluation-scale ambiguity.
    alternative = model_supplied/continuum
    alternative_raw = observed-alternative
    alternative_q5, _ = q5(alternative_raw, wave)
    audit = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
                 display_order=7, cube_index=5, enuma_index=order,
                 exposures=exposures.tolist(), matched_comparison_frames=exposures[5:].tolist(),
                 frame30_index=int(np.flatnonzero(exposures == 30)[0]),
                 input_sha256=hashes, wavelength_unit="nm",
                 wavelength_mapping="Same stored pixels; reverse pixel axis, no interpolation or velocity fit",
                 float32_wavelength_match_max_nm=quantization_error,
                 normalization_status="Unknown to user; full-order per-frame 95th percentile inferred from flux agreement",
                 conversion="flux_model * p95(raw observed order) / saved ExoJAX scalar / saved ExoJAX continuum",
                 supplied_full_model_rescaled_only=True,
                 telluric_transmission_rescaled=False,
                 source_scalar=source_scale.tolist(), full_order_p95=p95.tolist(),
                 model_scalar_conversion=scale.tolist(),
                 enuma_blaze_adjustment="Six multiplicative degree-five Chebyshev coefficients per frame, fitted on all 1848 pixels in the original continuum units",
                 blaze_minimum=float(blaze.min()), blaze_maximum=float(blaze.max()),
                 evaluation_blaze_minimum=float(blaze[5:].min()),
                 evaluation_blaze_maximum=float(blaze[5:].max()),
                 residual_sse_fraction_removed_by_blaze=float(
                     1-np.sum(corrected_residual[5:]**2)/np.sum(raw_residual[5:]**2)),
                 metrics=metrics,
                 alternative_saved_scalar=dict(raw_rms_percent=rms(alternative_raw[5:]),
                                                additive_q5_rms_percent=rms(alternative_q5[5:])),
                 limitations=["Supplied NPZ does not document flux normalization or enuma training frames.",
                              "These are matched frames, not a claim of enuma held-out validation.",
                              "The optional blaze curve uses the plotted data, including evaluation frames; it is a conditional display diagnostic.",
                              "No line parameters, wavelength shifts, instrumental profile, or stellar model are refitted.",
                              "RMS units are percent of the same saved ExoJAX continuum, not percent of local absorbed flux."])
    arrays = dict(wavelength_um=wave, wavenumber_cm1=nu, exposures=exposures,
                  observed=observed, score_mask=score,
                  strong_telluric=np.min(current["telluric_only_baseline"][0, :5], axis=0) <= .5,
                  exojax_model=current["component"][0]+exojax_smooth,
                  exojax_stellar=current["stellar_only"][0],
                  exojax_telluric=current["telluric_only"][0],
                  exojax_residual=exojax_residual,
                  enuma_model_supplied=enuma, enuma_model_blaze_corrected=corrected,
                  enuma_telluric=transmission, enuma_residual_supplied=raw_residual,
                  enuma_residual_blaze_corrected=corrected_residual,
                  enuma_residual_additive_q5=additive_residual,
                  enuma_additive_smooth=additive_smooth,
                  enuma_blaze_curve=blaze, enuma_blaze_coefficients=coefficients)
    for values in arrays.values():
        assert np.isfinite(values).all()
    np.savez_compressed(OUT / "order7_enuma_comparison.npz", **arrays)
    for path in paths:
        assert sha256(path) == hashes[str(path.relative_to(ROOT))]
    return arrays, audit


def draw(arrays, audit, filename, window=None):
    """Overlay the spectrum and put each model's residuals in its own column."""
    a = arrays
    nu = a["wavenumber_cm1"]
    frame = audit["frame30_index"]
    visible = np.ones(len(nu), bool) if window is None else (nu >= window[0]) & (nu <= window[1])
    score = a["score_mask"] & visible
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, left=.065, right=.985, top=.795, bottom=.13,
                         height_ratios=[2.35, 1.2, 1.3], hspace=.38, wspace=.14)
    top = fig.add_subplot(gs[0, :])
    axes = np.array([[fig.add_subplot(gs[row, col]) for col in range(2)] for row in [1, 2]])
    fig.suptitle("WASP-77Ab / IGRINS Order 7: ExoJAX and enuma", y=.985, fontsize=17, weight="bold")
    fig.text(.5, .953, "Frame 30 spectrum | matched frames 10, 30, 50, 70 | all stored pixels retained",
             ha="center", fontsize=11)
    curves = [(a["observed"][frame], "black", "-", .7, "Observed"),
              (a["exojax_model"][frame], RED, "-", 1.05, "ExoJAX full model + Q5"),
              (a["enuma_model_supplied"][frame], PURPLE, "-", .95, "enuma full model: supplied"),
              (a["enuma_model_blaze_corrected"][frame], PURPLE, "--", .95, "enuma: degree-5 blaze adjusted"),
              (a["exojax_stellar"][frame], GREEN, "--", .9, "ExoJAX stellar model alone"),
              (a["exojax_telluric"][frame], ORANGE, ":", .9, "ExoJAX telluric alone"),
              (a["enuma_telluric"][frame], TEAL, ":", .9, "enuma telluric alone")]
    handles = []
    for values, color, style, width, label in curves:
        top.plot(nu, values, color=color, ls=style, lw=width)
        handles.append(Line2D([], [], color=color, ls=style, lw=1.2, label=label))
    handles += [Line2D([], [], color=BLUE, marker=".", ls="", label="Blue: corrected residual, score subset"),
                Patch(facecolor="#ffe5aa", label="Strong telluric absorption: included")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .929),
               ncol=3, frameon=False, fontsize=9.5)
    top_values = [curve[0][visible] for curve in curves]
    top_limits = limits(top_values)
    top.set(ylim=top_limits, ylabel="Flux / original continuum")
    top.set_title("Spectrum overlay" if window is None else "4383 cm$^{-1}$ close-up: same models and corrections", loc="left", fontsize=11)
    residuals = [a["exojax_residual"], a["enuma_residual_blaze_corrected"]]
    raw = a["enuma_residual_supplied"]
    frame_limits = limits([residuals[0][frame, visible]*100, raw[frame, visible]*100,
                           residuals[1][frame, visible]*100])
    all_limits = limits([residuals[0][5:, visible]*100, residuals[1][5:, visible]*100,
                         raw[5:, visible].mean(axis=0)*100])
    colors = ["#1b9e77", "#7570b3", "#d95f02", "#666666"]
    model_names = ["ExoJAX + Q5", "enuma: supplied / blaze adjusted"]
    clipping = 0
    for col, residual in enumerate(residuals):
        ax = axes[0, col]
        if col == 1:
            ax.plot(nu, raw[frame]*100, color=".65", lw=.8, label="Supplied")
            ax.plot(nu, residual[frame]*100, color=PURPLE, lw=.65, label="Blaze adjusted")
            ax.legend(loc="lower left", fontsize=8, ncol=2)
        else:
            ax.plot(nu, residual[frame]*100, color=".65", lw=.7)
        ax.scatter(nu[score], residual[frame, score]*100, color=BLUE, s=5, zorder=3)
        value = rms(residual[frame, visible])
        label = f"{value:.3f}%" if col == 0 else f"{rms(raw[frame, visible]):.3f}% / {value:.3f}%"
        ax.set_title(f"{model_names[col]} | frame 30 RMS {label}", loc="left", fontsize=10)
        ax.set(ylim=frame_limits, ylabel="Frame 30 residual (%)" if col == 0 else "")
        ax = axes[1, col]
        for exposure, row, color in zip(a["exposures"][5:], residual[5:], colors):
            ax.plot(nu, row*100, lw=.6, color=color, alpha=.75, label=f"Frame {exposure}")
        ax.plot(nu, residual[5:].mean(axis=0)*100, color="black", lw=1.1, label="Mean")
        if col == 1:
            ax.plot(nu, raw[5:].mean(axis=0)*100, color="black", ls=":", lw=.9, label="Supplied mean")
        value = rms(residual[5:, visible])
        label = f"{value:.3f}%" if col == 0 else f"{rms(raw[5:, visible]):.3f}% / {value:.3f}%"
        ax.set_title(f"Four-frame RMS {label}", loc="left", fontsize=10)
        ax.set(ylim=all_limits, ylabel="Matched-frame residual (%)" if col == 0 else "")
        ax.legend(fontsize=7, ncol=3, loc="lower left", framealpha=.85)
        for values, bounds in [(residual[frame, visible]*100, frame_limits),
                               (residual[5:, visible]*100, all_limits)]:
            clipping += int(np.sum((values < bounds[0]) | (values > bounds[1])))
    for ax in [top, *axes.ravel()]:
        shade(ax, nu, a["strong_telluric"])
        ax.set_xlim(nu[visible].min(), nu[visible].max())
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        clean(ax)
    for ax in axes.ravel():
        ax.axhline(0, color="black", lw=.5)
    for ax in axes[1]:
        ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.text(.5, .038, "enuma flux normalization is undocumented: per-frame, full-order 95th-percentile scaling is inferred.\n"
             "Blaze adjustment fits only a smooth multiplicative degree-5 curve; supplied telluric transmission is unchanged.\n"
             "The enuma training split is unknown. Gray-only pixels are included; no new masks or line refits.",
             ha="center", fontsize=9)
    fig.savefig(OUT / filename, dpi=170)
    plt.close(fig)
    assert clipping == 0
    return dict(filename=filename, window_cm1=None if window is None else list(window),
                upper_axis_limits=list(top_limits), frame_residual_limits=list(frame_limits),
                matched_frame_residual_limits=list(all_limits), clipped_residual_points=clipping,
                upper_panel_clipped_points=0, sha256=sha256(OUT/filename))


def main():
    arrays, audit = compare()
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    audit["figures"] = [draw(arrays, audit, "order7_spectral_fit.png"),
                         draw(arrays, audit, "order7_spectral_fit_enuma_4383.png", (4382.25, 4383.65))]
    audit["script_sha256"] = sha256(Path(__file__))
    audit["array_sha256"] = sha256(OUT/"order7_enuma_comparison.npz")
    (OUT/"order7_enuma_comparison.json").write_text(json.dumps(audit, indent=2)+"\n")
    return audit


def report_lines(audit):
    """Replace the existing spectral-fit section without appending a new report."""
    m = audit["metrics"]
    return ["", "![Observed spectrum, ExoJAX and enuma overlays, and separate residual columns](ASSET_URL/order7_spectral_fit.png)", "",
            "**Upper panel:** frame 30 data (black), current ExoJAX full model plus Q5 (red), "
            "supplied enuma full model (purple solid), and its optional smooth blaze adjustment (purple dashed). "
            "The ExoJAX stellar model remains visible (green dashed). Telluric-only curves are "
            "orange dotted for ExoJAX and teal dotted for enuma; these curves are unchanged by the blaze adjustment. "
            "**Residual columns:** ExoJAX on the left; enuma supplied and blaze-adjusted on the right, "
            "with the same vertical scales. The middle row shows frame 30 and the bottom row shows "
            "frames 10, 30, 50 and 70 with their mean. Blue points mark the geometric score subset "
            "of the corrected residual (ExoJAX + Q5 on the left, enuma + blaze adjustment on the right). "
            "The gray enuma curve is the supplied-model residual. Gray-only pixels are also included. "
            "Yellow bands remain included strong absorption.", "",
            "The supplied `series_model.npz` contains 79 frames. Its wavelength array is in nm, "
            "with Order 7 at enuma index 38 (released cube index 5). The pixels match the original "
            "grid after wavelength-label quantization, so no interpolation or fitted velocity shift is applied.", "",
            "**Normalization is not documented in the NPZ and has not been confirmed.** The full model "
            "is consistent with normalization by each frame's full-order 95th percentile. "
            "That inferred scaling is converted to the same original ExoJAX continuum units. "
            "Telluric transmission is already dimensionless and is not rescaled. "
            "The optional blaze diagnostic multiplies enuma by a smooth degree-five polynomial "
            "fitted on all pixels in each frame; it changes no spectral-line parameters. "
            "The supplied enuma training split is unknown, so these are matched-frame residuals, "
            "not a claim of enuma held-out validation.", "",
            "| Model / display treatment | All-pixel RMS, frames 10/30/50/70 |",
            "| --- | ---: |",
            f"| Current ExoJAX + Q5 | {m['exojax_current_q5']['four_frame_all_rms_percent']:.6f}% |",
            f"| Supplied enuma; inferred flux normalization | {m['enuma_supplied']['four_frame_all_rms_percent']:.6f}% |",
            f"| enuma + multiplicative degree-five blaze adjustment | {m['enuma_multiplicative_degree5']['four_frame_all_rms_percent']:.6f}% |",
            f"| enuma + the same additive Q5 projection used for ExoJAX | {m['enuma_additive_q5']['four_frame_all_rms_percent']:.6f}% |", "",
            f"The smooth multiplicative correction removes {100*audit['residual_sse_fraction_removed_by_blaze']:.2f}% "
            "of enuma's residual sum of squares. Thus a smooth blaze mismatch has limited explanatory "
            "power for this comparison. Using the existing ExoJAX scalar instead of the inferred full-order "
            f"scalar gives {audit['alternative_saved_scalar']['raw_rms_percent']:.4f}% before correction and "
            f"{audit['alternative_saved_scalar']['additive_q5_rms_percent']:.4f}% after Q5; "
            "the normalization assumption does not explain most of the difference.", "",
            "<details><summary>4383 cm⁻¹ close-up of the same overlay</summary>", "",
            "![4383 cm-1 spectrum overlay and separate model residual columns](ASSET_URL/order7_spectral_fit_enuma_4383.png)",
            "", "The close-up uses the same full-order corrections; no extra local fit is performed. "
            "Its displayed RMS uses the visible 4382.25–4383.65 cm⁻¹ interval.",
            "", "</details>", ""]


if __name__ == "__main__":
    result = main()
    print(json.dumps(result["metrics"], indent=2))
