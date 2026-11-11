"""Measure temporal scatter after the provisionally adopted polynomial filter.

Adjacent differences quantify short-time scatter, not identified photon noise.
The physical fits are reused from the 75-exposure diagnostic without refitting.
"""
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

from smooth_residuals import smoother_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CONFIG = dict(kind="polynomial", degree=5)


def correlation(x, y):
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x.ravel(), y.ravel())[0, 1])


def statistics(values):
    return dict(rms_percent=float(100*np.sqrt(np.mean(values**2))),
                robust_mad_percent=float(100*1.4826*np.median(np.abs(values-np.median(values)))),
                mean_percent=float(100*np.mean(values)))


def describe(residual, mask, left, right):
    selection = np.broadcast_to(mask[:, None], residual.shape)
    mean = residual.mean(axis=1, keepdims=True)
    variable = residual-mean
    differences = (residual[:, right]-residual[:, left])/np.sqrt(2)
    pair_selection = np.broadcast_to(mask[:, None], differences.shape)
    adjacent = mask[:, 1:] & mask[:, :-1]
    spatial_selection = np.broadcast_to(adjacent[:, None], differences[:, :, :-1].shape)
    raw_spatial_selection = np.broadcast_to(adjacent[:, None], residual[:, :, :-1].shape)
    total = float(np.mean(residual[selection]**2))
    common = float(np.mean(mean[:, 0][mask]**2))
    time_power = float(np.mean(variable[selection]**2))
    cross = float(np.mean((mean*variable)[selection]))
    np.testing.assert_allclose(common+time_power, total, atol=1e-16, rtol=1e-12)
    return dict(pixels_per_exposure=int(mask.sum()), n_samples=int(selection.sum()),
                native_pair_count=int(left.size), difference_samples=int(pair_selection.sum()),
                residual=statistics(residual[selection]),
                common=dict(rms_percent=float(100*np.sqrt(common)), squared_error_fraction=common/total),
                temporal=dict(**statistics(variable[selection]), squared_error_fraction=time_power/total),
                adjacent_difference=statistics(differences[pair_selection]),
                temporal_lag1_correlation=correlation(variable[:, left][pair_selection],
                                                      variable[:, right][pair_selection]),
                difference_spectral_lag1_correlation=correlation(differences[:, :, :-1][spatial_selection],
                                                                 differences[:, :, 1:][spatial_selection]),
                residual_spectral_lag1_correlation=correlation(residual[:, :, :-1][raw_spatial_selection],
                                                               residual[:, :, 1:][raw_spatial_selection]),
                closure_error=common+time_power-total, common_temporal_cross_product=cross)


def main():
    source = OUT / "nonstellar_noise.npz"
    with np.load(source) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(OUT / "motion_clean_control.npz") as saved:
        score = saved["score_mask"][:, -1].copy()
        np.testing.assert_array_equal(data["allowed"], saved["included_mask"])
        np.testing.assert_array_equal(data["wavelength_um"], saved["wavelength_um"])
    allowed = data["allowed"]
    orders = data["orders"]
    exposures = data["exposures"]
    assert not np.isin(exposures, [10, 30, 50, 70]).any()
    left, right = data["pair_left"], data["pair_right"]
    np.testing.assert_array_equal(exposures[right]-exposures[left], 1)
    matrices = np.array([smoother_matrix(wave, mask, CONFIG)
                         for wave, mask in zip(data["wavelength_um"], allowed)])
    raw = (data["observed"]-data["model"])/data["continuum"]
    np.testing.assert_allclose(raw, data["residual_continuum"], atol=1e-15)
    correction = np.einsum("bij,bej->bei", matrices, raw)
    filtered = raw-correction
    # Preserve original units: filtering acts on continuum residuals before C.
    fixed_filtered = data["continuum"]*filtered
    full_selection = np.broadcast_to(allowed[:, None], raw.shape)
    filter_cross = float(np.mean((correction*filtered)[full_selection]))
    np.testing.assert_allclose(filter_cross, 0., atol=1e-15)
    np.testing.assert_allclose(np.einsum("bij,bej->bei", matrices, filtered)[:, :, :], 0., atol=1e-13)
    direct_difference = (filtered[:, right]-filtered[:, left])/np.sqrt(2)
    raw_difference = (raw[:, right]-raw[:, left])/np.sqrt(2)
    filtered_difference = raw_difference-np.einsum("bij,bej->bei", matrices, raw_difference)
    np.testing.assert_allclose(direct_difference, filtered_difference, atol=1e-15)
    statistics_by_units = {}
    for units, before, after in [("continuum", raw, filtered),
                                  ("fixed_flux", data["residual_fixed_scale"], fixed_filtered)]:
        statistics_by_units[units] = {}
        for mask_name, mask in [("all_included", allowed), ("same_scoring_subset", score)]:
            item = dict(before=describe(before, mask, left, right),
                        after=describe(after, mask, left, right))
            item["per_order"] = {str(order): describe(after[b:b+1], mask[b:b+1], left, right)
                                  for b, order in enumerate(orders)}
            statistics_by_units[units][mask_name] = item
    cadence = data["time_minutes"][right]-data["time_minutes"][left]
    ordinary = cadence < 1.5*np.median(cadence)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  source="nonstellar_noise.npz", config=CONFIG,
                  scope="Existing 75-exposure full-mask nuisance fits, then fixed degree-five filtering",
                  excluded_exposures=[10, 30, 50, 70], exposures=exposures.tolist(),
                  orders=orders.tolist(), included_pixels=int(allowed.sum()),
                  scoring_subset_pixels=int(score.sum()),
                  native_pair_count=int(left.size),
                  pair_cadence_minutes=dict(median=float(np.median(cadence)),
                                           minimum=float(cadence.min()), maximum=float(cadence.max())),
                  statistics=statistics_by_units,
                  ordinary_cadence_pair_count=int(ordinary.sum()),
                  ordinary_cadence_continuum=describe(filtered, allowed, left[ordinary], right[ordinary]),
                  filter_common_cross_product=filter_cross,
                  difference_commutation_max_error=float(np.max(np.abs(direct_difference-filtered_difference))),
                  limitations=[
                      "Adjacent differences include noise, rapid atmosphere, calibration and extraction changes.",
                      "Temporal correlation can suppress difference RMS; this is not a noise upper bound.",
                      "All retained pixels entered the original local nuisance fits and the polynomial filter.",
                      "The same-scoring subset matches wavelengths, but not the calibration-only fitting protocol of the four evaluation exposures.",
                      "The mean of 75 exposures and that of four exposures have different finite-sample noise contributions.",
                      "Do not subtract this difference variance from the four-exposure residual variance as an identified noise correction.",
                      "Common and temporal fractions describe residual behavior, not identified physical causes.",
                      "The original continuum stays fixed; fixed-flux residual means C times filtered continuum residual."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [source, OUT / "motion_clean_control.npz", Path(__file__), ROOT / "smooth_residuals.py"]})
    (OUT / "poly5_temporal.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "poly5_temporal.npz", orders=orders, exposures=exposures,
                        wavelength_um=data["wavelength_um"], included_mask=allowed,
                        scoring_mask=score, residual_continuum=filtered,
                        residual_fixed_flux=fixed_filtered, pair_left=left, pair_right=right,
                        time_minutes=data["time_minutes"])
    lines = ["Temporal scatter after degree-five low-frequency removal", "="*54, "",
             "Original physical fits are reused. No reserved exposure enters this diagnostic.",
             "Filtering is fitted to all 1467 retained pixels separately for each order/exposure.",
             "The 523-pixel scoring subset matches the four-exposure wavelength selection,",
             "but these 75 original nuisance fits used all retained pixels.", "",
             "Scope and units          Total RMS   Common RMS   Temporal RMS   Adj.diff RMS   Adj.diff MAD"]
    for units in statistics_by_units:
        for scope in statistics_by_units[units]:
            s = statistics_by_units[units][scope]["after"]
            values = [s["residual"]["rms_percent"], s["common"]["rms_percent"],
                      s["temporal"]["rms_percent"], s["adjacent_difference"]["rms_percent"],
                      s["adjacent_difference"]["robust_mad_percent"]]
            lines.append(f"{units:10s} {scope:20s}"+" ".join(f"{v:12.6f}%" for v in values))
    lines += ["", "Native adjacent differences use 70 pairs, without bridging excluded frames.",
              "The proxy is (r[e+1]-r[e])/sqrt(2). MAD is scaled by 1.4826.",
              "RMS squares of the temporal mean and centered residual add exactly.",
              "Neither this split nor the difference proxy identifies physical causes.", ""]
    for scope in ["all_included", "same_scoring_subset"]:
        s = statistics_by_units["continuum"][scope]["after"]
        lines.append(f"{scope}: time lag-one correlation {s['temporal_lag1_correlation']:.6f}; "
                     f"difference neighboring-pixel correlation {s['difference_spectral_lag1_correlation']:.6f}.")
    lines += ["", "The four-exposure temporal mean removes one of four temporal degrees of",
              "freedom even for IID noise. This 75-exposure diagnostic is not directly",
              "comparable through variance subtraction. Temporal covariance, atmospheric",
              "evolution and different fitted pixel sets remain material limitations.", "",
              "All artifacts are within Git-ignored IGRINS/."]
    (OUT / "POLY5_TEMPORAL_REPORT.txt").write_text("\n".join(lines)+"\n")
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    t = data["time_minutes"]
    for b, ax in enumerate(axes):
        ax.plot(t, 100*np.sqrt(np.mean(raw[b][:, allowed[b]]**2, axis=1)), color="0.6", lw=1, label="Before polynomial")
        ax.plot(t, 100*np.sqrt(np.mean(filtered[b][:, allowed[b]]**2, axis=1)), lw=1, label="After polynomial 5")
        ax.plot((t[left]+t[right])/2, 100*np.sqrt(np.mean(direct_difference[b][:, allowed[b]]**2, axis=1)),
                lw=1, label="Adjacent difference / sqrt(2)")
        ax.set(ylabel="Continuum RMS (%)", title=f"Order {orders[b]}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Minutes since first exposure")
    fig.suptitle("75 exposures: polynomial-filtered residual and short-time scatter")
    fig.savefig(OUT / "poly5_temporal.png", dpi=160)
    plt.close(fig)
    for scope in ["all_included", "same_scoring_subset"]:
        print(scope, json.dumps(statistics_by_units["continuum"][scope]["after"], indent=2))


if __name__ == "__main__":
    main()
