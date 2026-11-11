"""Describe fixed-mask residuals and conditional line-intensity corrections."""
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def residual_statistics(rows, mask):
    """Partition squared residual into exposure mean and temporal variation."""
    keep = np.broadcast_to(mask[:, None], rows.shape)
    count = int(keep.sum())
    if not count:
        return dict(n_samples=0, sse=0., rms_percent=None,
                    common_rms_percent=None, temporal_rms_percent=None)
    mean = rows.mean(axis=1)
    sse = float(np.sum(rows[keep]**2))
    common_sse = float(rows.shape[1]*np.sum(mean[mask]**2))
    temporal_sse = float(np.sum((rows-mean[:, None])[keep]**2))
    np.testing.assert_allclose(common_sse+temporal_sse, sse, rtol=1e-12)
    return dict(n_samples=count, sse=sse, rms_percent=float(100*np.sqrt(sse/count)),
                common_rms_percent=float(100*np.sqrt(common_sse/count)),
                temporal_rms_percent=float(100*np.sqrt(temporal_sse/count)),
                common_sse=common_sse, temporal_sse=temporal_sse)


def correction_statistics(theta, upper, strength, keep):
    z, delta, weight = theta[keep], (upper*theta)[keep], strength[keep]
    return dict(n_lines=int(keep.sum()),
                outside_original_uncertainty_count=int(np.sum(abs(z)>1+1e-6)),
                above_three_uncertainty_scales_count=int(np.sum(abs(z)>3+1e-6)),
                above_ten_uncertainty_scales_count=int(np.sum(abs(z)>10+1e-6)),
                absolute_fraction_above_10_percent_count=int(np.sum(abs(delta)>.1)),
                absolute_fraction_above_20_percent_count=int(np.sum(abs(delta)>.2)),
                absolute_fraction_above_50_percent_count=int(np.sum(abs(delta)>.5)),
                zero_intensity_count=int(np.sum(1+delta<1e-5)),
                median_absolute_fraction=float(np.median(abs(delta))),
                p90_absolute_fraction=float(np.percentile(abs(delta), 90)),
                maximum_absolute_fraction=float(np.max(abs(delta))),
                mean_fraction=float(np.mean(delta)),
                strength_weighted_mean_fraction=float(np.average(delta, weights=weight)),
                minimum_factor=float(np.min(1+delta)), maximum_factor=float(np.max(1+delta)))


def line_examples(records, theta, upper, keep, limit=12):
    """Rank by fractional change; nearest distances concern selected lines only."""
    indices = np.flatnonzero(keep)
    indices = indices[np.argsort(-abs((upper*theta)[keep]))[:limit]]
    result = []
    for index in indices:
        record = records[index]
        neighbors = [(C_KMS*abs(np.log(other["nu"]/record["nu"])), j)
                     for j, other in enumerate(records)
                     if j != index and other["order"] == record["order"]]
        separation, nearest = min(neighbors)
        result.append(dict(index=int(index), order=record["order"], species=record["species"],
                           nu_cm1=record["nu"], wavelength_um=1e4/record["nu"],
                           reference_strength_296K=record["sw"], ierr=record["ierr"],
                           upper_fraction=float(upper[index]), z=float(theta[index]),
                           fractional_correction=float(upper[index]*theta[index]),
                           intensity_factor=float(1+upper[index]*theta[index]),
                           nearest_included_velocity_kms=record["nearest_included_velocity_kms"],
                           nearest_selected_neighbor_index=int(nearest),
                           nearest_selected_neighbor_species=records[nearest]["species"],
                           nearest_selected_neighbor_separation_kms=float(separation)))
    return result


def main():
    paths = [OUT/"relaxed_strength_fit.json", OUT/"relaxed_strength_fit.npz",
             OUT/"line_strength_fit.json", OUT/"line_strength_fit.npz",
             OUT/"poly5_spectral_budget.json", OUT/"nonstellar_spectral_selection.npz"]
    meta = json.loads(paths[0].read_text())
    old_meta = json.loads(paths[2].read_text())
    regions = json.loads(paths[4].read_text())["region_sets"]["post_filter_training_selection"]
    with np.load(paths[1]) as data:
        arrays = {key: data[key] for key in data.files}
    with np.load(paths[3]) as data:
        old = {key: data[key] for key in data.files}
    with np.load(paths[5]) as data:
        category = data["category"]
        np.testing.assert_array_equal(data["wavelength_um"], arrays["wavelength_um"])
    for key in ["wavelength_um", "included_mask", "score_mask", "y_over_c", "continuum"]:
        np.testing.assert_array_equal(arrays[key], old[key])
    assert meta["line_records"] == old_meta["line_records"]
    np.testing.assert_allclose(arrays["component_cap1_prior1"], old["component_primary"], atol=1e-9, rtol=0)
    np.testing.assert_allclose(arrays["residual_baseline"], old["residual_baseline"], atol=1e-12, rtol=0)
    records, selected = meta["line_records"], meta["selected_case"]
    upper = np.array([record["uncertainty_upper"] for record in records])
    strength = np.array([record["sw"] for record in records])
    groups = {"all": np.ones(len(records), dtype=bool),
              "H2O": np.array([record["species"] == "H2O" for record in records]),
              "CO2": np.array([record["species"] == "CO2" for record in records]),
              "S296_ge_1e-23": strength>=1e-23,
              "S296_1e-24_to_1e-23": (strength>=1e-24)&(strength<1e-23),
              "S296_lt_1e-24": strength<1e-24}
    residuals = {"baseline": arrays["residual_baseline"][:, 5:],
                 "previous_tight": old["residual_primary"][:, 5:]}
    residuals.update({name: arrays[f"residual_{name}"][:, 5:] for name in meta["cases"]})
    score = arrays["score_mask"]
    overall = {name: residual_statistics(rows, score) for name, rows in residuals.items()}
    for name in meta["cases"]:
        np.testing.assert_allclose(overall[name]["rms_percent"],
                                   meta["cases"][name]["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12)
    categories = []
    for code, label in enumerate(["H2O-dominant", "CO2-dominant", "Mixed", "Weak absorption"]):
        mask = score&(category == code)
        categories.append(dict(category=label, statistics={
            name: residual_statistics(rows, mask) for name, rows in residuals.items()}))
    for name in residuals:
        np.testing.assert_allclose(sum(row["statistics"][name]["sse"] for row in categories),
                                   overall[name]["sse"], rtol=1e-12)
    region_results = []
    for region in regions:
        mask = np.zeros_like(score)
        b, (start, stop) = region["order_index"], region["array_interval"]
        mask[b, start:stop] = arrays["included_mask"][b, start:stop]
        region_results.append(dict(order=region["order"], peak_wavelength_um=region["peak_wavelength_um"],
            wavelength_min_um=region["wavelength_min_um"], wavelength_max_um=region["wavelength_max_um"],
            selection="Frozen post-filter training-selected interval", array_interval=[start, stop],
            score={name: residual_statistics(rows, mask&score) for name, rows in residuals.items()},
            all_included={name: residual_statistics(rows, mask) for name, rows in residuals.items()}))
    corrections = {}
    for name in meta["cases"]:
        theta = arrays[f"theta_{name}"]
        assert np.min(1+upper*theta) >= -1e-12
        corrections[name] = dict(groups={label: correction_statistics(theta, upper, strength, keep)
                                         for label, keep in groups.items()},
            largest_strong_line_changes=line_examples(records, theta, upper, groups["S296_ge_1e-23"]),
            largest_weak_line_changes=line_examples(records, theta, upper, groups["S296_lt_1e-24"]))
    prior_sensitivity = {}
    reference = overall["previous_tight"]
    for name, values in overall.items():
        prior_sensitivity[name] = dict(
            rms_reduction_from_baseline_fraction=1-values["rms_percent"]/overall["baseline"]["rms_percent"],
            sse_reduction_from_baseline_fraction=1-values["sse"]/overall["baseline"]["sse"],
            rms_reduction_from_tight_fraction=1-values["rms_percent"]/reference["rms_percent"],
            sse_reduction_from_tight_fraction=1-values["sse"]/reference["sse"])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=selected,
        normalization="Original fitted continuum; identical degree-five projection and fixed score mask",
        evaluation_exposures=arrays["exposures"][5:].tolist(), overall=overall, categories=categories,
        regions=region_results, corrections=corrections, improvements=prior_sensitivity,
        caveats=["HITRAN uncertainty-bin upper edges are scales for a conditional prior, not known Gaussian standard deviations.",
                 "Reference strength at 296 K is only a descriptive grouping, not a measure of observed information or spectral impact.",
                 "The nearest-neighbor search covers selected correction lines only; the full fixed line list contains additional blends.",
                 "Individual corrections are degenerate for unresolved blends and with the frozen molecular columns.",
                 "The fitted effective intensities can compensate for missing lines, wavelength errors, IP, TP, continuum and stellar-model errors.",
                 "Category and interval residual improvements locate changes but do not uniquely identify their physical cause.",
                 "Polynomial coefficients are estimated from all included pixels in each evaluation exposure; score RMS is post-filter descriptive.",
                 "Training CV is conditional on previously fitted physical nuisance parameters and tests the given prior-width grid only."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths+[Path(__file__)]})
    (OUT/"relaxed_strength_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for b, ax in enumerate(axes):
        for name, label, color in [("baseline", "Fixed HITRAN", "0.65"),
                                   ("previous_tight", "Previous hard cap", "C0"),
                                   (selected, f"Selected: {selected}", "C3")]:
            mean = np.where(score[b], residuals[name][b].mean(axis=0), np.nan)
            ax.plot(arrays["wavelength_um"][b], 100*mean, label=label, color=color, lw=.9)
        for region in regions:
            if region["order_index"] == b:
                ax.axvspan(region["wavelength_min_um"], region["wavelength_max_um"], color="0.7", alpha=.15)
        ax.axhline(0, color="0.3", lw=.5)
        ax.set(xlabel="Observed wavelength (micron)", ylabel="Mean filtered residual (%)",
               title=f"Order {arrays['orders'][b]}: fixed score wavelengths, four evaluation exposures")
        ax.legend(fontsize=8)
    fig.savefig(OUT/"relaxed_strength_spectral_diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(selected_case=selected, overall=overall,
                          selected_correction_groups=corrections[selected]["groups"]), indent=2))


if __name__ == "__main__":
    main()
