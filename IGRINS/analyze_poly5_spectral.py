"""Locate residual power after the adopted degree-five smooth subtraction.

Molecular categories are frozen from earlier training spectra. New intervals
are selected from filtered training residuals before reading evaluation data.
"""
from __future__ import annotations

import csv
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

from diagnose_spectral_budget import LABELS, PLAN, intervals
from smooth_residuals import smoother_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def load_npz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def metrics(residual, continuum, selection, total_continuum, total_fixed):
    selected = np.broadcast_to(selection, residual.shape)
    values = residual[selected]
    fixed = (residual*continuum)[selected]
    n = values.size
    power = float(np.sum(values**2))
    fixed_power = float(np.sum(fixed**2))
    common = np.broadcast_to(residual.mean(axis=1, keepdims=True), residual.shape)
    variable = residual-common
    fixed_full = residual*continuum
    fixed_common = np.broadcast_to(fixed_full.mean(axis=1, keepdims=True), residual.shape)
    fixed_variable = fixed_full-fixed_common
    common_power = float(np.sum(common[selected]**2))
    variable_power = float(np.sum(variable[selected]**2))
    fixed_common_power = float(np.sum(fixed_common[selected]**2))
    fixed_variable_power = float(np.sum(fixed_variable[selected]**2))
    np.testing.assert_allclose(common_power+variable_power, power, atol=1e-14, rtol=1e-12)
    np.testing.assert_allclose(fixed_common_power+fixed_variable_power, fixed_power, atol=1e-14, rtol=1e-12)
    return dict(n_samples=int(n), continuum_sse=power, fixed_sse=fixed_power,
                continuum_rms_percent=float(100*np.sqrt(np.mean(values**2))) if n else None,
                fixed_rms_percent=float(100*np.sqrt(np.mean(fixed**2))) if n else None,
                continuum_bias_percent=float(100*np.mean(values)) if n else None,
                fixed_bias_percent=float(100*np.mean(fixed)) if n else None,
                continuum_sse_percent=100*power/total_continuum,
                fixed_sse_percent=100*fixed_power/total_fixed,
                common_continuum_rms_percent=100*np.sqrt(common_power/n) if n else None,
                variable_continuum_rms_percent=100*np.sqrt(variable_power/n) if n else None,
                common_continuum_sse_percent=100*common_power/total_continuum,
                variable_continuum_sse_percent=100*variable_power/total_continuum,
                within_region_common_sse_fraction=common_power/power if power else None,
                common_fixed_rms_percent=100*np.sqrt(fixed_common_power/n) if n else None,
                variable_fixed_rms_percent=100*np.sqrt(fixed_variable_power/n) if n else None,
                common_fixed_sse_percent=100*fixed_common_power/total_fixed,
                variable_fixed_sse_percent=100*fixed_variable_power/total_fixed)


def region_record(b, order, start, end, mean, training, continuum, wave, category):
    peak = start+np.argmax(np.abs(mean[b, start:end]))
    return dict(order=int(order), order_index=b, array_interval=[int(start), int(end)],
                n_wavelength_pixels=int(end-start), peak_array_index=int(peak),
                peak_wavelength_um=float(wave[b, peak]),
                wavelength_min_um=float(np.min(wave[b, start:end])),
                wavelength_max_um=float(np.max(wave[b, start:end])),
                peak_mean_continuum_percent=float(100*mean[b, peak]),
                peak_category=LABELS[category[b, peak]],
                training_fixed_sse=float(np.sum((training[b, :, start:end]*continuum[b, :, start:end])**2)))


def main():
    source_path = OUT / "motion_clean_control.npz"
    source = load_npz(source_path)
    frozen_path = OUT / "nonstellar_spectral_selection.npz"
    frozen = load_npz(frozen_path)
    old_selection_path = OUT / "nonstellar_spectral_selection.json"
    old_regions = json.loads(old_selection_path.read_text())["regions"]
    included = source["included_mask"]
    wave, orders = source["wavelength_um"], source["orders"]
    category = frozen["category"]
    np.testing.assert_array_equal(category >= 0, included)
    np.testing.assert_array_equal(wave, frozen["wavelength_um"])
    np.testing.assert_array_equal(source["exposures"][:-1], [0, 19, 39, 59, 78])
    operators = np.array([smoother_matrix(w, keep, dict(kind="polynomial", degree=5))
                          for w, keep in zip(wave, included)])
    train_original = source["residual_continuum"][:, :-1]
    train = train_original - np.einsum("bij,bej->bei", operators, train_original)
    train_continuum = source["continuum"][:, :-1]
    train_mean = train.mean(axis=1)
    new_regions = []
    new_mask = np.zeros_like(included)
    old_mask = frozen["region_mask"]
    for b, order in enumerate(orders):
        for start, end in intervals(train_mean[b], included[b]):
            assert np.all(included[b, start:end]) and not np.any(new_mask[b, start:end])
            new_mask[b, start:end] = True
            new_regions.append(region_record(b, order, start, end, train_mean, train,
                                             train_continuum, wave, category))
    new_regions.sort(key=lambda r: r["training_fixed_sse"], reverse=True)
    for i, row in enumerate(new_regions, 1):
        row["rank"] = i
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), filter=dict(kind="polynomial", degree=5),
                     interval_rule=PLAN, regions=new_regions,
                     category_source_sha256=hashlib.sha256(frozen_path.read_bytes()).hexdigest(),
                     training_source_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
                     interpretation="Region ranking and molecular labels use training data only; no physical cause fractions")
    selection_path = OUT / "poly5_spectral_selection.json"
    selection_path.write_text(json.dumps(selection, indent=2)+"\n")
    np.savez_compressed(OUT / "poly5_spectral_selection.npz", category=category,
                        post_filter_region_mask=new_mask, previous_region_mask=old_mask,
                        wavelength_um=wave, orders=orders)
    # The final scoring arrays are accessed only after post-filter regions are fixed.
    final_path = OUT / "smooth_results.npz"
    final = load_npz(final_path)
    residual = final["residual_continuum"] - final["correction_poly5_all_included"]
    rebuilt = final["residual_continuum"]-np.einsum("bij,bej->bei", operators, final["residual_continuum"])
    np.testing.assert_allclose(residual, rebuilt, atol=1e-13, rtol=0)
    np.testing.assert_array_equal(included, final["included_mask"])
    scopes = dict(training=(train, train_continuum, included[:, None]),
                  final_score=(residual, final["continuum"], final["score_mask"][:, None]),
                  final_all_included=(residual, final["continuum"], included[:, None]))
    totals, rows = {}, []
    previous_regions = []
    for old in old_regions:
        b, (start, end) = old["order_index"], old["array_interval"]
        row = region_record(b, old["order"], start, end, train_mean, train,
                            train_continuum, wave, category)
        row["rank"] = old["rank"]
        row["selection_peak_before_filter_um"] = old["peak_wavelength_um"]
        row["selection_peak_before_filter_percent"] = old["peak_mean_continuum_percent"]
        previous_regions.append(row)
    region_sets = {"post_filter_training_selection": new_regions,
                   "previous_training_selection": previous_regions}
    for scope, (r, c, keep) in scopes.items():
        mask = np.broadcast_to(keep, r.shape)
        cp = float(np.sum(r[mask]**2))
        fp = float(np.sum((r*c)[mask]**2))
        totals[scope] = metrics(r, c, mask, cp, fp)
        for order in ["all", *map(int, orders)]:
            order_keep = mask.copy()
            if order != "all":
                order_keep[orders != order] = False
            for label_index, label in enumerate(LABELS):
                match = order_keep & (category[:, None] == label_index)
                rows.append(dict(scope=scope, order=order, category=label,
                                 **metrics(r, c, match, cp, fp)))
        pooled = [row for row in rows if row["scope"] == scope and row["order"] == "all"]
        assert sum(row["n_samples"] for row in pooled) == int(mask.sum())
        np.testing.assert_allclose(sum(row["continuum_sse"] for row in pooled), cp, rtol=1e-13)
        np.testing.assert_allclose(sum(row["fixed_sse"] for row in pooled), fp, rtol=1e-13)
        for set_name, region_rows in region_sets.items():
            union = np.zeros_like(mask)
            for row in region_rows:
                b, (start, end) = row["order_index"], row["array_interval"]
                local = np.zeros_like(mask)
                local[b, :, start:end] = mask[b, :, start:end]
                assert not np.any(union & local)
                union |= local
                row[scope] = metrics(r, c, local, cp, fp)
            totals[scope][set_name] = metrics(r, c, union, cp, fp)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), filter="Degree-five additive residual subtraction; all included pixels",
                  units="All percentages are original-continuum relative unless prefixed fixed",
                  totals=totals, categories=rows, region_sets=region_sets,
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [source_path, frozen_path, old_selection_path, final_path,
                                              selection_path, Path(__file__)]},
                  caveats=["Scoring pixels participate in smoothing; reported values are post-filter descriptive residuals.",
                           "Frozen molecular categories identify wavelength overlap, not physical error causes.",
                           "The projection is nonlocal: polynomial estimation at another wavelength can affect local residuals.",
                           "Neither additional masks nor atmospheric parameters are fitted here.",
                           "Stellar mismatch is not excluded by a weak predicted stellar spectrum.",
                           "Apply the same filter to planetary templates; do not subtract the time-common residual as a science correction."])
    (OUT / "poly5_spectral_budget.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUT / "poly5_spectral_budget.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    flat = []
    for set_name, region_rows in region_sets.items():
        for row in region_rows:
            flat.append(dict(region_set=set_name, rank=row["rank"], order=row["order"],
                             peak_wavelength_um=row["peak_wavelength_um"],
                             training_peak_continuum_percent=row["peak_mean_continuum_percent"],
                             final_score_samples=row["final_score"]["n_samples"],
                             final_score_rms_percent=row["final_score"]["continuum_rms_percent"],
                             final_score_sse_percent=row["final_score"]["continuum_sse_percent"],
                             final_all_sse_percent=row["final_all_included"]["continuum_sse_percent"]))
    with (OUT / "poly5_spectral_regions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), constrained_layout=True)
    colors = ["tab:blue", "tab:orange", "tab:purple", "0.6"]
    final_mean = residual.mean(axis=1)
    for b, (order, ax) in enumerate(zip(orders, axes)):
        for i, (label, color) in enumerate(zip(LABELS, colors)):
            keep = category[b] == i
            ax.scatter(wave[b, keep], 100*train_mean[b, keep], s=8, color=color, label=label)
        ax.plot(wave[b], np.where(final["score_mask"][b], 100*final_mean[b], np.nan), color="k", lw=.8, label="Evaluation mean on score pixels")
        for row in new_regions:
            if row["order_index"] == b:
                ax.axvspan(row["wavelength_min_um"], row["wavelength_max_um"], color="red", alpha=.1)
                ax.text(row["peak_wavelength_um"], 5.6, str(row["rank"]), ha="center", fontsize=8)
        ax.axhline(0, color="0.7", lw=.7)
        ax.set(title=f"Order {order}: after degree-five subtraction", xlabel="Observed vacuum wavelength (micron)",
               ylabel="Residual / original continuum (%)", ylim=(-6, 6))
    axes[0].legend(fontsize=8, ncol=3, loc="lower right")
    fig.savefig(OUT / "poly5_spectral_budget.png", dpi=150); plt.close(fig)
    lines = ["Residual wavelength budget after degree-five polynomial removal", "="*62, "",
             "All RMS and error shares below use the original fitted-continuum units.",
             "The polynomial is fitted on all included pixels for each exposure/order.",
             "Thus scoring pixels participate in the smooth correction; this is a",
             "descriptive residual budget, not independently predicted accuracy.", "",
             "The four-exposure scoring RMS is 0.926565%, not 0.73%. The previous",
             "flux normalization gives 0.815589%. Its time-common component alone is",
             "0.737144% in that previous normalization; order13 continuum RMS is",
             "0.728007%. Neither is the pooled total continuum RMS.", "",
             "The pooled continuum time-common component is 0.839171% (82.0255%",
             "of squared error), with time-variable component 0.392829% (17.9745%).",
             "These powers add exactly; the finite four-exposure mean contains noise.", "",
             "Frozen molecular categories", "---------------------------",
             "Category                N     RMS (%)   Common RMS (%)  Variable RMS (%)  SSE share (%)"]
    for row in rows:
        if row["scope"] == "final_score" and row["order"] == "all":
            lines.append(f"{row['category']:22s} {row['n_samples']:5d} {row['continuum_rms_percent']:11.6f}"
                         f" {row['common_continuum_rms_percent']:16.6f} {row['variable_continuum_rms_percent']:17.6f}"
                         f" {row['continuum_sse_percent']:14.4f}")
    lines += ["", "Labels use the existing training-only H2O/CO2 marginal-depth partition,",
              "unchanged by filtering. They locate residuals and do not measure the",
              "fraction caused by that molecule. The projection is nonlocal, so a",
              "residual in one category can alter the subtraction in another category.", "",
              "Training-selected post-filter intervals", "---------------------------------------",
              "Peak (micron)   Category               Final N   Local RMS (%)  Total SSE (%)  Common fraction (%)"]
    for row in new_regions:
        value = row["final_score"]
        if value["n_samples"]:
            lines.append(f"{row['peak_wavelength_um']:.9f} {row['peak_category']:22s} {value['n_samples']:7d}"
                         f" {value['continuum_rms_percent']:15.6f} {value['continuum_sse_percent']:14.4f}"
                         f" {100*value['within_region_common_sse_fraction']:20.4f}")
    lines += ["", "The four scoring intervals occupy 18 wavelength pixels (72 samples),",
              "and contain 24.2594% of remaining squared error. Their power is 95.1284%",
              "time-common. The selected 1.500091 and 1.494845 micron intervals lie",
              "outside the scoring blocks, though their training residuals remain large.",
              "The same threshold/bridging/padding rule as the earlier analysis was",
              "applied to filtered training means before reading evaluation arrays.",
              "No mask was changed and no interval was selected from final residuals.", "",
              "Interpretation", "--------------",
              "The newly selected 1.49797082 micron trough is H2O-dominant. At its",
              "training peak the modeled marginal H2O depth is 9.38%, CO2 0.090%,",
              "and stellar depth 0.598%. Its mean residual changes from -2.971% before",
              "filtering to -3.215% afterward. It was already present and is newly",
              "above the interval-selection threshold. It is not evidence of a newly",
              "discovered missing transition; line parameters, wavelength/IP mismatch,",
              "and unknown stellar structure remain possible explanations.", "",
              "The persistent 2.076826 and 1.570499 micron structures overlap CO2.",
              "The 1.500570 micron positive residual lies at weak modeled absorption.",
              "They remain strongly time-common after removing the broad polynomial.",
              "This supports localized reproducible mismatch, not a unique attribution",
              "to line lists or instrumental reduction. Prior pressure-shift and line-",
              "shape sensitivities are relevant bounds, not causal fractions.", "",
              "The high-pass projection can produce signed wings. The sign of a filtered",
              "residual alone cannot establish whether adding absorption improves it:",
              "every proposed line/physics response must be passed through the same",
              "degree-five operator. Unfiltered positive peaks provide separate evidence",
              "for continuum/background alternatives, but do not uniquely identify them.", "",
              "The ambiguous 1.494855 micron blend remains outside current score pixels",
              "and can still affect calibration and the global smooth correction.",
              "No planetary detection or signal amplitude is inferred from these budgets.",
              "The temporal mean is used diagnostically, not subtracted from science data.", "",
              "Validation: saved poly5 correction reproduced; fixed category counts/SSE",
              "sum to totals; common+variable powers close for each category and region;",
              "all regions are disjoint and included; source hashes saved in JSON.", "",
              "Files: poly5_spectral_budget.json/csv/png, poly5_spectral_regions.csv,",
              "poly5_spectral_selection.json/npz. Reproduce with:",
              "  python IGRINS/analyze_poly5_spectral.py"]
    (OUT / "poly5_spectral_report.txt").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(totals=totals, final_categories=[row for row in rows if row["scope"] == "final_score" and row["order"] == "all"],
                          regions=flat), indent=2))


if __name__ == "__main__":
    main()
