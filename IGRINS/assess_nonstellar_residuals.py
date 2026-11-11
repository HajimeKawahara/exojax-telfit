"""Evaluate residual RMS and orthogonal morphology on the current stellar mask.

The statistical components are not physical source identifications. The model
and mask are frozen; the four scoring exposures have calibration-only nuisance
fits. A separate residual template uses only the five training exposures.
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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCE = "motion_clean_control"
EXPOSURES = [10, 30, 50, 70]


def load_fit(tag):
    meta = json.loads((OUT / f"{tag}.json").read_text())
    with np.load(OUT / f"{tag}.npz") as saved:
        return meta, {key: saved[key] for key in saved.files}


def statistics(values):
    values = np.asarray(values)
    return dict(n_pixels=int(values.size), rms=float(np.sqrt(np.mean(values**2))),
                bias=float(np.mean(values)),
                mad=float(1.4826*np.median(np.abs(values-np.median(values)))),
                p95_absolute=float(np.percentile(np.abs(values), 95)))


def common_projection(mean, mask, degree):
    basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, mean.shape[-1]), degree)
    return np.array([basis @ np.linalg.lstsq(basis[keep], row[keep], rcond=None)[0]
                     for row, keep in zip(mean, mask)])


def decomposition(residual, scoring, smooth, localized, variable):
    shape = residual.shape
    common_parts = [np.broadcast_to(component[:, None], shape) for component in [smooth, localized]]
    parts = common_parts + [variable]
    total = float(np.mean(residual[scoring]**2))
    powers = [float(np.mean(part[scoring]**2)) for part in parts]
    cross = {f"{a}_{b}": float(np.mean(parts[a][scoring]*parts[b][scoring]))
             for a in range(3) for b in range(a+1, 3)}
    np.testing.assert_allclose(sum(parts)[scoring], residual[scoring], atol=1e-14)
    np.testing.assert_allclose(sum(powers), total, atol=1e-14, rtol=1e-12)
    return dict(total_mse=total, parts={name: dict(rms=float(np.sqrt(power)),
                                                mse_fraction=power/total)
                                      for name, power in zip(["smooth_common", "localized_common", "time_variable"], powers)},
                cross_products=cross, identity_error=sum(powers)-total)


def template_transfer(residual, template, scoring):
    prediction = np.broadcast_to(template[:, None], residual.shape)
    remainder = residual-prediction
    total = float(np.mean(residual[scoring]**2))
    static_power = float(np.mean(prediction[scoring]**2))
    remaining_power = float(np.mean(remainder[scoring]**2))
    cross = float(2*np.mean(prediction[scoring]*remainder[scoring]))
    np.testing.assert_allclose(static_power+remaining_power+cross, total, rtol=1e-12)
    return dict(before=statistics(residual[scoring]), after=statistics(remainder[scoring]),
                template_mse=static_power, remainder_mse=remaining_power,
                twice_cross_product=cross, total_mse=total,
                scoring_sse_reduction_fraction=1-remaining_power/total,
                interpretation="Predictive SSE reduction, not a causal variance fraction; cross term retained")


def main():
    meta, source = load_fit(SOURCE)
    assert meta["train_exposures"] == [0, 19, 39, 59, 78]
    template = np.mean(source["residual_continuum"][:, :-1], axis=1)
    allowed = source["included_mask"]
    residuals, fixed, continua, score_masks, paths = [], [], [], [], []
    for exposure in EXPOSURES:
        tag = f"nonstellar_control_e{exposure}"
        other_meta, data = load_fit(tag)
        assert other_meta["training_fit"]["reused"] and other_meta["heldout_calibration_fit"]["success"]
        assert other_meta["pixel_mask"]["sha256"] == meta["pixel_mask"]["sha256"]
        np.testing.assert_array_equal(data["included_mask"], allowed)
        for key in ["observed", "model", "continuum", "component"]:
            np.testing.assert_allclose(data[key][:, :-1], source[key][:, :-1], rtol=0, atol=1e-10)
        residuals.append(data["residual_continuum"][:, -1])
        fixed.append((data["observed"]-data["model"])[:, -1])
        continua.append(data["continuum"][:, -1])
        score_masks.append(data["score_mask"][:, -1])
        paths += [OUT / f"{tag}.json", OUT / f"{tag}.npz"]
    residual = np.stack(residuals, axis=1)
    fixed = np.stack(fixed, axis=1)
    continuum = np.stack(continua, axis=1)
    for score in score_masks:
        np.testing.assert_array_equal(score, score_masks[0])
    score = score_masks[0]
    scoring = np.broadcast_to(score[:, None], residual.shape)
    mean = np.mean(residual, axis=1)
    smooth = common_projection(mean, score, 3)
    localized = mean-smooth
    variable = residual-mean[:, None]
    fixed_template = continuum*template[:, None]
    fixed_after = fixed-fixed_template
    # Template construction has no dependency on scoring/development arrays.
    changed = source["residual_continuum"].copy()
    changed[:, -1] += 123.
    np.testing.assert_array_equal(np.mean(changed[:, :-1], axis=1), template)
    rows, order_results = [], {}
    for b, order in enumerate(source["orders"]):
        selection = scoring[b]
        row = dict(order=int(order), **statistics(residual[b][selection]),
                   fixed_flux_rms=float(np.sqrt(np.mean(fixed[b][selection]**2))),
                   after_template_rms=float(np.sqrt(np.mean((residual-template[:, None])[b][selection]**2))))
        rows.append(row)
        order_results[str(order)] = dict(metrics=row,
            decomposition=decomposition(residual[b:b+1], scoring[b:b+1], smooth[b:b+1], localized[b:b+1], variable[b:b+1]),
            template_transfer=template_transfer(residual[b:b+1], template[b:b+1], scoring[b:b+1]))
    pooled = dict(order="all", **statistics(residual[scoring]),
                  fixed_flux_rms=float(np.sqrt(np.mean(fixed[scoring]**2))),
                  after_template_rms=float(np.sqrt(np.mean((residual-template[:, None])[scoring]**2))))
    rows.append(pooled)
    with (OUT / "nonstellar_rms.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_fit=SOURCE,
                  evaluation_exposures=EXPOSURES, mask_path=meta["pixel_mask"]["path"],
                  mask_sha256=meta["pixel_mask"]["sha256"], included_wavelength_pixels=int(allowed.sum()),
                  scoring_pixels_per_exposure=int(score.sum()),
                  primary_definition="r=(observed-model)/fitted continuum; all RMS fractions, multiply by100 for percent",
                  prior_scale_definition="fixed_flux uses observed per-spectrum p95 normalization, not the continuum",
                  pooled=pooled, per_order=order_results,
                  decomposition=decomposition(residual, scoring, smooth, localized, variable),
                  decomposition_degree_sensitivity={},
                  training_template_transfer=template_transfer(residual, template, scoring),
                  fixed_flux_template_transfer=dict(before=statistics(fixed[scoring]), after=statistics(fixed_after[scoring]),
                      scoring_sse_reduction_fraction=float(1-np.sum(fixed_after[scoring]**2)/np.sum(fixed[scoring]**2)),
                      template_mse=float(np.mean(fixed_template[scoring]**2)),
                      remainder_mse=float(np.mean(fixed_after[scoring]**2)),
                      twice_cross_product=float(2*np.mean(fixed_template[scoring]*fixed_after[scoring]))),
                  limitations=["The mask excludes predicted strong stellar absorption and one motion-confirmed blend; remaining stellar mismatch is not proven absent.",
                               "The mask is exploratory and these exposures have been examined previously; this assessment does not retune the physical model or mask.",
                               "Common/localized/smooth/variable are orthogonal statistical descriptions, not physical cause labels.",
                               "The common pattern estimated from four scoring exposures contains finite-sample noise.",
                               "The independent training-template prediction confirms time transfer but cannot identify its physical source.",
                               "Smooth means a cubic projection across an order; this arbitrary scale is checked at degrees1 and5.",
                               "The post-template residual includes noise, time variation, template-estimation error and other mismatches.",
                               "Three orders from one night, without per-pixel uncertainty arrays."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in paths+[OUT / f"{SOURCE}.json", OUT / f"{SOURCE}.npz", Path(__file__)]})
    for degree in [1, 3, 5]:
        trial = common_projection(mean, score, degree)
        report["decomposition_degree_sensitivity"][str(degree)] = decomposition(residual, scoring, trial, mean-trial, variable)
    (OUT / "nonstellar_residual_budget.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT / "nonstellar_residual_budget.npz", orders=source["orders"], exposures=EXPOSURES,
                        wavelength_um=source["wavelength_um"], included_mask=allowed, scoring_mask=score,
                        residual_continuum=residual, residual_fixed_flux=fixed, continuum=continuum,
                        training_template=template, mean_residual=mean, smooth_common=smooth,
                        localized_common=localized, time_variable=variable)
    labels = [str(order) for order in source["orders"]]+["All"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(rows))
    axes[0].bar(x-.18, [100*r["rms"] for r in rows], .36, label="Physical model")
    axes[0].bar(x+.18, [100*r["after_template_rms"] for r in rows], .36, label="After frozen training residual")
    axes[0].set(xticks=x, xticklabels=labels, ylabel="Continuum-relative RMS (%)", xlabel="Order")
    axes[0].legend(fontsize=8)
    budgets = [order_results[str(order)]["decomposition"] for order in source["orders"]]+[report["decomposition"]]
    bottom = np.zeros(len(rows))
    for key, label in [("smooth_common", "Smooth common pattern"), ("localized_common", "Localized common pattern"),
                       ("time_variable", "Time-variable remainder")]:
        values = np.array([100*b["parts"][key]["mse_fraction"] for b in budgets])
        axes[1].bar(x, values, bottom=bottom, label=label); bottom += values
    axes[1].set(xticks=x, xticklabels=labels, ylabel="Share of residual mean square (%)", xlabel="Order", ylim=(0, 125))
    axes[1].legend(fontsize=8, loc="upper center", ncol=1)
    fig.suptitle("Current weak-stellar mask: residual morphology, not physical attribution")
    fig.savefig(OUT / "nonstellar_residual_budget.png", dpi=160); plt.close(fig)
    print(json.dumps(dict(pooled=pooled, decomposition=report["decomposition"],
                          training_template_transfer=report["training_template_transfer"]), indent=2))


if __name__ == "__main__":
    main()
