"""Regenerate the Issue figures from the exact current provisional model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_blase_stellar import Problem
from fit_line_strength import metrics
import report_blase_stellar as plotting

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def decomposition(residual, mask):
    rows = residual[:, 5:]
    keep = np.broadcast_to(mask[:, None], rows.shape)
    mean = rows.mean(axis=1)
    total = float(np.mean(rows[keep]**2))
    common = float(np.mean(mean[mask]**2))
    varying = float(np.mean((rows-mean[:, None])[keep]**2))
    np.testing.assert_allclose(common+varying, total, rtol=1e-12)
    return dict(total_rms_percent=100*np.sqrt(total), common_rms_percent=100*np.sqrt(common),
                varying_rms_percent=100*np.sqrt(varying), common_squared_fraction=common/total)


def main():
    manifest = json.loads((OUT/"adopted_telluric_model.json").read_text())
    product = ROOT/manifest["product"]
    assert hashlib.sha256(product.read_bytes()).hexdigest() == manifest["product_sha256"]
    with np.load(product) as saved:
        current = {key: saved[key] for key in saved.files}
    with np.load(OUT/"blase_stellar_fit.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    experiment = json.loads((OUT/"blase_stellar_fit.json").read_text())
    problem = Problem()
    stage = problem.predict(current["q"], problem.adopted_z)
    stage_metrics, stage_residual = metrics(problem.context, stage)
    np.savez_compressed(OUT/"current_blase_stellar_stage.npz", component=stage,
                        residual_filtered=stage_residual, q=current["q"], z=problem.adopted_z)
    data.update(component_refit=current["component"], residual_refit=current["residual_filtered"],
                stellar_only_corrected=current["stellar_only"], component_stellar=stage,
                residual_stellar=stage_residual)
    experiment["metrics"]["refit"] = manifest["metrics"]
    experiment["metrics"]["stellar"] = stage_metrics
    directory = OUT/"current_blase_figures"
    directory.mkdir(exist_ok=True)
    plotting.OUT = directory
    plotting.CASES = ("adopted", "refit")
    plotting.LABELS.update(adopted="Previous fixed PHOENIX", refit="Current provisional model")
    adjusted = {case: plotting.corrected_model(data, case) for case in plotting.CASES}
    plotting.plots(data, experiment, adjusted)
    for source, target in [("blase_stellar_fit.png", "current_blase_spectral_fit.png"),
                           ("blase_stellar_4820.png", "current_blase_4820.png"),
                           ("blase_stellar_overshoot.png", "current_blase_overshoot.png")]:
        (directory/source).replace(OUT/target)
    directory.rmdir()

    # Replaces the historical residual plot with the currently selected arrays.
    fig, axes = plt.subplots(2, 3, figsize=(17, 8))
    residual, old = current["residual_filtered"], data["residual_adopted"]
    windows = []
    for b, order in enumerate(current["orders"]):
        nu = 1e4/current["wavelength_um"][b]
        sort = np.argsort(nu)
        included, score = current["included_mask"][b], current["score_mask"][b]
        mean = residual[b, 5:].mean(axis=0)
        peak = np.flatnonzero(score)[np.argmax(np.abs(mean[score]))]
        window = (float(nu[peak]-.6), float(nu[peak]+.6))
        windows.append(dict(order=int(order), center_cm1=float(nu[peak]), window=window))
        for row, ax in enumerate(axes[:, b]):
            for e in range(5, 9):
                ax.plot(nu[sort], 100*np.where(included, residual[b, e], np.nan)[sort],
                        lw=.65, alpha=.55, label=f"Frame {current['exposures'][e]}")
            ax.plot(nu[sort], 100*np.where(included, mean, np.nan)[sort], color="k", lw=1.1,
                    label="Current four-frame mean")
            ax.plot(nu[sort], 100*np.where(included, old[b, 5:].mean(axis=0), np.nan)[sort],
                    color=".5", ls="--", lw=.9, label="Previous PHOENIX mean")
            plotting.shade_mask(ax, nu, included)
            ax.axhline(0, color=".4", lw=.6)
            ax.grid(alpha=.12)
            ax.set(xlim=window if row else (nu.min(), nu.max()),
                   xlabel=r"Wavenumber (cm$^{-1}$)", ylabel="Residual (%)")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        axes[0, b].set_title(f"Order {order}: current score RMS "
                             f"{manifest['metrics']['evaluation_score']['per_order'][str(order)]:.3f}%")
        axes[1, b].set_title(f"Largest remaining mean score residual: {nu[peak]:.3f} cm$^{{-1}}$")
    axes[0, 0].legend(fontsize=7, ncol=2)
    fig.suptitle("Residuals of the current provisional model\n"
                 "Original masks retained. Zooms locate remaining structure for display; they do not define new masks.")
    fig.tight_layout(rect=(0, 0, 1, .92))
    fig.savefig(OUT/"current_blase_residuals.png", dpi=155)
    plt.close(fig)

    decomposition_current = decomposition(residual, current["score_mask"])
    decomposition_old = decomposition(old, current["score_mask"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(4)
    cases = [("Previous PHOENIX", experiment["metrics"]["adopted"], "tab:blue"),
             ("Current star; previous telluric strengths", stage_metrics, "tab:orange"),
             ("Current provisional model", manifest["metrics"], "tab:green")]
    for i, (label, entry, color) in enumerate(cases):
        score = entry["evaluation_score"]
        values = [score["per_order"][str(order)] for order in current["orders"]]+[score["rms_percent"]]
        axes[0].bar(x+(i-1)*.25, values, .25, label=label, color=color)
    axes[0].set(xticks=x, xticklabels=["Order13", "Order34", "Order40", "Pooled"],
                ylabel="Evaluation score RMS (%)", title="Matched comparison on unchanged score pixels")
    axes[0].legend(fontsize=7)
    parts = [decomposition_old, decomposition_current]
    common = np.array([p["common_rms_percent"]**2 for p in parts])
    varying = np.array([p["varying_rms_percent"]**2 for p in parts])
    axes[1].bar([0, 1], common, label="Four-frame common component", color="tab:purple")
    axes[1].bar([0, 1], varying, bottom=common, label="Frame-to-frame deviations", color="tab:gray")
    axes[1].set(xticks=[0, 1], xticklabels=["Previous PHOENIX", "Current provisional"],
                ylabel="Mean squared residual (% squared)", title="Statistical decomposition; not a causal noise budget")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis="y", alpha=.15)
    fig.tight_layout()
    fig.savefig(OUT/"current_blase_comparison.png", dpi=165)
    plt.close(fig)
    record = dict(current_product=manifest["product"], current_product_sha256=manifest["product_sha256"],
                  stellar_only_stage_metrics=stage_metrics, current_decomposition=decomposition_current,
                  previous_decomposition=decomposition_old, residual_zoom_windows=windows,
                  figures={path.name: dict(size=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            git_blob_sha1=hashlib.sha1(b"blob "+str(path.stat().st_size).encode()+b"\0"+path.read_bytes()).hexdigest())
                           for path in sorted(OUT.glob("current_blase_*.png"))},
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), ROOT/"report_blase_stellar.py", product,
                                              OUT/"adopted_telluric_model.json", OUT/"blase_stellar_fit.npz"]})
    (OUT/"current_blase_figures.json").write_text(json.dumps(record, indent=2)+"\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
