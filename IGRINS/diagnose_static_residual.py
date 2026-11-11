"""Freeze a training-only residual template and measure its time transfer.

This empirical diagnostic does not identify stellar or telluric line errors.
No physical, calibration, or continuum parameters are refitted after adding
the fixed residual template to a saved physical spectrum.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
from scipy.ndimage import gaussian_filter1d

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
METHODS = ["mean", "mean_sigma1"]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_fit(tag):
    meta = json.loads((OUT / f"{tag}.json").read_text())
    with np.load(OUT / f"{tag}.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    return meta, data


def build_template(source, path, plan):
    """Use only the five explicitly listed training exposures, with no clipping."""
    if source != plan["source_fit"]:
        raise ValueError("Template source must match the predeclared physical fit.")
    meta, data = read_fit(source)
    required = plan["template_exposures"]
    if meta["train_exposures"] != required:
        raise ValueError("Template source must use exactly the predeclared five training exposures.")
    indices = [int(np.flatnonzero(data["exposures"] == exposure).item()) for exposure in required]
    assert meta["heldout_exposure"] not in required
    allowed = data["included_mask"]
    np.testing.assert_array_equal(data["train_mask"][:, indices],
                                   np.broadcast_to(allowed[:, None], (len(allowed), len(indices), allowed.shape[1])))
    residual = (data["observed"][:, indices]-data["model"][:, indices]) / data["continuum"][:, indices]
    mean = np.where(allowed, np.mean(residual, axis=1), 0.)
    numerator = gaussian_filter1d(mean, 1., axis=-1, mode="constant", cval=0., truncate=3.)
    denominator = gaussian_filter1d(allowed.astype(float), 1., axis=-1,
                                    mode="constant", cval=0., truncate=3.)
    smoothed = np.where(allowed, numerator/np.maximum(denominator, 1e-30), 0.)
    # An explicit perturbation test verifies that excluded wavelengths and the
    # development exposure cannot affect the template-construction expression.
    changed = residual + (~allowed[:, None])*123.
    check = np.where(allowed, np.mean(changed, axis=1), 0.)
    np.testing.assert_array_equal(check, mean)
    np.savez_compressed(path, orders=data["orders"], wavelength_um=data["wavelength_um"],
                        included_mask=allowed, mean=mean, mean_sigma1=smoothed,
                        template_exposures=required)
    metadata = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_fit=source,
                    source_json_sha256=file_hash(OUT / f"{source}.json"),
                    source_npz_sha256=file_hash(OUT / f"{source}.npz"),
                    template_sha256=file_hash(path), template_exposures=required,
                    heldout_exposure_not_used=meta["heldout_exposure"], methods=plan["methods"],
                    source_mask_sha256=meta["pixel_mask"]["sha256"],
                    source_code_sha256=file_hash(__file__),
                    interpretation=plan["interpretation"], excluded_pixel_perturbation_max_change=0.)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2)+"\n")


def statistics(residual, mask):
    value = residual[mask]
    return dict(n_pixels=int(len(value)), rms=float(np.sqrt(np.mean(value**2))),
                bias=float(np.mean(value)), mad=float(1.4826*np.median(np.abs(value-np.median(value)))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="inv5_control")
    parser.add_argument("--targets", nargs="+")
    parser.add_argument("--template", type=Path, default=OUT / "static_residual_template.npz")
    parser.add_argument("--build-template", action="store_true")
    parser.add_argument("--allow-reserved", action="store_true",
                        help="Apply the already frozen template to reserved exposures after final settings are frozen")
    parser.add_argument("--output", default="static_residual_comparison")
    args = parser.parse_args()
    plan_path = OUT / "static_residual_plan.json"
    plan = json.loads(plan_path.read_text())
    if args.build_template:
        if args.template.exists():
            raise FileExistsError("Frozen template already exists; choose a new path for a distinct diagnostic.")
        build_template(args.source, args.template, plan)
    with np.load(args.template) as saved:
        template = {key: saved[key] for key in saved.files}
    template_meta = json.loads(args.template.with_suffix(".json").read_text())
    assert file_hash(args.template) == template_meta["template_sha256"]
    targets = args.targets or [args.source]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  plan_sha256=file_hash(plan_path), template_sha256=file_hash(args.template),
                  template_source=template_meta, methods=plan["methods"],
                  interpretation=plan["interpretation"], limitations=plan["limitations"], targets={})
    bar_labels, bar_values = [], []
    for tag in targets:
        meta = json.loads((OUT / f"{tag}.json").read_text())
        if meta["heldout_exposure"] in plan["reserved_exposures"] and not args.allow_reserved:
            raise ValueError("Reserved exposure requires an explicit final-evaluation invocation.")
        if set(meta["train_exposures"]) & set(plan["reserved_exposures"]):
            raise ValueError("A reserved exposure appears in the target training set.")
        _, data = read_fit(tag)
        for key in ("orders", "wavelength_um", "included_mask"):
            np.testing.assert_array_equal(template[key], data[key])
        e = len(data["exposures"])-1
        mask = data["score_mask"][:, e]
        baseline = data["observed"][:, e]-data["model"][:, e]
        continuum = data["continuum"][:, e]
        row = dict(heldout_exposure=int(data["exposures"][e]), source_json_sha256=file_hash(OUT / f"{tag}.json"),
                   source_npz_sha256=file_hash(OUT / f"{tag}.npz"), methods={})
        residuals = {"physical_model": baseline}
        for method in METHODS:
            residuals[method] = baseline - continuum*template[method]
        for method, residual in residuals.items():
            fixed = statistics(residual, mask)
            normalized = statistics(residual/continuum, mask)
            row["methods"][method] = dict(fixed_scale=fixed, continuum_normalized=normalized,
                per_order=[dict(order=int(order), fixed_scale=statistics(residual[b], mask[b]),
                                continuum_normalized=statistics(residual[b]/continuum[b], mask[b]))
                           for b, order in enumerate(data["orders"])])
            if method != "physical_model":
                row["methods"][method]["fractional_scoring_sse_reduction"] = float(1-np.sum(residual[mask]**2)/np.sum(baseline[mask]**2))
        report["targets"][tag] = row
        bar_labels.append(f"{tag}\nExposure {int(data['exposures'][e])}")
        bar_values.append([100*row["methods"][method]["fixed_scale"]["rms"]
                           for method in ["physical_model"]+METHODS])
        print(tag, {method: 100*values["fixed_scale"]["rms"] for method, values in row["methods"].items()}, flush=True)
    (OUT / f"{args.output}.json").write_text(json.dumps(report, indent=2)+"\n")
    x = np.arange(len(bar_labels))
    values = np.asarray(bar_values)
    fig, ax = plt.subplots(figsize=(max(7, 2*len(bar_labels)), 4.5), constrained_layout=True)
    for k, (method, label) in enumerate(zip(["physical_model"]+METHODS,
                                           ["Physical model", "+ fixed mean residual", "+ fixed smoothed residual"])):
        ax.bar(x+(k-1)*.24, values[:, k], width=.24, label=label)
    ax.set_xticks(x, bar_labels)
    ax.set_ylabel("Scoring RMS, fixed observed flux scale (%)")
    ax.set_title("Training-only static residual: empirical time-transfer diagnostic")
    ax.legend(fontsize=8)
    fig.savefig(OUT / f"{args.output}.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
