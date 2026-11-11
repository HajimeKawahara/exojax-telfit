"""Compare predeclared telluric variants on identical weak-stellar pixels."""
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
LABELS = {
    "focus_base": "Reference profile",
    "focus_shift": "+ pressure shift",
    "focus_ip": "+ pressure shift + IP wings",
    "focus_cool": "+ pressure shift, T - 5 K",
    "focus_warm": "+ pressure shift, T + 5 K",
}


def rms_percent(values, mask):
    """Return an unweighted RMS, with null for empty subsets."""
    return float(100 * np.sqrt(np.mean(values[mask] ** 2))) if mask.any() else None


def load_arrays(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def main():
    plan_path = OUT / "telluric_focus_plan.json"
    plan = json.loads(plan_path.read_text())
    mask_path = OUT / plan["mask"]
    selected = load_arrays(mask_path)
    mask_hash = hashlib.sha256(mask_path.read_bytes()).hexdigest()
    tags = list(plan["variants"])
    required = [OUT / f"{tag}.{suffix}" for tag in tags for suffix in ["json", "npz"]]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("All predeclared variants are required: " + ", ".join(missing))
    records = {
        tag: (json.loads((OUT / f"{tag}.json").read_text()), load_arrays(OUT / f"{tag}.npz"))
        for tag in tags}
    reference_meta, reference = records["focus_base"]
    stellar_parameters = {key: reference_meta["parameters"][key] for key in
                          ["stellar_rv_reference_kms", "stellar_extra_sigma_kms"]}
    np.testing.assert_array_equal(reference["orders"], selected["orders"])
    np.testing.assert_array_equal(reference["wavelength_um"], selected["wavelength_um"])
    rows, global_rows, checks = [], [], []
    for tag, (meta, data) in records.items():
        settings = plan["variants"][tag]
        if (meta["pressure_shift"] != settings["shift"]
                or meta["ip"] != settings["ip"]
                or meta["temperature_offset_K"] != settings["temperature_offset_K"]):
            raise ValueError(f"{tag} settings differ from the declared comparison plan.")
        if meta["pixel_mask"]["sha256"] != mask_hash:
            raise ValueError(f"{tag} used a different mask.")
        if meta["train_exposures"] != plan["train_exposures"] or meta["heldout_exposure"] != plan["fresh_conditional_validation_exposure"]:
            raise ValueError(f"{tag} exposure selection differs from the plan.")
        for key in ["observed", "orders", "exposures", "wavelength_um", "train_mask",
                    "calibration_mask", "score_mask", "included_mask"]:
            np.testing.assert_array_equal(data[key], reference[key], err_msg=f"{tag}: {key}")
        np.testing.assert_array_equal(data["included_mask"], selected["mask"])
        for key, value in stellar_parameters.items():
            if key not in meta["fixed_parameters"] or meta["parameters"][key] != value:
                raise ValueError(f"{tag} did not freeze the same stellar parameters.")
        residual = data["observed"] - data["model"]
        if not np.all(np.isfinite(residual)) or np.any(data["continuum"] <= 0):
            raise ValueError(f"{tag} has invalid model arrays.")
        continuum_residual = residual / data["continuum"]
        reference_continuum_residual = residual / reference["continuum"]
        global_row = dict(tag=tag, label=LABELS[tag], **settings)
        for phase, key in [("train", "train_mask"), ("validation", "score_mask")]:
            phase_mask = data[key]
            for order in ["all", *map(int, data["orders"])]:
                order_mask = np.ones_like(phase_mask)
                if order != "all":
                    order_mask[:] = False
                    order_mask[np.flatnonzero(data["orders"] == order)[0]] = True
                for subset in ["all", "H2O", "CO2"]:
                    line_mask = np.ones_like(selected["mask"]) if subset == "all" else selected[f"{subset}_mask"]
                    active = phase_mask & order_mask & line_mask[:, None, :]
                    row = dict(tag=tag, phase=phase, order=order, subset=subset,
                               n_pixels=int(active.sum()),
                               fixed_scale_rms_percent=rms_percent(residual, active),
                               fitted_continuum_rms_percent=rms_percent(continuum_residual, active),
                               reference_continuum_rms_percent=rms_percent(reference_continuum_residual, active))
                    rows.append(row)
                    if order == "all" and subset == "all":
                        global_row[f"{phase}_n_pixels"] = row["n_pixels"]
                        for metric in ["fixed_scale", "fitted_continuum", "reference_continuum"]:
                            global_row[f"{phase}_{metric}_rms_percent"] = row[f"{metric}_rms_percent"]
            for name, stored in [("fixed_scale", meta[f"{'training' if phase == 'train' else 'heldout'}_fixed_scale_metrics"]),
                                 ("fitted_continuum", meta["training_metrics" if phase == "train" else "heldout_score_metrics"])]:
                np.testing.assert_allclose(global_row[f"{phase}_{name}_rms_percent"], 100 * stored["rms"], rtol=1e-12)
        global_rows.append(global_row)
        checks.append(dict(tag=tag, training=meta["training_fit"],
                           heldout_calibration=meta["heldout_calibration_fit"],
                           bounds_hit=meta["bounds_hit"]))

    with (OUT / "focus_comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    indexed = {row["tag"]: row for row in global_rows}
    comparisons = []
    for tag, comparator in [("focus_shift", "focus_base"), ("focus_ip", "focus_shift"),
                            ("focus_cool", "focus_shift"), ("focus_warm", "focus_shift")]:
        record = dict(tag=tag, compared_to=comparator)
        for phase in ["train", "validation"]:
            key = f"{phase}_fixed_scale_rms_percent"
            record[f"{phase}_rms_change_percentage_points"] = indexed[tag][key] - indexed[comparator][key]
            record[f"{phase}_relative_rms_change_percent"] = 100 * (indexed[tag][key] / indexed[comparator][key] - 1)
        comparisons.append(record)
    sources = [plan_path, mask_path, OUT / "telluric_mask.json", ROOT / "fit_joint.py",
               OUT / "joint_star_profile.json", Path(__file__), *required]
    summary = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), plan=plan,
        primary_metric="Unweighted RMS as percent of identical observed p95 flux scales",
        normalization="Per order/exposure p95 of included training pixels; included calibration pixels only for exposure 20. Identical observed arrays verified across all variants.",
        auxiliary_metrics="Each fitted continuum RMS and one fixed focus_base continuum RMS; the latter avoids denominator changes in line-subset comparisons.",
        mask_sha256=mask_hash, fixed_stellar_parameters=stellar_parameters,
        pixel_count_per_order={str(order): int(mask.sum()) for order, mask in zip(selected["orders"], selected["mask"])},
        global_comparison=global_rows, isolated_changes=comparisons, subset_statistics=rows,
        fit_checks=checks,
        assertions="Identical observed arrays, included masks, train/calibration/score masks, wavelength coordinates, exposures and fixed stellar parameters; saved metrics reproduced.",
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in sources})
    (OUT / "focus_comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    x = np.arange(len(tags))
    for ax, metric, title in zip(axes, ["fixed_scale", "fitted_continuum"],
                               ["Primary: identical observed flux scale", "Auxiliary: each fitted continuum"]):
        for offset, phase, label, color in [(-.18, "train", "Training (exposures 0, 78)", "tab:blue"),
                                          (.18, "validation", "Conditional validation (exposure 20)", "tab:orange")]:
            values = [row[f"{phase}_{metric}_rms_percent"] for row in global_rows]
            bars = ax.bar(x + offset, values, width=.36, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
        ax.set_xticks(x, ["Reference", "+ shift", "+ shift\n+ IP wings", "+ shift\nT - 5 K", "+ shift\nT + 5 K"])
        ax.set(title=title, ylabel="RMS (%)", ylim=(0, ax.get_ylim()[1] * 1.12))
        ax.legend(fontsize=8)
    fig.suptitle("Telluric comparison: fixed 1,477 weak-stellar pixels; fixed stellar parameters")
    fig.savefig(OUT / "focus_comparison.png", dpi=170)
    plt.close(fig)

    report = """WASP-77A / IGRINS: telluric models on fixed weak-stellar pixels

Primary comparison: identical observed p95 flux scales
----------------------------------------------------
Model                                      Train RMS (%)   Validation RMS (%)
"""
    for row in global_rows:
        report += f"{row['label']:42s}{row['train_fixed_scale_rms_percent']:14.5f}{row['validation_fixed_scale_rms_percent']:21.5f}\n"
    report += "\nAuxiliary: residual RMS divided by each fitted continuum (%)\n"
    for row in global_rows:
        report += f"{row['label']:42s}{row['train_fitted_continuum_rms_percent']:14.5f}{row['validation_fitted_continuum_rms_percent']:21.5f}\n"
    report += "\nControlled changes in primary RMS (negative is an improvement)\n"
    for row in comparisons:
        report += (f"{row['tag']} vs {row['compared_to']}: train {row['train_rms_change_percentage_points']:+.5f}, "
                   f"validation {row['validation_rms_change_percentage_points']:+.5f} percentage points\n")
    report += "\nPrimary per-order RMS: train / validation (%)\n"
    for tag in tags:
        values = [row for row in rows if row["tag"] == tag and row["order"] != "all" and row["subset"] == "all"]
        report += LABELS[tag] + ": "
        for order in selected["orders"]:
            pair = [next(row for row in values if row["order"] == order and row["phase"] == phase)
                    for phase in ["train", "validation"]]
            report += f"order {order} {pair[0]['fixed_scale_rms_percent']:.5f}/{pair[1]['fixed_scale_rms_percent']:.5f}; "
        report += "\n"
    report += "\nPrimary molecular line-subset RMS: train / validation (%)\n"
    for tag in tags:
        report += LABELS[tag] + ": "
        for subset in ["H2O", "CO2"]:
            pair = [next(row for row in rows if row["tag"] == tag and row["order"] == "all"
                         and row["subset"] == subset and row["phase"] == phase)
                    for phase in ["train", "validation"]]
            report += (f"{subset} {pair[0]['fixed_scale_rms_percent']:.5f}/{pair[1]['fixed_scale_rms_percent']:.5f} "
                       f"(N={pair[0]['n_pixels']}/{pair[1]['n_pixels']}); ")
        report += "\n"
    report += f"""
Matched data and frozen stellar model
------------------------------------
Orders 13, 34, 40; original detector pixels 600:1500; exposures 0 and 78 train.
Exposure 20 was reserved when the five variants and pixel mask were declared.
Its water amplitude, common calibration RV and separate cubic continua use only
alternating 80-pixel calibration blocks with five-pixel guards. Other guarded
blocks are scored. Shared telluric/IP parameters use the training exposures.
This is conditional prediction after nuisance calibration, not an untouched
exposure prediction. All variants use the exact same observed p95 scaling:
included pixels for training, included calibration pixels only for exposure 20.
Exact equality of saved observed arrays and all three masks is checked here.
Primary RMS therefore compares the same flux values with a fixed denominator.
Continuum RMS is auxiliary because fitted continua vary between models.
The reference_continuum RMS columns use the fixed focus_base continuum.
Train/scoring counts: {global_rows[0]['train_n_pixels']} / {global_rows[0]['validation_n_pixels']}.

Mask SHA256: {mask_hash}
Fixed stellar RV reference: {stellar_parameters['stellar_rv_reference_kms']:.10f} km/s
Fixed stellar extra Gaussian width: {stellar_parameters['stellar_extra_sigma_kms']:.10f} km/s
The existing PHOENIX template, normalization, rotation and stellar parameters
are held fixed. Stellar-line adjustments are outside this experiment.
The fixed mask retains 578/458/441 pixels in orders 13/34/40. It excludes
stellar-only absorption above 2%, known stellar/missing-feature residual
regions with two-pixel padding, and telluric transmission at or below 15%.
H2O/CO2 subsets require >2% marginal absorption in any discovery exposure.
These subsets can overlap; their statistics are descriptive, not disjoint
error budgets. Every per-order/per-species count is in focus_comparison.csv.

Telluric variants and interpretation
-----------------------------------
All models use the same HITRAN cache and 20-layer reanalysis pressure/composition
profile. Reference self broadening is omitted in all five variants.
Pressure shifts apply the cached line delta_air coefficients by layer.
The IP mixture adds one shared wing fraction; its width is twice the core width.
The +/-5 K cases uniformly perturb all atmospheric temperatures while holding
pressure boundaries and molecular column profiles fixed. They are sensitivity
tests, not a measured/retrieved T-P profile. Comparing them to focus_shift
isolates the temperature perturbation from pressure shifts.
No per-pixel errors are supplied: RMS values are not reduced chi-square or
statistical evidence that a small improvement is significant.

Selection limits
----------------
The mask is selected post hoc from discovery exposures 0, 78 and 39, using both
the stellar template and previously identified residual regions. Selection
removes known problematic features; its RMS cannot be compared directly with
the earlier all-pixel fits to claim physical-model improvement. Exposure 20
tests transfer to another exposure of the same night and same wavelengths;
it does not remove the wavelength-selection dependence. Weak model stellar
absorption does not prove that all remaining stellar errors are negligible.
Order 13 has few unabsorbed continuum anchors, so its normalization still
depends on the assumed continuum and telluric model. The comparison covers
three orders, not all 44 orders or all 79 exposures.

Reproduction
------------
Run the five variants recorded in results/telluric_focus_plan.json using the
same results/telluric_mask.npz and frozen stellar parameters. Then run:
python IGRINS/summarize_telluric_focus.py
Summary: focus_comparison.csv/.json/.png. The JSON stores source hashes,
fixed stellar parameters, optimization checks, and the declared plan.
"""
    (OUT / "FOCUS_REPORT.txt").write_text(report)
    print(json.dumps(global_rows, indent=2))


if __name__ == "__main__":
    main()
