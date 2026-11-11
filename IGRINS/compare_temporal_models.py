"""Compare saved discovery models after a training-only temporal projection.

The method is declared in temporal_model_comparison_plan.json. No model or
pixel mask is changed, and no reserved-exposure array is opened.
"""
from __future__ import annotations

import csv
import argparse
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
OUT = ROOT/"results"
RESERVED = {10, 30, 50, 70}


def rms(values, mask):
    selected = values[mask]
    return float(100*np.sqrt(np.mean(selected**2))) if len(selected) else None


def load_arrays(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="Use the frozen final comparison plan")
    args = parser.parse_args()
    stem = "temporal_final_comparison" if args.final else "temporal_model_comparison"
    plan_path = OUT/f"{stem}_plan.json"
    plan = json.loads(plan_path.read_text())
    line_masks = load_arrays(OUT/"telluric_mask.npz")
    rows, group_metadata, provenance = [], {}, {}
    for prefix, configuration in plan["groups"].items():
        if isinstance(configuration, str):
            configuration = dict(control=configuration, pattern=f"{prefix}*.json", heldout_exposure=20)
        control_tag = configuration["control"]
        expected_holdout = configuration["heldout_exposure"]
        for tag in configuration.get("required_tags", []):
            if not (OUT/f"{tag}.npz").exists():
                raise ValueError(f"Final output not yet complete: {tag}")
        meta = json.loads((OUT/f"{control_tag}.json").read_text())
        exposures = meta["train_exposures"]+[meta["heldout_exposure"]]
        allowed_final = {expected_holdout} if args.final else set()
        if (RESERVED.intersection(exposures)-allowed_final
                or RESERVED.intersection(meta["train_exposures"])
                or meta["heldout_exposure"] != expected_holdout):
            raise ValueError("Reserved or unexpected control exposure")
        control = load_arrays(OUT/f"{control_tag}.npz")
        denominator = control["continuum"]
        observed = control["observed"]
        observed_norm = observed/denominator
        if np.any(denominator <= 0) or np.any(observed_norm <= 0):
            raise ValueError("Nonpositive baseline continuum or flux")
        training, scoring = control["train_mask"], control["score_mask"]
        np.testing.assert_array_equal(control["exposures"], exposures)
        for e in range(len(exposures)-1):
            np.testing.assert_array_equal(training[:, e], training[:, 0])
        group_metadata[prefix] = dict(control=control_tag, exposures=exposures,
                                      n_training=int(training.sum()), n_scoring=int(scoring.sum()))
        observed_ratio = observed_norm/observed_norm[:, :1]
        for json_path in sorted(OUT.glob(configuration["pattern"])):
            if not json_path.with_suffix(".npz").exists():
                continue
            variant_meta = json.loads(json_path.read_text())
            variant_exposures = variant_meta.get("train_exposures", [])+[variant_meta.get("heldout_exposure")]
            if RESERVED.intersection(variant_exposures)-allowed_final:
                continue
            if variant_exposures != exposures:
                continue
            data = load_arrays(json_path.with_suffix(".npz"))
            for name in ["orders", "exposures", "wavelength_um", "observed", "train_mask", "calibration_mask", "score_mask"]:
                np.testing.assert_array_equal(data[name], control[name])
            residual = (observed-data["model"])/denominator
            training_mean = residual[:, :-1].mean(axis=1)
            centered = residual-training_mean[:, None]
            model_norm = data["model"]/denominator
            if np.any(model_norm <= 0):
                raise ValueError("Nonpositive modeled flux")
            ratio_residual = observed_ratio-model_norm/model_norm[:, :1]
            ratio_training = training.copy()
            ratio_training[:, 0] = False
            for species in ["all", "H2O", "CO2"]:
                subset = np.ones(training.shape[::2], dtype=bool) if species == "all" else line_masks[f"{species}_mask"]
                for b in [None]+list(range(len(control["orders"]))):
                    select = np.broadcast_to(subset[:, None], training.shape).copy()
                    if b is not None:
                        select[np.arange(len(control["orders"])) != b] = False
                    train_mask = training & select
                    score_mask = scoring & select
                    row = dict(group=prefix, tag=json_path.stem,
                               order="all" if b is None else int(control["orders"][b]), subset=species,
                               n_training=int(train_mask.sum()), n_scoring=int(score_mask.sum()),
                               training_fixed_flux_rms_percent=rms(observed-data["model"], train_mask),
                               heldout_fixed_flux_rms_percent=rms(observed-data["model"], score_mask),
                               training_raw_rms_percent=rms(residual, train_mask),
                               training_temporal_rms_percent=rms(centered, train_mask),
                               heldout_raw_rms_percent=rms(residual, score_mask),
                               heldout_training_template_rms_percent=rms(centered, score_mask),
                               training_ratio_rms_percent=rms(ratio_residual, ratio_training & select),
                               heldout_ratio_rms_percent=rms(ratio_residual, score_mask))
                    rows.append(row)
            provenance[json_path.name] = hashlib.sha256(json_path.read_bytes()).hexdigest()
            provenance[json_path.with_suffix(".npz").name] = hashlib.sha256(json_path.with_suffix(".npz").read_bytes()).hexdigest()
        provenance[f"{control_tag}.npz"] = hashlib.sha256((OUT/f"{control_tag}.npz").read_bytes()).hexdigest()
    pooled = []
    if args.final:
        for variant in ["control", "tp3", "ipepoch", "tp3epoch"]:
            for order in ["all"]+list(map(int, control["orders"])):
                for subset in ["all", "H2O", "CO2"]:
                    selected = [row for row in rows if row["tag"].startswith(f"final_{variant}_e")
                                and row["order"] == order and row["subset"] == subset]
                    counts = np.asarray([row["n_scoring"] for row in selected])
                    item = dict(variant=variant, order=order, subset=subset, n_scoring=int(counts.sum()))
                    for key in [key for key in rows[0] if key.startswith("heldout_")]:
                        item[key] = (float(np.sqrt(np.sum([row[key]**2*row["n_scoring"] for row in selected
                                                         if row[key] is not None])/counts.sum()))
                                     if counts.sum() else None)
                    pooled.append(item)
    report = dict(method=plan, groups=group_metadata, statistics=rows, pooled_holdout_statistics=pooled,
                  assertions="Observed values, masks, exposure indices, orders and wavelengths match exactly within each group",
                  notes=["The per-model training mean residual is a learned empirical per-pixel nuisance template",
                         "Holdout scoring data never enter that template; development exposure20 was examined previously",
                         "Subtracting a static template is not evidence that stellar or atmospheric line parameters are correct",
                         "Ratio denominators include reference-frame noise and full modeled spectral structure; no formal independent-noise inference"],
                  source_sha256=provenance,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest())
    (OUT/f"{stem}.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUT/f"{stem}.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    if pooled:
        with (OUT/f"{stem}_pooled.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(pooled[0]))
            writer.writeheader(); writer.writerows(pooled)
    fig, axes = plt.subplots(2 if args.final else 1, 2,
                             figsize=(13, 8 if args.final else 10), constrained_layout=True)
    for ax, group in zip(axes.flat, plan["groups"]):
        selected = [row for row in rows if row["group"] == group and row["order"] == "all" and row["subset"] == "all"]
        positions = np.arange(len(selected))
        ax.plot([row["heldout_raw_rms_percent"] for row in selected], positions, "o", label="Raw residual")
        ax.plot([row["heldout_training_template_rms_percent"] for row in selected], positions, "o", label="Training-mean template removed")
        ax.set_yticks(positions, [row["tag"].removeprefix(group) for row in selected])
        ax.invert_yaxis()
        heldout = group_metadata[group]["exposures"][-1]
        ax.set(xlabel="Scoring RMS (%) / fixed control continuum",
               title=f"Exposure {heldout}; {len(group_metadata[group]['exposures'])-1} training exposures")
        ax.legend(fontsize=8)
        print(json.dumps(selected, indent=2))
    fig.suptitle("Temporal model diagnostic; static templates learned from training exposures only")
    fig.savefig(OUT/f"{stem}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
