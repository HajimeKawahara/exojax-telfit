"""Tabulate individual residual-window signal costs without changing any mask."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from check_wavenumber_mask_signal import build_template_bank, compare_signal, project
from design_wavenumber_masks import make_candidates, residual_statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def costs(signals, specs, original_mask, retained_mask, exposure_slice):
    """Summarize raw-power losses using the original template support."""
    rows = signals[:, :, exposure_slice]
    before = np.sum(np.where(original_mask[None, :, None], rows**2, 0.), axis=(1, 2, 3))
    after = np.sum(np.where(retained_mask[None, :, None], rows**2, 0.), axis=(1, 2, 3))
    before_epoch = np.sum(np.where(original_mask[None, :, None], rows**2, 0.), axis=(1, 3))
    after_epoch = np.sum(np.where(retained_mask[None, :, None], rows**2, 0.), axis=(1, 3))
    result = {}
    for pattern in ["gaussian", "water"]:
        selected = [i for i, s in enumerate(specs) if s["pattern"] == pattern and s["role"] == "moving"]
        ratio = after[selected]/before[selected]
        epoch_ratio = after_epoch[selected]/before_epoch[selected]
        worst = selected[int(np.argmin(ratio))]
        result[pattern] = dict(minimum_total_retention=float(ratio.min()), median_total_retention=float(np.median(ratio)),
            minimum_epoch_retention=float(epoch_ratio.min()), worst_template=specs[worst]["name"],
            per_template={specs[i]["name"]: float(after[i]/before[i]) for i in selected})
    return result


def main():
    data, specs, signals = build_template_bank()
    path = OUT/"wavenumber_mask_candidates.npz"
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path) as source:
        training = source["selection_training_residual"]
    candidates, eligible, power = make_candidates(training, data["wavelength_um"], data["included_mask"])
    assert len(candidates) == len(meta["candidate_windows"])
    selected_ids = set(meta["cases"]["budget02"]["selected_candidate_ids"])
    results, flat = [], []
    for candidate in candidates:
        b, window = candidate["order_index"], candidate["window"]
        exclude = np.zeros_like(data["included_mask"])
        exclude[b] = window
        retained = data["score_mask"] & ~exclude
        row = {k:v for k,v in candidate.items() if k != "window"}
        row.update(order=int(data["orders"][b]), included_in_budget02_selection=candidate["candidate_id"] in selected_ids,
            removed_included=int(np.sum(exclude & data["included_mask"])), removed_score=int(np.sum(exclude & data["score_mask"])),
            training_included_residual_power=float(np.sum(power["total_power"][exclude & data["included_mask"]])),
            training=costs(signals, specs, data["score_mask"], retained, slice(0, 5)),
            evaluation=costs(signals, specs, data["score_mask"], retained, slice(5, None)))
        results.append(row)
        table = {key: row[key] for key in ["candidate_id", "order", "center_index", "bin_nu_lower_cm1", "bin_nu_upper_cm1",
            "included_in_budget02_selection", "removed_included", "removed_score", "training_included_residual_power"]}
        for split in ["training", "evaluation"]:
            for pattern in ["gaussian", "water"]:
                for key in ["minimum_total_retention", "minimum_epoch_retention"]:
                    table[f"{split}_{pattern}_{key}"] = row[split][pattern][key]
        flat.append(table)
    # This interval was already selected by training-only residual ranking.
    hotspot = next(r for r in meta["cases"]["budget01"]["intervals"] if r["order"] == 34)
    exclusion = np.zeros_like(data["included_mask"])
    exclusion[hotspot["order_index"], hotspot["start_index"]:hotspot["stop_index_exclusive"]] = True
    keep_included, keep_score = data["included_mask"] & ~exclusion, data["score_mask"] & ~exclusion
    old = np.array([project(s, data["wavelength_um"], data["included_mask"]) for s in signals])
    new = np.array([project(s, data["wavelength_um"], keep_included) for s in signals])
    single_cases = []
    for spec, raw, previous, changed in zip(specs, signals, old, new):
        single_cases.append(dict(spec=spec,
            evaluation_score=compare_signal(raw[:, 5:], previous[:, 5:], changed[:, 5:], data["score_mask"], keep_score),
            evaluation_included=compare_signal(raw[:, 5:], previous[:, 5:], changed[:, 5:], data["included_mask"], keep_included)))
    single = dict(name="single_order34_hotspot", interval=hotspot,
        raw_information_cost_training=costs(signals, specs, data["score_mask"], keep_score, slice(0, 5)),
        raw_information_cost_evaluation=costs(signals, specs, data["score_mask"], keep_score, slice(5, None)),
        removed_included=int(np.sum(exclusion & data["included_mask"])), removed_score=int(np.sum(exclusion & data["score_mask"])),
        original_q5_retained_residual=residual_statistics(data["residual_filtered"][:, 5:], keep_score),
        reprojected_retained_residual=residual_statistics(project(data["residual_unfiltered"], data["wavelength_um"], keep_included)[:, 5:], keep_score),
        signal_cases=single_cases,
        caveat="The finite water-pattern bank contains no lines in order34; exact water retention here does not establish preservation for actual planetary water or other molecules.")
    sources = [Path(__file__), ROOT/"check_wavenumber_mask_signal.py", ROOT/"design_wavenumber_masks.py", path, path.with_suffix(".json"),
        OUT/"adopted_telluric_fit.npz", OUT/"line_shape_fit.json", OUT/"quadratic_strength_fit.npz", ROOT/"data/20201214_info.csv"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), candidate_count=len(results), candidates=results,
        single_hotspot=single, method="Individual window costs are not additive when windows overlap; retained powers use the original unmasked score support",
        limitations=["All candidate windows come from the existing five-training-exposure residual ranking.",
            "Information costs are conditional equal-independent-noise template-power fractions, not measured detection significance.",
            "Evaluation synthetic-template costs describe forecast geometry only; no evaluation residual changes the masks.",
            "Single-hotspot residual changes select data and reproject Q5; no physical model parameters are refitted here."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_mask_interval_signal.json").write_text(json.dumps(report, indent=2)+"\n")
    with (OUT/"wavenumber_mask_interval_signal.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
    np.savez_compressed(OUT/"wavenumber_single_hotspot_mask.npz", case_names=np.array([single["name"]]),
        exclusion_masks=exclusion[None], included_masks=keep_included[None], score_masks=keep_score[None],
        included_mask=data["included_mask"], score_mask=data["score_mask"],
        wavelength_um=data["wavelength_um"], wavenumber_cm1=1e4/data["wavelength_um"], orders=data["orders"], exposures=data["exposures"])
    print("Individual windows", len(results), "single hotspot removed", single["removed_included"], single["removed_score"],
          "original-Q5 retained RMS", single["original_q5_retained_residual"]["rms_percent"], flush=True)
    for pattern, summary in single["raw_information_cost_evaluation"].items():
        print(pattern, summary["minimum_total_retention"], summary["minimum_epoch_retention"], flush=True)


if __name__ == "__main__":
    main()
