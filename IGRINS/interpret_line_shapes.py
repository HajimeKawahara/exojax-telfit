"""Describe saved line-shape corrections and their associated residual regions."""
from __future__ import annotations

import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C = 299792.458
CASES = ["strength_control", "width_prior3", "center_prior3", "joint_prior3"]


def main():
    source = OUT / "line_shape_fit.json"
    meta = json.loads(source.read_text())
    data = np.load(OUT / "line_shape_fit.npz")
    metadata_path = OUT / "pressure_line_shape_metadata.json"
    metadata = json.loads(metadata_path.read_text())["records"]
    lookup = {(r["order"], r["raw_index"]): r for r in metadata}
    shape = meta["shape_records"]
    baseline = data["residual_strength_control"]
    summary, records = {}, []
    for case in CASES[1:]:
        theta = data[f"theta_{case}"]
        width = np.array([theta[r["width_parameter_index"]] for r in shape])
        center = np.array([theta[r["center_parameter_index"]] for r in shape])
        wp = np.array([r["log_width_prior"] for r in shape])
        cp = np.array([r["center_prior_kms"] for r in shape])
        summary[case] = dict(
            width_factor_min=float(np.exp(width).min()), width_factor_max=float(np.exp(width).max()),
            median_absolute_width_fraction=float(np.median(abs(np.expm1(width)))),
            center_min_kms=float(center.min()), center_max_kms=float(center.max()),
            median_absolute_center_kms=float(np.median(abs(center))),
            width_at_lower_bound=int(np.sum(abs(width + np.log(2)) < 1e-7)),
            width_at_upper_bound=int(np.sum(abs(width - np.log(2)) < 1e-7)),
            center_at_bound=int(np.sum(abs(center) > 1 - 1e-7)),
            width_exceeding_original_log_bin_scale={str(k):int(np.sum(abs(width) / wp > k)) for k in [1, 3, 10]},
            width_fraction_exceeding_original_bin_upper={str(k):int(np.sum(abs(np.expm1(width)) / np.expm1(wp) > k)) for k in [1, 3, 10]},
            center_exceeding_original_bin_upper={str(k):int(np.sum(abs(center) / cp > k)) for k in [1, 3, 10]},
            metrics=meta["cases"][case]["metrics"])

    for i, line in enumerate(shape):
        original = lookup[line["order"], line["raw_index"]]
        row = dict(line_index=i, order=line["order"], raw_index=line["raw_index"],
            wavelength_um=1e4 / line["nu"], nu_cm1=line["nu"],
            original_contrast=line["original_contrast"], ierr=line["ierr"],
            original_gamma_air_cm1_per_atm=original["gamma_air_cm1_per_atm"],
            original_delta_air_cm1_per_atm=original["delta_air_cm1_per_atm"],
            width_relative_bin_upper=original["gamma_relative_upper"],
            center_bin_upper_kms=line["center_prior_kms"], corrections={})
        for case in CASES[1:]:
            theta = data[f"theta_{case}"]
            width, center = theta[[line["width_parameter_index"], line["center_parameter_index"]]]
            row["corrections"][case] = dict(width_factor=float(np.exp(width)),
                gamma_air_cm1_per_atm=float(np.exp(width) * original["gamma_air_cm1_per_atm"]),
                width_log_in_original_bin_scales=float(width / line["log_width_prior"]),
                width_log_in_fitted_prior_scales=float(width / (3 * line["log_width_prior"])),
                center_kms=float(center), center_delta_nu_cm1=float(line["nu"] * center / C),
                center_in_original_bin_upper=float(center / line["center_prior_kms"]),
                center_in_fitted_prior_scales=float(center / (3 * line["center_prior_kms"])),
                width_at_bound=bool(abs(abs(width) - np.log(2)) < 1e-7))
        neighbors = []
        for other in metadata:
            if other["order"] != line["order"] or (other["species"] == "H2O" and other["raw_index"] == line["raw_index"]):
                continue
            separation = C * abs(np.log(other["nu_cm1"] / line["nu"]))
            if separation < 10 and other["baseline_contrast"] >= .005:
                neighbors.append(dict(species=other["species"], raw_index=other["raw_index"],
                    separation_kms=float(separation), baseline_contrast=other["baseline_contrast"],
                    also_shape_fitted=other["selected_h2o_shape"]))
        row["nearby_metadata_lines_with_contrast_at_least_half_percent"] = sorted(neighbors, key=lambda x: x["separation_kms"])
        records.append(row)

    def region_stats(b, pixels):
        result = {}
        for scope, es, mask in [("training", slice(0, 5), data["included_mask"][b]),
                                ("evaluation_score", slice(5, None), data["score_mask"][b])]:
            selected = pixels & mask
            count = int(selected.sum()) * (5 if scope == "training" else 4)
            cases = {}
            if count:
                original_sse = float(np.sum(baseline[b, es][:, selected]**2))
                for case in CASES:
                    residual = data[f"residual_{case}"][b, es][:, selected]
                    sse = float(np.sum(residual**2))
                    cases[case] = dict(rms_percent=float(100 * np.sqrt(sse / count)), sse=sse,
                        sse_reduction=float(original_sse - sse),
                        relative_sse_reduction_percent=float(100 * (1 - sse / original_sse)))
            result[scope] = dict(n_samples=count, cases=cases)
        return result

    # Assign each scored pixel to its nearest selected center within 12 km/s.
    # This avoids duplicate accounting; it is not a causal line ablation.
    for b, order in enumerate(data["orders"]):
        selected = [i for i, r in enumerate(shape) if r["order"] == order]
        if not selected:
            continue
        nu = np.array([shape[i]["nu"] for i in selected])
        distance = C * abs(np.log((1e4 / data["wavelength_um"][b])[:, None] / nu[None]))
        nearest = distance.argmin(axis=1)
        for j, i in enumerate(selected):
            pixels = (nearest == j) & (distance[:, j] <= 12)
            records[i]["nearest_center_neighborhood"] = region_stats(b, pixels)

    old_regions = []
    regions_path = OUT / "nonstellar_residual_regions.csv"
    with regions_path.open() as handle:
        for original in csv.DictReader(handle):
            order = int(original["order"])
            b = int(np.flatnonzero(data["orders"] == order)[0])
            low, high = float(original["wavelength_min_um"]), float(original["wavelength_max_um"])
            pixels = (data["wavelength_um"][b] >= low - 1e-12) & (data["wavelength_um"][b] <= high + 1e-12)
            old_regions.append(dict(original_rank=int(original["rank"]), order=order,
                peak_wavelength_um=float(original["peak_wavelength_um"]),
                wavelength_bounds_um=[low, high], original_category=original["peak_category"],
                metrics=region_stats(b, pixels)))
    scored_lines = [r for r in records if r["nearest_center_neighborhood"]["evaluation_score"]["n_samples"]]
    scored_lines.sort(key=lambda r: r["nearest_center_neighborhood"]["evaluation_score"]["cases"]["joint_prior3"]["sse_reduction"], reverse=True)
    poly5_path = OUT / "poly5_spectral_budget.json"
    poly5_regions = []
    for original in json.loads(poly5_path.read_text())["region_sets"]["post_filter_training_selection"]:
        b = original["order_index"]
        pixels = np.zeros(len(data["wavelength_um"][b]), dtype=bool)
        pixels[slice(*original["array_interval"])] = True
        poly5_regions.append(dict(order=int(data["orders"][b]), peak_wavelength_um=original["peak_wavelength_um"],
                                  metrics=region_stats(b, pixels)))
    score = np.broadcast_to(data["score_mask"][:, None], baseline[:, 5:].shape)
    total_gain = float(np.sum(baseline[:, 5:][score]**2)
                       - np.sum(data["residual_joint_prior3"][:, 5:][score]**2))
    for row in scored_lines:
        stats = row["nearest_center_neighborhood"]["evaluation_score"]["cases"]["joint_prior3"]
        stats["fraction_of_total_model_sse_improvement"] = stats["sse_reduction"] / total_gain
    sources = [source, OUT / "line_shape_fit.npz", metadata_path, regions_path,
               poly5_path, OUT / "adopted_telluric_model.json", OUT / "adopted_telluric_fit.npz", Path(__file__)]
    caveats = [
        "The corrections are conditional effective line-shape adjustments, not measurements of HITRAN database errors.",
        "HITRAN interval upper edges are regularization scales here, not calibrated Gaussian sigmas; joint_prior3 widens those chosen scales by three.",
        "An effective center change can absorb rest-wavenumber error, pressure-shift error, local wavelength distortion, unresolved blends, or asymmetric response errors.",
        "An effective width change can absorb broadening coefficients, temperature exponents, vertical atmospheric weighting, unresolved blending, or fixed-IP mismatch.",
        "All 269 intensity corrections are refitted in each case; width-only/center-only differences are conditional comparisons, not additive error-budget terms.",
        "Nearby-line lists cover only the existing 269 metadata records above 0.5% contrast; absence from that list does not establish isolation from weaker or stellar lines.",
        "Local SSE gains are associations in the saved full-model spectra, not leave-one-line-out causal contributions. Q5 projection also couples distant pixels.",
        "Only H2O lines with original contrast above 2% were shape-fitted: six in order 13 and 29 in order 40. No shape change was allowed in the CO2-rich order 34.",
        "Configuration selection used leave-one-training-exposure-out predictions; descriptive evaluation-region rankings do not change that selection.",
        "Two widths reach the factor-0.5 bound, so those values should not be interpreted as precisely determined physical coefficients."]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case="joint_prior3",
        summary=summary, line_corrections=records,
        neighborhoods_ranked_by_evaluation_sse_improvement=[r["raw_index"] for r in scored_lines],
        original_residual_regions=old_regions, post_poly5_training_selected_regions=poly5_regions,
        selected_n_air_uncertainty_codes=dict(Counter(r["ierr"][4] for r in shape)),
        selected_delta_air_uncertainty_codes=dict(Counter(r["ierr"][5] for r in shape)),
        total_evaluation_sse_improvement=total_gain, caveats=caveats,
        center_sign="Positive parameter increases wavenumber and decreases wavelength; it is not the usual positive receding radial velocity.",
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT / "line_shape_interpretation.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["Interpretation of saved H2O line-shape experiments", "",
             "35 H2O lines; original IP and adopted weak-line intensity priors retained.", ""]
    for case, values in summary.items():
        lines.append(f"{case}: evaluation RMS {values['metrics']['evaluation_score']['rms_percent']:.8f}%; width factors {values['width_factor_min']:.4f} to {values['width_factor_max']:.4f}, median absolute change {100 * values['median_absolute_width_fraction']:.2f}%; center offsets {values['center_min_kms']:.4f} to {values['center_max_kms']:.4f} km/s.")
    values = summary["joint_prior3"]
    lines += ["", f"Joint corrections exceed the original width-bin upper edge in {values['width_fraction_exceeding_original_bin_upper']['1']}/35 lines, and the center-bin upper edge in {values['center_exceeding_original_bin_upper']['1']}/35 lines. These are not sigma significances.",
              "Bound widths: 1.49559184 and 1.49555388 um (factor 0.5); both nearest-center neighborhoods have zero final score pixels. They are 7.61 km/s apart; the second has a 0.75%-contrast H2O neighbor at 0.324 km/s."]
    lines += ["", "Largest associated gains in non-overlapping selected-line neighborhoods (evaluation score mask):"]
    for row in scored_lines[:8]:
        region = row["nearest_center_neighborhood"]["evaluation_score"]
        old = region["cases"]["strength_control"]
        new = region["cases"]["joint_prior3"]
        fit = row["corrections"]["joint_prior3"]
        lines.append(f"Order {row['order']}, {row['wavelength_um']:.8f} um (raw {row['raw_index']}): RMS {old['rms_percent']:.4f}% -> {new['rms_percent']:.4f}%, SSE change {new['relative_sse_reduction_percent']:.1f}% ({100 * new['fraction_of_total_model_sse_improvement']:.1f}% of total model gain); width x{fit['width_factor']:.3f}, center {fit['center_kms']:+.4f} km/s; {len(row['nearby_metadata_lines_with_contrast_at_least_half_percent'])} nearby metadata lines.")
    lines += ["", "Original nonstellar hotspot intervals with evaluation score pixels:"]
    for row in old_regions:
        region = row["metrics"]["evaluation_score"]
        if not region["n_samples"]:
            continue
        old = region["cases"]["strength_control"]
        new = region["cases"]["joint_prior3"]
        lines.append(f"Order {row['order']}, peak {row['peak_wavelength_um']:.8f} um ({row['original_category']}): {old['rms_percent']:.4f}% -> {new['rms_percent']:.4f}%, SSE change {new['relative_sse_reduction_percent']:.1f}%.")
    lines += ["", "Post-Q5 training-selected hotspot intervals, same final score mask:"]
    for row in poly5_regions:
        region = row["metrics"]["evaluation_score"]
        if region["n_samples"]:
            old = region["cases"]["strength_control"]
            new = region["cases"]["joint_prior3"]
            lines.append(f"Order {row['order']}, peak {row['peak_wavelength_um']:.8f} um: {old['rms_percent']:.4f}% -> {new['rms_percent']:.4f}%, SSE change {new['relative_sse_reduction_percent']:.1f}%.")
    lines += ["", "Interpretation limits:"] + caveats + [""]
    (OUT / "LINE_SHAPE_INTERPRETATION.txt").write_text("\n".join(lines))
    print("Saved", len(records), "line records and", len(old_regions), "original hotspot intervals")


if __name__ == "__main__":
    main()
