"""Independently audit training-selected wavelength masks and masked refits."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def project(values, wave, mask):
    """Use monomials and least squares instead of the production Chebyshev QR."""
    result = np.empty_like(values)
    for b, w in enumerate(wave):
        log_nu = np.log(1e4/w)
        x = 2*(log_nu-log_nu.min())/np.ptp(log_nu)-1
        design = np.vander(x, 6, increasing=True)
        coefficient = np.linalg.lstsq(design[mask[b]], values[b][:, mask[b]].T, rcond=None)[0]
        result[b] = values[b]-(design @ coefficient).T
    return result


def statistic(residual, mask, epochs):
    rows = residual[:, epochs]
    keep = np.broadcast_to(mask[:, None], rows.shape)
    if not keep.any():
        return dict(n_samples=0, sse=0., rms_percent=None,
                    common_rms_percent=None, temporal_rms_percent=None)
    common = np.broadcast_to(rows.mean(axis=1)[:, None], rows.shape)
    return dict(n_samples=int(keep.sum()), sse=float(np.sum(rows[keep]**2)),
        rms_percent=float(100*np.sqrt(np.mean(rows[keep]**2))),
        common_rms_percent=float(100*np.sqrt(np.mean(common[keep]**2))),
        temporal_rms_percent=float(100*np.sqrt(np.mean((rows-common)[keep]**2))))


def main():
    report_path = OUT/"wavenumber_mask_candidates.json"
    report = json.loads(report_path.read_text())
    with np.load(OUT/"wavenumber_mask_candidates.npz") as saved:
        a = {key: saved[key] for key in saved.files}
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        original = {key: saved[key] for key in saved.files}
    hashes_checked = []
    for key, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected, key
        hashes_checked.append(key)
    assert hashlib.sha256((OUT/"wavenumber_mask_candidates.npz").read_bytes()).hexdigest() == report["mask_npz_sha256"]
    for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask"]:
        np.testing.assert_array_equal(a[key], original[key])
    wave, included, score = [a[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    np.testing.assert_allclose(a["wavenumber_cm1"], 1e4/wave, atol=0, rtol=0)
    old = project(original["y_over_c"]-original["component"], wave, included)
    np.testing.assert_allclose(old, original["residual_filtered"], atol=1e-11, rtol=0)
    with np.load(OUT/"line_shape_cv_predictions.npz") as saved:
        held = np.stack([saved[f"component_strength_control_fold{e}"][:, e] for e in range(5)], axis=1)
    np.testing.assert_array_equal(held, a["selection_heldout_components"])
    training = project(original["y_over_c"][:, :5]-held, wave, included)
    np.testing.assert_allclose(training, a["selection_training_residual"], atol=1e-11, rtol=0)
    power = np.mean(training**2, axis=1)
    mean = training.mean(axis=1)
    temporal = np.mean((training-mean[:, None])**2, axis=1)
    eligible = included & (np.sqrt(power) >= .01) & ((abs(training) >= .01).sum(axis=1) >= 3)
    np.testing.assert_array_equal(eligible, a["eligible_centers"])
    for key, value in [("selection_total_power", power), ("selection_training_mean", mean),
                       ("selection_common_power", mean**2), ("selection_temporal_power", temporal)]:
        np.testing.assert_allclose(a[key], value, atol=5e-12, rtol=0)
    edge_arrays = []
    for w in wave:
        x = np.log(1e4/w)
        mid = .5*(x[:-1]+x[1:])
        edge_arrays.append(np.r_[2*x[0]-mid[0], mid, 2*x[-1]-mid[-1]])
    candidates = []
    for record in report["candidate_windows"]:
        b, center = record["order_index"], record["center_index"]
        assert eligible[b, center]
        mid = np.log(1e4/wave[b, center])
        low, high = mid-2.65/C_KMS, mid+2.65/C_KMS
        edges = edge_arrays[b]
        window = (edges[:-1] < high) & (edges[1:] > low)
        candidates.append((b, window))
        np.testing.assert_allclose(np.exp([low, high]),
            [record["requested_nu_lower_cm1"], record["requested_nu_upper_cm1"]], atol=1e-10, rtol=0)
    assert len(candidates) == eligible.sum()
    exclusion = np.zeros_like(included)
    step_cursor, case_results, max_stat_error = 0, {}, 0.
    for j, name in enumerate(a["case_names"].tolist()):
        case = report["cases"][name]
        icap = int(np.floor(case["budget_fraction"]*included.sum()+1e-9))
        scap = int(np.floor(case["budget_fraction"]*score.sum()+1e-9))
        assert icap == case["included_cap"] and scap == case["score_cap"]
        while step_cursor < case["selection_steps"]:
            possibilities = []
            for cid, (b, window) in enumerate(candidates):
                add = window & ~exclusion[b]
                ni, ns = int((add & included[b]).sum()), int((add & score[b]).sum())
                if ni and (exclusion & included).sum()+ni <= icap and (exclusion & score).sum()+ns <= scap:
                    captured = float(power[b, add & included[b]].sum())
                    possibilities.append((captured/ni, captured, -cid, cid, ni, ns))
            best = max(possibilities)
            step = report["greedy_selection_steps"][step_cursor]
            assert best[3] == step["candidate_id"], (name, step_cursor, best[3], step["candidate_id"])
            assert best[4:] == (step["added_included_pixels"], step["added_score_pixels"])
            b, window = candidates[best[3]]
            exclusion[b] |= window
            step_cursor += 1
        np.testing.assert_array_equal(exclusion, a["exclusion_masks"][j])
        np.testing.assert_array_equal(included & ~exclusion, a["included_masks"][j])
        np.testing.assert_array_equal(score & ~exclusion, a["score_masks"][j])
        assert (included & exclusion).sum() <= icap and (score & exclusion).sum() <= scap
        from_intervals = np.zeros_like(exclusion)
        for interval in case["intervals"]:
            b, lo, hi = interval["order_index"], interval["start_index"], interval["stop_index_exclusive"]
            assert hi > lo
            from_intervals[b, lo:hi] = True
            low, high = np.exp(edge_arrays[b][[lo, hi]])
            np.testing.assert_allclose([low, high], [interval["nu_lower_cm1"], interval["nu_upper_cm1"]], atol=1e-10, rtol=0)
            np.testing.assert_allclose([1e4/high, 1e4/low],
                [interval["wavelength_lower_um"], interval["wavelength_upper_um"]], atol=1e-13, rtol=0)
            assert int(included[b, lo:hi].sum()) == interval["newly_removed_included"]
            assert int(score[b, lo:hi].sum()) == interval["newly_removed_score"]
        np.testing.assert_array_equal(from_intervals, exclusion)
        remaining = statistic(old, score & ~exclusion, slice(5, None))
        removed = statistic(old, score & exclusion, slice(5, None))
        for label, actual in [("adopted_evaluation_retained_score", remaining),
                              ("adopted_evaluation_removed_score", removed)]:
            expected = case["original_q5_statistics"][label]
            for key, value in actual.items():
                if value is None:
                    assert expected[key] is None
                else:
                    max_stat_error = max(max_stat_error, abs(value-expected[key]))
                    np.testing.assert_allclose(value, expected[key], atol=1e-10, rtol=0)
        np.testing.assert_allclose(remaining["sse"]+removed["sse"], statistic(old, score, slice(5, None))["sse"], atol=1e-13)
        case_results[name] = dict(removed_included=int((included & exclusion).sum()),
                                  removed_score=int((score & exclusion).sum()), evaluation=remaining)
    # The callable itself has no evaluation target argument. Perturb the unused
    # evaluation rows before constructing the five training-only inputs.
    from design_wavenumber_masks import choose_masks
    contaminated = original["y_over_c"].copy()
    contaminated[:, 5:] = 1e5
    changed_training = project(contaminated[:, :5]-held, wave, included)
    changed = choose_masks(changed_training, wave, included, score)[0]
    np.testing.assert_array_equal(changed, a["exclusion_masks"])
    refit_result = None
    refit_path = OUT/"wavenumber_mask_refit.json"
    if refit_path.exists():
        refit = json.loads(refit_path.read_text())
        with np.load(refit_path.with_suffix(".npz")) as saved:
            fitted = {key: saved[key] for key in saved.files}
        discrepancies = []
        for i, name in enumerate(a["case_names"].tolist()):
            kept, scored = a["included_masks"][i], a["score_masks"][i]
            projected = project(original["y_over_c"]-original["component"], wave, kept)
            residual = project(original["y_over_c"]-fitted[f"component_{name}"], wave, kept)
            np.testing.assert_allclose(projected, fitted[f"residual_reproject_{name}"], atol=1e-10, rtol=0)
            np.testing.assert_allclose(residual, fitted[f"residual_refit_{name}"], atol=1e-10, rtol=0)
            for stage, values in [("pure_mask", old), ("reproject_only", projected), ("strength_refit", residual),
                                   ("remaining_change_from_reprojection", projected-old),
                                   ("remaining_change_from_refit", residual-projected)]:
                for scope, mask, epochs in [("training", kept, slice(0, 5)),
                                            ("evaluation_score", scored, slice(5, None)),
                                            ("evaluation_included", kept, slice(5, None))]:
                    measured = statistic(values, mask, epochs)
                    expected = refit["cases"][name][stage][scope]
                    for key, value in measured.items():
                        discrepancies.append(abs(value-expected[key]))
                        np.testing.assert_allclose(value, expected[key], atol=1e-10, rtol=0)
        for key, expected in refit["source_sha256"].items():
            assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected, key
        refit_result = dict(passed=True, maximum_metric_discrepancy=max(discrepancies),
                            stages=["pure_mask", "reproject_only", "strength_refit"])
    sources = [Path(__file__), report_path, OUT/"wavenumber_mask_candidates.npz",
               OUT/"adopted_telluric_fit.npz", OUT/"line_shape_cv_predictions.npz"]
    if refit_result:
        sources += [refit_path, refit_path.with_suffix(".npz")]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        independent_original_q5_error=float(np.max(abs(old-original["residual_filtered"]))),
        independent_oof_q5_error=float(np.max(abs(training-a["selection_training_residual"]))),
        maximum_candidate_metric_discrepancy=max_stat_error, candidate_count=len(candidates),
        greedy_steps_verified=step_cursor, physical_windows_and_intervals_verified=True,
        evaluation_contamination_changed_mask_pixels=int(np.sum(changed != a["exclusion_masks"])),
        cases=case_results, refit=refit_result,
        limitations=["This verifies arithmetic and selection data flow, not physical attribution of residuals.",
                     "OOF predictions retain earlier fitted nuisance parameters and are not fully independent reductions.",
                     "Masks remove samples. Improved retained-sample RMS is not improvement of every original datum.",
                     "Independent refit audit verifies saved forward spectra and projections, not another opacity computation."],
        checked_candidate_source_hashes=hashes_checked,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_mask_independent_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Mask audit passed; refit available:", refit_result is not None)
    print("Independent Q5 discrepancy", result["independent_original_q5_error"], "candidate metric", max_stat_error)


if __name__ == "__main__":
    main()
