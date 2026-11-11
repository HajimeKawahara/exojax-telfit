"""Locate persistent residual intervals using training-only held-out predictions.

The masks select wavelengths; they do not repair retained or removed samples.
Historical models and masks remain unchanged.
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
C_KMS = 299792.458
BUDGETS = [("none", 0.), ("budget01", .01), ("budget02", .02),
           ("budget05", .05), ("budget10", .10)]
MINIMUM_RESIDUAL = .01
MINIMUM_EXPOSURES = 3
WINDOW_FWHM_KMS = 5.3


def project(values, wavelength, included):
    """Apply the original degree-five projection independently to each order."""
    output = []
    for row, wave, mask in zip(values, wavelength, included):
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, r = np.linalg.qr(basis[mask], mode="reduced")
        coefficients = np.linalg.solve(r, q.T @ row[..., mask].T)
        output.append(row-(basis @ coefficients).T)
    return np.array(output)


def bin_edges(log_nu):
    """Return monotonic pixel-bin edges in natural-log wavenumber."""
    log_nu = np.asarray(log_nu)
    if not np.all(np.diff(log_nu) > 0):
        raise ValueError("The existing windows must have ascending wavenumber.")
    middle = (log_nu[:-1]+log_nu[1:])/2
    return np.r_[log_nu[0]-(middle[0]-log_nu[0]), middle,
                 log_nu[-1]+(log_nu[-1]-middle[-1])]


def descriptive_power(residual):
    """Split empirical power into a sample mean and temporal deviations."""
    mean = residual.mean(axis=1)
    temporal = np.mean((residual-mean[:, None])**2, axis=1)
    common = mean**2
    total = np.mean(residual**2, axis=1)
    np.testing.assert_allclose(total, common+temporal, atol=1e-16, rtol=1e-12)
    return dict(mean=mean, common_power=common, temporal_power=temporal,
                total_power=total, rms=np.sqrt(total),
                median=np.median(residual, axis=1),
                n_above_threshold=np.sum(abs(residual) >= MINIMUM_RESIDUAL, axis=1))


def make_candidates(training_residual, wavelength, included):
    """Define fixed-width windows only from five training residual spectra."""
    if training_residual.shape[1] != 5:
        raise ValueError("Mask selection requires exactly five training rows.")
    power = descriptive_power(training_residual)
    eligible = (included & (power["rms"] >= MINIMUM_RESIDUAL)
                & (power["n_above_threshold"] >= MINIMUM_EXPOSURES))
    candidates = []
    for b, wave in enumerate(wavelength):
        log_nu = np.log(1e4/wave)
        edges = bin_edges(log_nu)
        for center in np.flatnonzero(eligible[b]):
            half = WINDOW_FWHM_KMS/(2*C_KMS)
            low, high = log_nu[center]-half, log_nu[center]+half
            window = (edges[:-1] < high) & (edges[1:] > low)
            indices = np.flatnonzero(window)
            candidates.append(dict(candidate_id=len(candidates), order_index=b,
                center_index=int(center), window=window,
                requested_nu_lower_cm1=float(np.exp(low)),
                requested_nu_upper_cm1=float(np.exp(high)),
                bin_nu_lower_cm1=float(np.exp(edges[indices[0]])),
                bin_nu_upper_cm1=float(np.exp(edges[indices[-1]+1])),
                bin_span_kms=float(C_KMS*(edges[indices[-1]+1]-edges[indices[0]])),
                center_training_rms_percent=float(100*power["rms"][b, center]),
                center_n_above_threshold=int(power["n_above_threshold"][b, center]),
                clipped_by_window_edge=bool(low < edges[0] or high > edges[-1])))
    return candidates, eligible, power


def choose_masks(training_residual, wavelength, included, score):
    """Greedily capture training power under two nested wavelength budgets."""
    if np.any(score & ~included):
        raise ValueError("Scoring pixels must be a subset of included pixels.")
    candidates, eligible, power = make_candidates(training_residual, wavelength, included)
    exclusion = np.zeros_like(included)
    chosen, steps, masks, cases = [], [], [], {}
    for name, fraction in BUDGETS:
        included_cap = int(np.floor(fraction*included.sum()+1e-9))
        score_cap = int(np.floor(fraction*score.sum()+1e-9))
        while fraction > 0:
            options = []
            for candidate in candidates:
                b, window = candidate["order_index"], candidate["window"]
                added = window & ~exclusion[b]
                ni, ns = int(np.sum(added & included[b])), int(np.sum(added & score[b]))
                if ni == 0:
                    continue
                if np.sum(exclusion & included)+ni > included_cap or np.sum(exclusion & score)+ns > score_cap:
                    continue
                added_power = float(np.sum(power["total_power"][b, added & included[b]]))
                efficiency = added_power/ni
                options.append((efficiency, added_power, -candidate["candidate_id"], candidate, ni, ns))
            if not options:
                break
            _, added_power, _, candidate, ni, ns = max(options, key=lambda row: row[:3])
            b = candidate["order_index"]
            exclusion[b] |= candidate["window"]
            chosen.append(candidate["candidate_id"])
            steps.append(dict(step=len(steps), entered_at_budget=name,
                candidate_id=candidate["candidate_id"], order_index=b,
                center_index=candidate["center_index"], added_included_pixels=ni,
                added_score_pixels=ns, added_training_mean_squared_residual=added_power,
                training_power_per_added_included_pixel=added_power/ni,
                cumulative_removed_included=int(np.sum(exclusion & included)),
                cumulative_removed_score=int(np.sum(exclusion & score))))
        masks.append(exclusion.copy())
        cases[name] = dict(budget_fraction=fraction, included_cap=included_cap, score_cap=score_cap,
            selected_candidate_ids=chosen.copy(), selection_steps=len(steps),
            removed_included=int(np.sum(exclusion & included)),
            removed_score=int(np.sum(exclusion & score)),
            retained_included=int(np.sum(~exclusion & included)),
            retained_score=int(np.sum(~exclusion & score)),
            removed_included_fraction=float(np.sum(exclusion & included)/included.sum()),
            removed_score_fraction=float(np.sum(exclusion & score)/score.sum()))
    masks = np.array(masks)
    assert not np.any(masks[:-1] & ~masks[1:])
    descriptions = [{k: v for k, v in candidate.items() if k != "window"} for candidate in candidates]
    return masks, cases, descriptions, steps, eligible, power


def residual_statistics(residual, mask):
    """Describe unchanged residual samples on an explicitly supplied subset."""
    keep = np.broadcast_to(mask[:, None], residual.shape)
    if not keep.any():
        return dict(n_samples=0, sse=0., rms_percent=None, common_rms_percent=None,
                    temporal_rms_percent=None)
    mean = np.broadcast_to(residual.mean(axis=1)[:, None], residual.shape)
    return dict(n_samples=int(keep.sum()), sse=float(np.sum(residual[keep]**2)),
        rms_percent=float(100*np.sqrt(np.mean(residual[keep]**2))),
        common_rms_percent=float(100*np.sqrt(np.mean(mean[keep]**2))),
        temporal_rms_percent=float(100*np.sqrt(np.mean((residual-mean)[keep]**2))))


def intervals(exclusion, wavelength, included, score, orders):
    """Merge contiguous excluded bins, including already unavailable pixels."""
    rows = []
    for b, masked in enumerate(exclusion):
        changes = np.diff(np.r_[False, masked, False].astype(int))
        starts, stops = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
        edges = bin_edges(np.log(1e4/wavelength[b]))
        for start, stop in zip(starts, stops):
            low, high = np.exp(edges[[start, stop]])
            rows.append(dict(order=int(orders[b]), order_index=b, start_index=int(start),
                stop_index_exclusive=int(stop), nu_lower_cm1=float(low), nu_upper_cm1=float(high),
                wavelength_lower_um=float(1e4/high), wavelength_upper_um=float(1e4/low),
                span_kms=float(C_KMS*(edges[stop]-edges[start])), total_grid_bins=int(stop-start),
                newly_removed_included=int(included[b, start:stop].sum()),
                newly_removed_score=int(score[b, start:stop].sum())))
    return rows


def main():
    adopted_path = OUT/"adopted_telluric_fit.npz"
    cv_path = OUT/"line_shape_cv_predictions.npz"
    with np.load(adopted_path) as saved:
        data = {key: saved[key] for key in saved.files}
    wave, included, score = [data[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    with np.load(cv_path) as saved:
        held_components = np.stack([saved[f"component_strength_control_fold{held}"][:, held]
                                    for held in range(5)], axis=1)
    oof = project(data["y_over_c"][:, :5]-held_components, wave, included)
    mask_arrays, cases, candidates, steps, eligible, power = choose_masks(oof, wave, included, score)
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline="adopted fixed_contrast10, fixed previous IP and original Q5 operator",
        training_exposures=data["exposures"][:5].tolist(),
        evaluation_exposures=data["exposures"][5:].tolist(),
        selection_source="Out-of-fold strength_control predictions: each of the five training exposures is excluded from its corresponding intensity fit",
        ranking="Mean squared OOF residual captured per newly removed included wavelength pixel, recomputed after each union; ties use candidate_id",
        eligibility=dict(minimum_training_rms=MINIMUM_RESIDUAL,
                         minimum_absolute_residual=MINIMUM_RESIDUAL, minimum_training_exposures=MINIMUM_EXPOSURES),
        window_definition="All pixel bins intersecting a +/-2.65 km/s log-wavenumber interval around an eligible included center",
        requested_window_span_kms=WINDOW_FWHM_KMS,
        budget_definition="Each cumulative mask must remove at most floor(fraction*N) of both original included and original score wavelength pixels",
        budget_fractions={name: value for name, value in BUDGETS},
        nesting="Start each budget from the complete previous budget mask, then greedily add feasible full windows",
        fixed_in_time=True, mask_selection_uses_evaluation=False,
        original_included_count=int(included.sum()), original_score_count=int(score.sum()),
        original_included_per_order=included.sum(axis=1).tolist(),
        original_score_per_order=score.sum(axis=1).tolist())
    # Freeze the selected masks and plan before reading evaluation residual slices.
    np.savez_compressed(OUT/"wavenumber_mask_candidates.npz",
        case_names=np.array([name for name, _ in BUDGETS]), exclusion_masks=mask_arrays,
        included_masks=included[None] & ~mask_arrays, score_masks=score[None] & ~mask_arrays,
        included_mask=included, score_mask=score, wavelength_um=wave, wavenumber_cm1=1e4/wave,
        orders=data["orders"], exposures=data["exposures"], eligible_centers=eligible,
        selection_training_residual=oof, selection_heldout_components=held_components,
        original_residual_filtered=data["residual_filtered"],
        selection_training_mean=power["mean"], selection_common_power=power["common_power"],
        selection_temporal_power=power["temporal_power"], selection_total_power=power["total_power"],
        selection_training_median=power["median"], selection_n_above_threshold=power["n_above_threshold"])
    (OUT/"wavenumber_mask_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    base_evaluation = residual_statistics(data["residual_filtered"][:, 5:], score)
    rows = []
    for j, (name, _) in enumerate(BUDGETS):
        exclusion = mask_arrays[j]
        kept_included, kept_score = included & ~exclusion, score & ~exclusion
        current = cases[name]
        current["intervals"] = intervals(exclusion, wave, included, score, data["orders"])
        current["original_q5_statistics"] = {
            "selection_oof_retained_included": residual_statistics(oof, kept_included),
            "selection_oof_removed_included": residual_statistics(oof, included & exclusion),
            "adopted_training_retained_included": residual_statistics(data["residual_filtered"][:, :5], kept_included),
            "adopted_evaluation_retained_score": residual_statistics(data["residual_filtered"][:, 5:], kept_score),
            "adopted_evaluation_removed_score": residual_statistics(data["residual_filtered"][:, 5:], score & exclusion),
            "adopted_evaluation_retained_included": residual_statistics(data["residual_filtered"][:, 5:], kept_included)}
        current["evaluation_sse_isolated_fraction"] = current["original_q5_statistics"]["adopted_evaluation_removed_score"]["sse"]/base_evaluation["sse"]
        current["per_order_counts"] = {str(int(order)): dict(
            removed_included=int((included[b] & exclusion[b]).sum()),
            retained_included=int(kept_included[b].sum()),
            removed_score=int((score[b] & exclusion[b]).sum()), retained_score=int(kept_score[b].sum()))
            for b, order in enumerate(data["orders"])}
        for interval in current["intervals"]:
            b = interval["order_index"]
            local = np.zeros_like(score)
            local[b, interval["start_index"]:interval["stop_index_exclusive"]] = True
            entry = dict(case=name, **interval)
            entry["selection_oof_rms_percent"] = residual_statistics(oof, local & included)["rms_percent"]
            entry["evaluation_original_q5_rms_percent"] = residual_statistics(data["residual_filtered"][:, 5:], local & score)["rms_percent"]
            rows.append(entry)
    with (OUT/"wavenumber_mask_candidates.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    sources = [Path(__file__), adopted_path, OUT/"adopted_telluric_model.json", cv_path,
               OUT/"line_shape_fit.json", OUT/"line_shape_saved_audit.json"]
    report = dict(plan=plan, cases=cases, candidate_windows=candidates, greedy_selection_steps=steps,
        descriptive_decomposition="Per wavelength, mean(r^2)=mean(r)^2+mean((r-mean(r))^2). The sample mean includes finite-sample noise; this is not a causal noise/systematics decomposition.",
        caveats=["Masks select and discard wavelength samples; no spectrum is corrected by this artifact.",
            "All reported residual values here retain the original Q5 projection and original fitted model. Reprojection and refitting are separate experiments.",
            "Only five training exposures determine mask intervals; final evaluation exposures and planet injections do not select them.",
            "OOF intensity fits still condition on earlier fitted atmosphere, instrument, star and continuum, so this is not end-to-end independent validation.",
            "The 1% repeated-residual threshold is an exploratory scale, not a formal noise significance or a guarantee of planet preservation.",
            "Pixel removal fraction is not the same as lost planet information; template-dependent power and recovered signals must be checked separately.",
            "Masks apply in the supplied common-grid frame, identically across nine exposures; they are not masks moving with a planetary velocity.",
            "The10% budget is an aggressive diagnostic. No candidate is adopted automatically."],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        mask_npz_sha256=hashlib.sha256((OUT/"wavenumber_mask_candidates.npz").read_bytes()).hexdigest())
    (OUT/"wavenumber_mask_candidates.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["Training-selected wavenumber mask candidates", "",
        "Baseline: adopted fixed_contrast10. Selection uses five out-of-fold training residual spectra only.",
        "Fixed windows span5.3 km/s before pixel-bin discretization. Budgets cap both included and score wavelength pixels.",
        "Residual values below use the original Q5 and model, with wavelengths selected afterward; they are not fit improvements.", "",
        "Case | Removed included | Removed score | Retained evaluation RMS (%) | Evaluation SSE isolated (%)"]
    for name, _ in BUDGETS:
        row = cases[name]
        lines.append(f"{name} | {row['removed_included']}/{included.sum()} | {row['removed_score']}/{score.sum()} | "
                     f"{row['original_q5_statistics']['adopted_evaluation_retained_score']['rms_percent']:.8f} | "
                     f"{100*row['evaluation_sse_isolated_fraction']:.6f}")
    lines += ["", report["descriptive_decomposition"], "", "Caveats:"]+report["caveats"]+[""]
    (OUT/"WAVENUMBER_MASK_CANDIDATES_REPORT.txt").write_text("\n".join(lines))
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    show = [name for name, _ in BUDGETS].index("budget02")
    for b, ax in enumerate(axes):
        nu = 1e4/wave[b]
        ax.plot(nu, np.where(included[b], 100*np.sqrt(power["common_power"][b]), np.nan),
                label="OOF common amplitude", lw=.8)
        ax.plot(nu, np.where(included[b], 100*np.sqrt(power["temporal_power"][b]), np.nan),
                label="OOF temporal RMS", lw=.8, alpha=.7)
        for interval in cases["budget02"]["intervals"]:
            if interval["order_index"] == b:
                ax.axvspan(interval["nu_lower_cm1"], interval["nu_upper_cm1"], color="C3", alpha=.18)
        ax.set(xlabel="Wavenumber (cm^-1)", ylabel="Original-continuum residual (%)",
               title=f"Order {data['orders'][b]}: shaded = 2% budget mask")
        ax.legend(fontsize=8)
    fig.savefig(OUT/"wavenumber_mask_candidates.png", dpi=160)
    plt.close(fig)
    print("\n".join(lines[:10]))


if __name__ == "__main__":
    main()
