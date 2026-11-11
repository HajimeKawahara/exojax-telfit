"""Construct a residual mask subject to fixed artificial-template information guards.

The guard is a sensitivity experiment for the finite declared template bank,
not a claim of lossless masking for every possible planetary spectrum.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np

from check_wavenumber_mask_signal import build_template_bank, project as signal_project
from design_wavenumber_masks import (C_KMS, make_candidates, intervals,
                                     residual_statistics)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PIXEL_BUDGET = .02
MINIMUM_RAW_SCORE_RETENTION = .98
MINIMUM_INCLUDED_INFORMATION = .98
MINIMUM_EPOCH_RAW_SCORE_RETENTION = .95
HOTSPOT_WAVELENGTH_UM = 1.5704986667683538


def information_checker(data, specs, signals):
    """Return exact white-noise polynomial-profiled information for each mask."""
    moving = [i for i, spec in enumerate(specs) if spec["role"] == "moving"]
    templates = signals[moving, :, 5:]
    wave = data["wavelength_um"]
    included, score = data["included_mask"], data["score_mask"]
    bases = []
    for w in wave:
        x = 2*(np.log(w)-np.log(w).min())/np.ptp(np.log(w))-1
        bases.append(np.polynomial.chebyshev.chebvander(x, 5))
    bases = np.array(bases)

    def evaluate(exclusion):
        keep_i, keep_s = included & ~exclusion, score & ~exclusion
        raw_per_epoch = np.einsum("tbep,bp->te", templates**2, keep_s)
        information = np.zeros(len(moving))
        for b, basis in enumerate(bases):
            keep = keep_i[b]
            local = templates[:, b, :, keep]
            # Basic indexing above would reorder the boolean axis; take it explicitly.
            local = templates[:, b][:, :, keep]
            design = basis[keep]
            gram = design.T @ design
            rhs = np.einsum("tep,pk->tek", local, design)
            coefficients = np.linalg.solve(gram, rhs.reshape(-1, 6).T).T.reshape(rhs.shape)
            projected_power = np.sum(local**2, axis=-1)-np.sum(rhs*coefficients, axis=-1)
            information += projected_power.sum(axis=1)
        return raw_per_epoch, information

    raw_before, info_before = evaluate(np.zeros_like(included))
    assert np.all(raw_before > 0) and np.all(info_before > 0)

    def check(exclusion):
        raw, information = evaluate(exclusion)
        raw_ratio = raw.sum(axis=1)/raw_before.sum(axis=1)
        epoch_ratio = raw/raw_before
        info_ratio = information/info_before
        passed = (np.all(raw_ratio >= MINIMUM_RAW_SCORE_RETENTION-1e-12)
                  and np.all(info_ratio >= MINIMUM_INCLUDED_INFORMATION-1e-12)
                  and np.all(epoch_ratio >= MINIMUM_EPOCH_RAW_SCORE_RETENTION-1e-12))
        return dict(passed=bool(passed), raw_score_retention=raw_ratio,
                    included_information_retention=info_ratio,
                    epoch_raw_score_retention=epoch_ratio,
                    raw_score_power_before=raw_before.sum(axis=1),
                    raw_score_power_after=raw.sum(axis=1),
                    included_information_before=info_before,
                    included_information_after=information)

    return check, moving


def serializable_guard(guard, specs, moving, exposures):
    records = []
    for j, index in enumerate(moving):
        records.append(dict(template=specs[index],
            raw_score_retention=float(guard["raw_score_retention"][j]),
            included_information_retention=float(guard["included_information_retention"][j]),
            epoch_raw_score_retention={str(int(e)):float(v) for e, v in
                                     zip(exposures[5:], guard["epoch_raw_score_retention"][j])},
            raw_score_power_before=float(guard["raw_score_power_before"][j]),
            raw_score_power_after=float(guard["raw_score_power_after"][j]),
            included_information_before=float(guard["included_information_before"][j]),
            included_information_after=float(guard["included_information_after"][j])))
    return dict(passed=guard["passed"],
        minimum_raw_score_retention=float(guard["raw_score_retention"].min()),
        minimum_included_information_retention=float(guard["included_information_retention"].min()),
        minimum_epoch_raw_score_retention=float(guard["epoch_raw_score_retention"].min()),
        templates=records)


def select_protected(training, wave, included, score, checker):
    """Use the original OOF ranking, rejecting windows that violate a guard."""
    candidates, eligible, power = make_candidates(training, wave, included)
    included_cap, score_cap = int(np.floor(PIXEL_BUDGET*included.sum())), int(np.floor(PIXEL_BUDGET*score.sum()))
    exclusion = np.zeros_like(included)
    steps, rejected = [], []
    while True:
        ranked = []
        for candidate in candidates:
            b = candidate["order_index"]
            added = candidate["window"] & ~exclusion[b]
            ni, ns = int(np.sum(added & included[b])), int(np.sum(added & score[b]))
            if not ni or np.sum(exclusion & included)+ni > included_cap or np.sum(exclusion & score)+ns > score_cap:
                continue
            added_power = float(np.sum(power["total_power"][b, added & included[b]]))
            ranked.append((added_power/ni, added_power, -candidate["candidate_id"], candidate, ni, ns))
        ranked.sort(key=lambda row:row[:3], reverse=True)
        accepted = None
        for efficiency, added_power, _, candidate, ni, ns in ranked:
            trial = exclusion.copy()
            trial[candidate["order_index"]] |= candidate["window"]
            guard = checker(trial)
            record = dict(step=len(steps), candidate_id=candidate["candidate_id"],
                order_index=candidate["order_index"], center_index=candidate["center_index"],
                added_included_pixels=ni, added_score_pixels=ns,
                training_power_per_added_included_pixel=efficiency,
                added_training_mean_squared_residual=added_power,
                minimum_raw_score_retention=float(guard["raw_score_retention"].min()),
                minimum_included_information_retention=float(guard["included_information_retention"].min()),
                minimum_epoch_raw_score_retention=float(guard["epoch_raw_score_retention"].min()))
            if guard["passed"]:
                steps.append(record)
                accepted = trial
                break
            rejected.append(record)
        if accepted is None:
            break
        exclusion = accepted
    return exclusion, candidates, steps, rejected, eligible, power


def main():
    data, specs, signals = build_template_bank()
    with np.load(OUT/"wavenumber_mask_candidates.npz") as old:
        training = old["selection_training_residual"]
        for key in ["wavelength_um", "included_mask", "score_mask", "orders", "exposures"]:
            np.testing.assert_array_equal(data[key], old[key])
    included, score, wave = data["included_mask"], data["score_mask"], data["wavelength_um"]
    checker, moving = information_checker(data, specs, signals)
    protected, candidates, steps, rejected, eligible, power = select_protected(training, wave, included, score, checker)
    order34 = int(np.flatnonzero(data["orders"] == 34)[0])
    fallback_candidate = min([candidate for candidate in candidates if candidate["order_index"] == order34],
        key=lambda candidate:abs(C_KMS*np.log(wave[order34,candidate["center_index"]]/HOTSPOT_WAVELENGTH_UM)))
    distance = abs(C_KMS*np.log(wave[order34,fallback_candidate["center_index"]]/HOTSPOT_WAVELENGTH_UM))
    assert distance < 2.65, distance
    fallback = np.zeros_like(included)
    fallback[order34] = fallback_candidate["window"]
    names = ["none", "protected02", "order34_hotspot"]
    masks = np.array([np.zeros_like(included), protected, fallback])
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        baseline="adopted fixed_contrast10", selection_training_exposures=data["exposures"][:5].tolist(),
        template_guard_evaluation_exposures=data["exposures"][5:].tolist(),
        pixel_budget_fraction=PIXEL_BUDGET,
        minimum_pooled_raw_score_retention=MINIMUM_RAW_SCORE_RETENTION,
        minimum_pooled_included_information_retention=MINIMUM_INCLUDED_INFORMATION,
        minimum_per_evaluation_exposure_raw_score_retention=MINIMUM_EPOCH_RAW_SCORE_RETENTION,
        moving_template_names=[specs[i]["name"] for i in moving],
        ranking="Same five OOF training residual spectra, eligibility and5.3km/s windows as the unprotected design; greedily take highest training-power-per-pixel window satisfying all cumulative guards",
        fallback="Single fixed order34 hotspot at1.5704986667683538um, supplied as a separate transparent sensitivity case",
        guard_uses="Frozen artificial template bank, phases and approximate attenuation; no evaluation observed flux or residuals select or rank windows",
        information="Exact amplitude information for equal independent pre-filter noise and degree-five polynomial nuisance terms on retained included pixels, with telluric model fixed",
        original_files_unchanged=True)
    # Freeze alternative mask arrays before calculating observed evaluation summaries.
    np.savez_compressed(OUT/"wavenumber_mask_protected.npz", case_names=np.array(names),
        exclusion_masks=masks, included_masks=included[None] & ~masks, score_masks=score[None] & ~masks,
        included_mask=included, score_mask=score, wavelength_um=wave, wavenumber_cm1=1e4/wave,
        orders=data["orders"], exposures=data["exposures"], selection_training_residual=training,
        original_residual_filtered=data["residual_filtered"], eligible_centers=eligible)
    (OUT/"wavenumber_mask_protected_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    cases, rows = {}, []
    original_sse = residual_statistics(data["residual_filtered"][:,5:], score)["sse"]
    qr_errors = []
    for name, exclusion in zip(names, masks):
        guard = checker(exclusion)
        assert guard["passed"], name
        keep_i, keep_s = included & ~exclusion, score & ~exclusion
        # Independently check Gram information using full QR-projected spectra.
        qr_information = []
        for i in moving:
            filtered = signal_project(signals[i], wave, keep_i)[:,5:]
            keep = np.broadcast_to(keep_i[:,None], filtered.shape)
            qr_information.append(float(np.sum(filtered[keep]**2)))
        error = float(np.max(abs(np.array(qr_information)/guard["included_information_after"]-1)))
        qr_errors.append(error)
        assert error < 1e-10, error
        regs = intervals(exclusion, wave, included, score, data["orders"])
        kept = residual_statistics(data["residual_filtered"][:,5:], keep_s)
        removed = residual_statistics(data["residual_filtered"][:,5:], score & exclusion)
        cases[name] = dict(removed_included=int(np.sum(included & exclusion)),
            removed_score=int(np.sum(score & exclusion)), retained_included=int(keep_i.sum()),
            retained_score=int(keep_s.sum()),
            removed_included_fraction=float(np.sum(included & exclusion)/included.sum()),
            removed_score_fraction=float(np.sum(score & exclusion)/score.sum()),
            intervals=regs, guard=serializable_guard(guard, specs, moving, data["exposures"]),
            original_q5_evaluation_retained_score=kept, original_q5_evaluation_removed_score=removed,
            original_q5_evaluation_sse_isolated_fraction=removed["sse"]/original_sse)
        rows.extend([dict(case=name, **reg) for reg in regs])
    with (OUT/"wavenumber_mask_protected.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader();writer.writerows(rows)
    sources = [Path(__file__), ROOT/"design_wavenumber_masks.py", ROOT/"check_wavenumber_mask_signal.py",
        OUT/"wavenumber_mask_candidates.json", OUT/"wavenumber_mask_candidates.npz",
        OUT/"adopted_telluric_fit.npz", OUT/"quadratic_strength_fit.npz", OUT/"line_shape_fit.json",
        ROOT/"data/20201214_info.csv"]
    report = dict(plan=plan, cases=cases, accepted_steps=steps, rejected_window_attempts=rejected,
        fallback_candidate_id=fallback_candidate["candidate_id"], qr_information_relative_error_max=max(qr_errors),
        limitations=["This is an explicitly template-aware sensitivity alternative to the original residual-only masks, which remain unchanged.",
            "The sixteen moving templates are artificial Gaussian and terrestrial-weighted35-line H2O patterns, not all possible real planetary spectra.",
            "Information guards are evaluated at known exposure phases and model attenuation. They are not based on a detected signal or evaluation residual amplitude.",
            "The included-information condition assumes equal independent noise and profiles polynomial nuisances only; telluric refitting and real covariance require separate checks.",
            "The reported observed residual RMS still uses the original Q5/model and simply omits masked samples. Reprojection/refitting is separate.",
            "The water template bank has no order34 lines, so the fallback cannot establish preservation of other planetary molecular spectra in that order.",
            "No finite nonempty mask can guarantee preservation of every unknown isolated planetary line.",
            "No alternative mask or model is automatically adopted."],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources},
        mask_npz_sha256=hashlib.sha256((OUT/"wavenumber_mask_protected.npz").read_bytes()).hexdigest())
    (OUT/"wavenumber_mask_protected.json").write_text(json.dumps(report, indent=2)+"\n")
    lines = ["Template-aware wavenumber mask sensitivity alternative", "",
        "Case | Removed included/score | Original-Q5 retained evaluation RMS(%) | Isolated SSE(%) | Min raw-score/info/epoch power retention(%)"]
    for name, record in cases.items():
        g=record["guard"]
        lines.append(f"{name} | {record['removed_included']}/{record['removed_score']} | "
            f"{record['original_q5_evaluation_retained_score']['rms_percent']:.8f} | "
            f"{100*record['original_q5_evaluation_sse_isolated_fraction']:.5f} | "
            f"{100*g['minimum_raw_score_retention']:.5f}/{100*g['minimum_included_information_retention']:.5f}/{100*g['minimum_epoch_raw_score_retention']:.5f}")
    lines += ["", "Guards: no more than2% of both original included and score pixels, at least98% pooled raw scoring power and98% exact included white-noise polynomial-profiled information for every moving template, and at least95% raw scoring power in every evaluation exposure.", "", "Limitations:"]+report["limitations"]+[""]
    (OUT/"WAVENUMBER_MASK_PROTECTED_REPORT.txt").write_text("\n".join(lines))
    print("\n".join(lines[:7]))


if __name__ == "__main__":
    main()
