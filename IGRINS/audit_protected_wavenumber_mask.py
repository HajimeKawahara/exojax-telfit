"""Check template-aware masks with independent polynomial projections."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_wavenumber_masks import project, statistic
from design_protected_wavenumber_mask import select_protected

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    path = OUT/"wavenumber_mask_protected.json"
    report = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as saved:
        a = {key: saved[key] for key in saved.files}
    signal_report = json.loads((OUT/"wavenumber_mask_signal.json").read_text())
    with np.load(OUT/"wavenumber_mask_signal.npz") as saved:
        moving = [i for i, s in enumerate(signal_report["templates"]) if s["role"] == "moving"]
        signal = saved["signals"][moving, :, 5:]
    wave, included, score = [a[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    designs = []
    for w in wave:
        lognu = np.log(1e4/w)
        x = 2*(lognu-lognu.min())/np.ptp(lognu)-1
        designs.append(np.vander(x, 6, increasing=True))

    def powers(exclusion):
        raw = np.sum(signal**2*(score & ~exclusion)[None,:,None], axis=(1, 3))
        information = np.zeros(len(moving))
        for b, design in enumerate(designs):
            keep = included[b] & ~exclusion[b]
            q, _ = np.linalg.qr(design[keep], mode="reduced")
            local = signal[:, b][:, :, keep]
            coefficients = local @ q
            projected = local-coefficients @ q.T
            information += np.sum(projected**2, axis=(1, 2))
        return raw, information

    raw0, info0 = powers(np.zeros_like(included))

    def checker(exclusion):
        raw, information = powers(exclusion)
        raw_ratio = raw.sum(axis=1)/raw0.sum(axis=1)
        epoch_ratio, info_ratio = raw/raw0, information/info0
        return dict(passed=bool(np.all(raw_ratio >= .98-1e-12)
                               and np.all(info_ratio >= .98-1e-12)
                               and np.all(epoch_ratio >= .95-1e-12)),
                    raw_score_retention=raw_ratio, included_information_retention=info_ratio,
                    epoch_raw_score_retention=epoch_ratio)

    derived, _, steps, rejected, _, _ = select_protected(a["selection_training_residual"], wave, included, score, checker)
    pindex = a["case_names"].tolist().index("protected02")
    np.testing.assert_array_equal(derived, a["exclusion_masks"][pindex])
    assert [s["candidate_id"] for s in steps] == [s["candidate_id"] for s in report["accepted_steps"]]
    assert [s["candidate_id"] for s in rejected] == [s["candidate_id"] for s in report["rejected_window_attempts"]]
    metric_error, guarded = 0., {}
    for i, name in enumerate(a["case_names"].tolist()):
        exclusion = a["exclusion_masks"][i]
        np.testing.assert_array_equal(included & ~exclusion, a["included_masks"][i])
        np.testing.assert_array_equal(score & ~exclusion, a["score_masks"][i])
        assert (included & exclusion).sum() <= np.floor(.02*included.sum())
        assert (score & exclusion).sum() <= np.floor(.02*score.sum())
        guard = checker(exclusion)
        assert guard["passed"]
        expected = report["cases"][name]["guard"]
        for j, row in enumerate(expected["templates"]):
            values = [guard["raw_score_retention"][j], guard["included_information_retention"][j]]
            targets = [row["raw_score_retention"], row["included_information_retention"]]
            metric_error = max(metric_error, float(np.max(abs(np.asarray(values)-targets))))
            np.testing.assert_allclose(values, targets, atol=1e-11, rtol=0)
            for k, exposure in enumerate(a["exposures"][5:]):
                np.testing.assert_allclose(guard["epoch_raw_score_retention"][j,k], row["epoch_raw_score_retention"][str(exposure)], atol=1e-11, rtol=0)
        kept = statistic(a["original_residual_filtered"], score & ~exclusion, slice(5, None))
        for key, value in kept.items():
            np.testing.assert_allclose(value, report["cases"][name]["original_q5_evaluation_retained_score"][key], atol=1e-11, rtol=0)
        restored = np.zeros_like(included)
        for interval in report["cases"][name]["intervals"]:
            b, start, stop = interval["order_index"], interval["start_index"], interval["stop_index_exclusive"]
            restored[b,start:stop] = True
            centers = np.log(1e4/wave[b])
            midpoints = .5*(centers[1:]+centers[:-1])
            edges = np.r_[2*centers[0]-midpoints[0], midpoints, 2*centers[-1]-midpoints[-1]]
            low, high = np.exp(edges[[start,stop]])
            np.testing.assert_allclose([low,high], [interval["nu_lower_cm1"],interval["nu_upper_cm1"]], atol=1e-10, rtol=0)
            np.testing.assert_allclose([1e4/high,1e4/low], [interval["wavelength_lower_um"],interval["wavelength_upper_um"]], atol=1e-13, rtol=0)
            assert int(included[b,start:stop].sum()) == interval["newly_removed_included"]
            assert int(score[b,start:stop].sum()) == interval["newly_removed_score"]
        np.testing.assert_array_equal(restored, exclusion)
        guarded[name] = dict(minimum_raw=float(guard["raw_score_retention"].min()),
                            minimum_information=float(guard["included_information_retention"].min()),
                            minimum_epoch=float(guard["epoch_raw_score_retention"].min()), evaluation=kept)
    # The selection's full argument list contains no observed evaluation spectra.
    # Corrupt the stored evaluation residual display array and repeat unchanged inputs.
    display = a["original_residual_filtered"].copy()
    display[:,5:] = 1e6
    again = select_protected(a["selection_training_residual"], wave, included, score, checker)[0]
    np.testing.assert_array_equal(again, derived)
    for key, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected, key
    assert hashlib.sha256(path.with_suffix(".npz").read_bytes()).hexdigest() == report["mask_npz_sha256"]
    sources = [Path(__file__), ROOT/"audit_wavenumber_masks.py", ROOT/"design_protected_wavenumber_mask.py",
               path, path.with_suffix(".npz"), OUT/"wavenumber_mask_signal.json", OUT/"wavenumber_mask_signal.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        independent_information_projection="Monomial QR, reconstructing projected spectra instead of Gram power subtraction",
        independent_guard_metric_max_discrepancy=metric_error, accepted_steps_verified=len(steps),
        rejected_attempts_verified=len(rejected), evaluation_display_contamination_changed_mask_pixels=0,
        guards=guarded,
        limitations=["Protection is conditional on the finite design bank and its supplied attenuation, phases, and polynomial model.",
                     "The order34 hotspot has no signal support in the selected35-line water template and effectively none in the initial Gaussian bank; independent support-covering tests remain necessary.",
                     "This arithmetic audit does not guarantee preservation of every real planetary line or remove pre-existing fit influence."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_mask_protected_independent_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Protected-mask audit passed", metric_error, "steps", len(steps), "rejections", len(rejected))


if __name__ == "__main__":
    main()
