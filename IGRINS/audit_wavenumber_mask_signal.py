"""Verify fixed-mask signal losses using independent arrays and projections."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_wavenumber_masks import project

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def summary(raw, old, new, original, retained):
    before = np.broadcast_to(original[:, None], raw.shape)
    after = np.broadcast_to(retained[:, None], raw.shape)
    x, y = old[after], new[after]
    a, b = raw[before], raw[after]
    old_all = old[before]
    if a @ a < 1e-100 or old_all @ old_all < 1e-100:
        return dict(no_signal=True)
    return dict(n_original=int(before.sum()), n_retained=int(after.sum()),
        raw_original_power=float(a @ a), raw_retained_power=float(b @ b),
        original_q5_power=float(old_all @ old_all), retained_original_q5_power=float(x @ x),
        retained_new_q5_power=float(y @ y), raw_power_retention=float((b @ b)/(a @ a)),
        original_q5_power_retention=float((x @ x)/(old_all @ old_all)),
        new_q5_power_ratio_original=float((y @ y)/(old_all @ old_all)),
        retained_pixel_gain=float((x @ y)/(x @ x)),
        retained_pixel_shape_correlation=float((x @ y)/(np.linalg.norm(x)*np.linalg.norm(y))),
        retained_pixel_projection_change_rms=float(np.sqrt(np.mean((y-x)**2))))


def main():
    path = OUT/"wavenumber_mask_signal.json"
    report = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as saved:
        a = {key:saved[key] for key in saved.files}
    with np.load(OUT/"quadratic_strength_fit.npz") as saved:
        attenuation = saved["component_baseline"]
    records = json.loads((OUT/"line_shape_fit.json").read_text())["shape_records"]
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][a["exposures"]]
    wave, included, score = [a[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    signals = []
    for specification in report["templates"]:
        rows = []
        for b, order in enumerate(a["orders"]):
            reference = np.exp(np.mean(np.log(wave[b])))
            velocity = C_KMS*np.log(wave[b]/reference)
            if specification["pattern"] == "gaussian":
                rng = np.random.default_rng(20261111+int(order))
                centers = np.linspace(velocity.min()+100, velocity.max()-100, 24)
                centers += rng.uniform(-20, 20, len(centers))
                weights = rng.uniform(.3, 1., len(centers))
            else:
                relevant = [r for r in records if r["order"] == order]
                centers = [C_KMS*np.log(1e4/r["nu"]/reference) for r in relevant]
                weights = [r["original_contrast"]/max(r["original_contrast"] for r in records) for r in relevant]
            row = np.zeros((len(phase), len(velocity)))
            shifted = velocity[None]-specification["kp_kms"]*np.sin(2*np.pi*phase)[:, None]-specification["systemic_kms"]
            for center, weight in zip(centers, weights):
                row += weight*np.exp(-4*np.log(2)*((shifted-center)/specification["fwhm_kms"])**2)
            rows.append(-specification["depth"]*attenuation[b]*row)
        signals.append(rows)
    signals = np.asarray(signals)
    template_error = float(np.max(abs(signals-a["signals"])))
    np.testing.assert_allclose(signals, a["signals"], atol=2e-14, rtol=0)
    old = np.array([project(s, wave, included) for s in signals])
    np.testing.assert_allclose(old, a["original_filtered_signals"], atol=2e-13, rtol=0)
    metric_error, projection_error = 0., 0.
    information_by_case, previous = {}, None
    for i, name in enumerate(a["case_names"].tolist()):
        keep, kept_score = a["included_masks"][i], a["score_masks"][i]
        new = np.array([project(s, wave, keep) for s in signals])
        projection_error = max(projection_error, float(np.max(abs(new-a[f"filtered_signals_{name}"]))))
        np.testing.assert_allclose(new, a[f"filtered_signals_{name}"], atol=2e-13, rtol=0)
        conditional_information = []
        for j, entry in enumerate(report["masks"][name]["cases"]):
            for scope, epochs, original, retained in [
                ("training_included", slice(0, 5), included, keep),
                ("training_score", slice(0, 5), score, kept_score),
                ("evaluation_score", slice(5, None), score, kept_score),
                ("evaluation_included", slice(5, None), included, keep)]:
                measured = summary(signals[j, :, epochs], old[j, :, epochs], new[j, :, epochs], original, retained)
                for key, value in measured.items():
                    metric_error = max(metric_error, abs(value-entry[scope][key]))
                    np.testing.assert_allclose(value, entry[scope][key], atol=2e-10, rtol=0)
                assert measured["raw_power_retention"] <= 1+1e-12
                if scope == "evaluation_included":
                    conditional_information.append(measured["new_q5_power_ratio_original"])
            for b, order in enumerate(a["orders"]):
                measured = summary(signals[j,b:b+1,5:], old[j,b:b+1,5:], new[j,b:b+1,5:], score[b:b+1], kept_score[b:b+1])
                expected = entry["per_order"][str(int(order))]
                for key, value in measured.items():
                    np.testing.assert_allclose(value, expected[key], atol=2e-10, rtol=0)
            for e in range(5, len(a["exposures"])):
                measured = summary(signals[j,:,e:e+1], old[j,:,e:e+1], new[j,:,e:e+1], score, kept_score)
                expected = entry["per_exposure"][str(int(a["exposures"][e]))]
                for key, value in measured.items():
                    np.testing.assert_allclose(value, expected[key], atol=2e-10, rtol=0)
        conditional_information = np.asarray(conditional_information)
        assert np.all(conditional_information <= 1+1e-10)
        if previous is not None:
            assert np.all(conditional_information <= previous+1e-10)
        previous = conditional_information
        information_by_case[name] = dict(minimum=float(conditional_information.min()), maximum=float(conditional_information.max()))
    for key, expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected, key
    paths = [Path(__file__), ROOT/"audit_wavenumber_masks.py", path, path.with_suffix(".npz"),
             OUT/"wavenumber_mask_candidates.npz", OUT/"quadratic_strength_fit.npz", OUT/"line_shape_fit.json",
             ROOT/"data/20201214_info.csv"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        independently_rebuilt_template_max_error=template_error,
        independent_projection_max_error=projection_error, maximum_metric_error=metric_error,
        original_domain_raw_power_checked=True, included_information_nested_monotonicity_checked=True,
        per_order_and_per_exposure_checked=True, included_information_by_case=information_by_case,
        interpretation="Equal independent noise and polynomial nuisance assumptions define the included information ratio. Score-subset new-Q5 norms are only proxies, not independent calibrated detection information.",
        limitations=["This mask-only audit does not refit telluric parameters or quantify real pixel covariance.",
                     "Removing few pixels can remove much of a particular line; average pattern retention is not a universal guarantee."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/"wavenumber_mask_signal_independent_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Mask signal audit passed", template_error, projection_error, metric_error)


if __name__ == "__main__":
    main()
