"""Measure signal power lost to fixed wavenumber masks and changed Q5 fits.

Masks are supplied by a training-only residual analysis. No signal diagnostic
changes a mask. The white-noise information quantities are conditional proxies
for the real observations, which have no supplied pixel covariance matrix.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def project(values, wave, fit_mask):
    """Apply Q5 to arrays with order first and wavelength last."""
    output = np.empty_like(values)
    for b, wavelength in enumerate(wave):
        x = 2*(np.log(wavelength)-np.log(wavelength).min())/np.ptp(np.log(wavelength))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, r = np.linalg.qr(basis[fit_mask[b]], mode="reduced")
        rows = values[b].reshape(-1, values.shape[-1])
        coefficient = np.linalg.solve(r, (rows[:, fit_mask[b]] @ q).T).T
        output[b] = (rows-coefficient @ basis.T).reshape(values[b].shape)
    return output


def make_signal(data, spec):
    """Make the frozen Gaussian or water pattern without loading a fit context."""
    signal = np.zeros_like(data["component"])
    shifts = spec["kp_kms"]*np.sin(2*np.pi*data["phase"])+spec.get("systemic_kms", 0.)
    sigma = spec["fwhm_kms"]/np.sqrt(8*np.log(2))
    normalization = max(r["original_contrast"] for r in data["shape_records"])
    for b, order in enumerate(data["orders"]):
        reference = np.exp(np.mean(np.log(data["wavelength_um"][b])))
        velocity = C_KMS*np.log(data["wavelength_um"][b]/reference)
        if spec["pattern"] == "gaussian":
            rng = np.random.default_rng(20261111+int(order))
            centers = np.linspace(velocity.min()+100, velocity.max()-100, 24)
            centers += rng.uniform(-20, 20, centers.size)
            weights = rng.uniform(.3, 1., centers.size)
        else:
            records = [r for r in data["shape_records"] if r["order"] == order]
            if not records:
                continue
            centers = C_KMS*np.log(np.array([1e4/r["nu"] for r in records])/reference)
            weights = np.array([r["original_contrast"] for r in records])/normalization
        offset = velocity[None, :, None]-centers[None, None]-shifts[:, None, None]
        forest = np.sum(weights[None, None]*np.exp(-.5*(offset/sigma)**2), axis=-1)
        signal[b] = -spec["depth"]*data["attenuation"][b]*forest
    return signal


def build_template_bank():
    """Return data, specifications and signals for external paired-refit tests."""
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(OUT/"quadratic_strength_fit.npz") as saved:
        data["attenuation"] = saved["component_baseline"]
    data["shape_records"] = json.loads((OUT/"line_shape_fit.json").read_text())["shape_records"]
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    data["phase"] = info["Phase"][data["exposures"]]
    specs = [dict(name=f"gaussian_w{w:g}_k{k:g}", pattern="gaussian", fwhm_kms=w,
                  kp_kms=k, systemic_kms=0., depth=1e-3, role="moving")
             for w in [6., 10.] for k in [100., 200.]]
    specs += [dict(name="gaussian_stationary", pattern="gaussian", fwhm_kms=10.,
                   kp_kms=0., systemic_kms=0., depth=1e-3, role="stationary_control"),
              dict(name="gaussian_negative", pattern="gaussian", fwhm_kms=10.,
                   kp_kms=200., systemic_kms=0., depth=-1e-3, role="linearity_control"),
              dict(name="gaussian_weak", pattern="gaussian", fwhm_kms=10.,
                   kp_kms=200., systemic_kms=0., depth=1e-4, role="linearity_control")]
    specs += [dict(name=f"water_w{w:g}_k{k:g}_v{v:+g}", pattern="water", fwhm_kms=w,
                   kp_kms=k, systemic_kms=v, depth=1e-4, role="moving")
              for w in [6., 10.] for k in [100., 200.] for v in [-30., 0., 30.]]
    specs += [dict(name="water_stationary", pattern="water", fwhm_kms=6.,
                   kp_kms=0., systemic_kms=0., depth=1e-4, role="stationary_control"),
              dict(name="water_deep", pattern="water", fwhm_kms=6.,
                   kp_kms=200., systemic_kms=0., depth=1e-3, role="linearity_control"),
              dict(name="water_negative", pattern="water", fwhm_kms=6.,
                   kp_kms=200., systemic_kms=0., depth=-1e-3, role="linearity_control")]
    signals = np.array([make_signal(data, spec) for spec in specs])
    return data, specs, signals


def compare_signal(raw, old_filtered, new_filtered, original_mask, retained_mask):
    """Keep removal loss separate from reprojected amplitude on retained data."""
    before = np.broadcast_to(original_mask[:, None], raw.shape)
    after = np.broadcast_to(retained_mask[:, None], raw.shape)
    raw_before, raw_after = float(np.sum(raw[before]**2)), float(np.sum(raw[after]**2))
    old_before, old_after = float(np.sum(old_filtered[before]**2)), float(np.sum(old_filtered[after]**2))
    new_after = float(np.sum(new_filtered[after]**2))
    if raw_before <= 1e-100 or old_before <= 1e-100:
        return dict(no_signal=True)
    expected, actual = old_filtered[after], new_filtered[after]
    return dict(n_original=int(before.sum()), n_retained=int(after.sum()),
        raw_original_power=raw_before, raw_retained_power=raw_after,
        original_q5_power=old_before, retained_original_q5_power=old_after,
        retained_new_q5_power=new_after,
        raw_power_retention=raw_after/raw_before,
        original_q5_power_retention=old_after/old_before,
        new_q5_power_ratio_original=new_after/old_before,
        retained_pixel_gain=float(expected @ actual/old_after) if old_after else None,
        retained_pixel_shape_correlation=float(expected @ actual/np.sqrt(old_after*new_after)) if old_after*new_after else None,
        retained_pixel_projection_change_rms=float(np.sqrt(np.mean((actual-expected)**2))))


def range_summary(values):
    a = np.array(values)
    return dict(minimum=float(a.min()), p05=float(np.percentile(a, 5)),
                median=float(np.median(a)), maximum=float(a.max()))


def isolated_line_sweep(data, candidate_masks):
    """Stress-test masking with moving single lines at many possible positions."""
    output = {name: [] for name in candidate_masks}
    phase = data["phase"][5:]
    for b, order in enumerate(data["orders"]):
        wavelength = data["wavelength_um"][b]
        velocity = C_KMS*np.log(wavelength/np.exp(np.mean(np.log(wavelength))))
        centers = np.flatnonzero(data["score_mask"][b])[::3]
        for width in [6., 10.]:
            sigma = width/np.sqrt(8*np.log(2))
            for kp in [100., 200.]:
                shift = kp*np.sin(2*np.pi*phase)
                distance = velocity[None, :, None]-velocity[centers][None, None]-shift[:, None, None]
                signals = data["attenuation"][b, 5:, :, None]*np.exp(-.5*(distance/sigma)**2)
                original_power = np.sum(signals[:, data["score_mask"][b]]**2, axis=(0, 1))
                eligible = original_power >= .01*original_power.max()
                for name, mask in candidate_masks.items():
                    remaining = np.sum(signals[:, mask[b]]**2, axis=(0, 1))
                    fractions = remaining[eligible]/original_power[eligible]
                    worst = np.flatnonzero(eligible)[int(np.argmin(fractions))]
                    output[name].append(dict(order=int(order), fwhm_kms=width, kp_kms=kp,
                        n_centers=int(eligible.sum()), retention=range_summary(fractions),
                        worst_rest_wavelength_um=float(wavelength[centers[worst]]),
                        worst_original_power=float(original_power[worst]),
                        largest_original_power=float(original_power.max())))
    return output


def main():
    data, specs, signals = build_template_bank()
    with np.load(OUT/"wavenumber_mask_candidates.npz") as saved:
        names = saved["case_names"].tolist()
        included = saved["included_masks"]
        scoring = saved["score_masks"]
        np.testing.assert_array_equal(saved["wavelength_um"], data["wavelength_um"])
        np.testing.assert_array_equal(saved["included_mask"], data["included_mask"])
        np.testing.assert_array_equal(saved["score_mask"], data["score_mask"])
    old = np.array([project(s, data["wavelength_um"], data["included_mask"]) for s in signals])
    np.testing.assert_allclose(project(data["residual_unfiltered"], data["wavelength_um"], data["included_mask"]),
                               data["residual_filtered"], atol=1e-11, rtol=0)
    suites, filtered = {}, {}
    for name, keep_included, keep_score in zip(names, included, scoring):
        assert np.all(~keep_included | data["included_mask"])
        assert np.all(~keep_score | data["score_mask"])
        new = np.array([project(s, data["wavelength_um"], keep_included) for s in signals])
        filtered[f"filtered_signals_{name}"] = new
        records = []
        for spec, raw, previous, changed in zip(specs, signals, old, new):
            record = dict(spec=spec, per_order={}, per_exposure={})
            for split, es, original, retained in [
                ("training_included", slice(0, 5), data["included_mask"], keep_included),
                ("training_score", slice(0, 5), data["score_mask"], keep_score),
                ("evaluation_score", slice(5, None), data["score_mask"], keep_score),
                ("evaluation_included", slice(5, None), data["included_mask"], keep_included)]:
                record[split] = compare_signal(raw[:, es], previous[:, es], changed[:, es], original, retained)
                if split.endswith("included"):
                    assert record[split]["new_q5_power_ratio_original"] <= 1+1e-10
            for b, order in enumerate(data["orders"]):
                record["per_order"][str(int(order))] = compare_signal(raw[b:b+1, 5:], previous[b:b+1, 5:],
                    changed[b:b+1, 5:], data["score_mask"][b:b+1], keep_score[b:b+1])
            for e in range(5, len(data["exposures"])):
                record["per_exposure"][str(int(data["exposures"][e]))] = compare_signal(raw[:, e:e+1], previous[:, e:e+1],
                    changed[:, e:e+1], data["score_mask"], keep_score)
            records.append(record)
        summary = {}
        for pattern in ["gaussian", "water"]:
            moving = [r for r in records if r["spec"]["pattern"] == pattern and r["spec"]["role"] == "moving"]
            summary[pattern] = {}
            for split in ["training_score", "evaluation_score", "evaluation_included"]:
                summary[pattern][split] = {metric: range_summary([r[split][metric] for r in moving]) for metric in
                    ["raw_power_retention", "original_q5_power_retention", "new_q5_power_ratio_original",
                     "retained_pixel_gain", "retained_pixel_shape_correlation"]}
            # Exclude groups carrying less than 0.01% of their parent template power.
            for group in ["per_order", "per_exposure"]:
                rows = [value for r in moving for value in r[group].values()
                        if not value.get("no_signal") and value["raw_original_power"] > 1e-4*r["evaluation_score"]["raw_original_power"]]
                summary[pattern][group] = {metric: range_summary([row[metric] for row in rows])
                    for metric in ["raw_power_retention", "new_q5_power_ratio_original", "retained_pixel_gain"]}
        suites[name] = dict(removed_included_fraction=float(1-keep_included.sum()/data["included_mask"].sum()),
                            removed_score_fraction=float(1-keep_score.sum()/data["score_mask"].sum()),
                            cases=records, summary=summary)
        print(name, "score removed", suites[name]["removed_score_fraction"],
              "water min raw retention", summary["water"]["evaluation_score"]["raw_power_retention"]["minimum"],
              "water min new Q5 score proxy", summary["water"]["evaluation_score"]["new_q5_power_ratio_original"]["minimum"], flush=True)
    isolated = isolated_line_sweep(data, dict(zip(names, scoring)))
    # Since masking and projection are frozen, sign/depth linearity is exact.
    index = {s["name"]: i for i, s in enumerate(specs)}
    checks = []
    for array in [signals, old]+list(filtered.values()):
        checks.extend([float(np.max(abs(array[index["water_negative"]]+array[index["water_deep"]]))),
                       float(np.max(abs(array[index["water_deep"]]-10*array[index["water_w6_k200_v+0"]]))),
                       float(np.max(abs(array[index["gaussian_negative"]]+array[index["gaussian_w10_k200"]]))),
                       float(np.max(abs(array[index["gaussian_weak"]]-.1*array[index["gaussian_w10_k200"]])))])
    assert max(checks) < 1e-14
    sources = [Path(__file__), OUT/"wavenumber_mask_candidates.json", OUT/"wavenumber_mask_candidates.npz",
               OUT/"adopted_telluric_fit.npz", OUT/"quadratic_strength_fit.npz", OUT/"line_shape_fit.json",
               ROOT/"data/20201214_info.csv"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), templates=specs, masks=suites,
        isolated_line_sweep=isolated, sign_depth_linearity_max_error=max(checks),
        definitions=dict(raw_power_retention="Sum retained s^2 / sum original s^2 on the stated original pixel set; equal independent pre-filter noise and known baseline",
            original_q5_power_retention="Fraction of the old Q5 template power remaining after masking, without changing Q5",
            new_q5_power_ratio_original="New Q5 template power on retained pixels / old Q5 template power on all original pixels",
            included_information="On included pixels, the new/old Q5 power ratio is the exact amplitude-information ratio for equal independent noise with degree-five polynomial nuisance terms and fixed telluric model",
            score_proxy="On the scoring subset, new/old Q5 power is a proxy: projection-induced noise covariance is not modeled",
            retained_pixel_gain="Projection of new Q5 template on old Q5 template, restricted to retained pixels; does not count removed data",
            isolated_line_sweep="Moving single lines at every third original score pixel; omit centers with original power below1% of maximum within each order/width/Kp"),
        limitations=["Masks are frozen from separate training residuals; none of these signal results changes mask boundaries.",
            "No telluric strength or atmospheric parameter is refitted in this mask-only test.",
            "Attenuation is the original saved stellar/telluric component at observed resolution.",
            "The water pattern uses35 terrestrial H2O lines and frozen weights, not a physical hot-planet spectrum; it has no lines in order34.",
            "Kp and systemic offsets are hypothetical stress tests; finite lines entering from outside the modeled windows are absent.",
            "Masking necessarily discards data. Good mean retention does not guarantee preservation of an individual line or all unknown planetary spectra.",
            "Real pixel uncertainties/covariances are unavailable. Information fractions do not establish actual detection SNR or significance."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    np.savez_compressed(OUT/"wavenumber_mask_signal.npz", signals=signals, original_filtered_signals=old,
        template_names=np.array([s["name"] for s in specs]), wavelength_um=data["wavelength_um"],
        orders=data["orders"], exposures=data["exposures"], included_mask=data["included_mask"],
        score_mask=data["score_mask"], case_names=np.array(names), included_masks=included, score_masks=scoring, **filtered)
    (OUT/"wavenumber_mask_signal.json").write_text(json.dumps(result, indent=2)+"\n")
    lines = ["Signal information loss from fixed wavenumber masks", "",
        "All fractions compare against the original unmasked scoring/included pixels; retained-pixel gain alone is not preservation.", "",
        "Mask | Removed score pixels (%) | Moving-water raw score power retained min (%) | Moving-water new Q5 score power ratio min (%) | White-noise included information min (%)"]
    for name, suite in suites.items():
        s = suite["summary"]["water"]
        lines.append(f"{name} | {100*suite['removed_score_fraction']:.4f} | {100*s['evaluation_score']['raw_power_retention']['minimum']:.4f} | "
                     f"{100*s['evaluation_score']['new_q5_power_ratio_original']['minimum']:.4f} | {100*s['evaluation_included']['new_q5_power_ratio_original']['minimum']:.4f}")
    lines += ["", "The score power ratio is a proxy, not covariance-aware Fisher information. The included Q5 quantity is the exact conditional white-noise information ratio with polynomial nuisances.",
              "Worst moving isolated-line raw-power retention by mask:"]
    for name, rows in isolated.items():
        worst = min(rows, key=lambda row: row["retention"]["minimum"])
        lines.append(f"{name}: {100*worst['retention']['minimum']:.4f}% (order {worst['order']}, FWHM{worst['fwhm_kms']:g}, Kp{worst['kp_kms']:g}, rest wavelength{worst['worst_rest_wavelength_um']:.9f} um)")
    lines += ["", "Limitations:"]+result["limitations"]+[""]
    (OUT/"WAVENUMBER_MASK_SIGNAL_REPORT.txt").write_text("\n".join(lines))
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    x = [100*suites[name]["removed_score_fraction"] for name in names]
    for pattern in ["gaussian", "water"]:
        y = [100*suites[name]["summary"][pattern]["evaluation_score"]["raw_power_retention"]["minimum"] for name in names]
        ax.plot(x, y, "o-", label=f"{pattern}: minimum moving-template raw power")
    ax.set(xlabel="Removed original score pixels (%)", ylabel="Retained original signal power (%)",
           title="Fixed mask loss across hypothetical moving line patterns")
    ax.legend(fontsize=8)
    fig.savefig(OUT/"wavenumber_mask_signal.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
