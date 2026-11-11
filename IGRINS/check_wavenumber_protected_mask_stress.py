"""Stress-test frozen masks on new line patterns and prior paired injections."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from check_wavenumber_mask_signal import build_template_bank, compare_signal, make_signal, project, range_summary

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def extended_forest(data, spec):
    """Include entering lines beyond both ends of each observed window."""
    signal = np.zeros_like(data["component"])
    shift = spec["kp_kms"]*np.sin(2*np.pi*data["phase"])
    sigma = spec["fwhm_kms"]/np.sqrt(8*np.log(2))
    for b, order in enumerate(data["orders"]):
        reference = np.exp(np.mean(np.log(data["wavelength_um"][b])))
        velocity = C_KMS*np.log(data["wavelength_um"][b]/reference)
        rng = np.random.default_rng(spec["seed"]+int(order))
        # Keep approximately the previous forest density while extending support.
        count = int(np.ceil((np.ptp(velocity)+440)/80))
        centers = np.linspace(velocity.min()-220, velocity.max()+220, count)
        centers += rng.uniform(-20, 20, count)
        weights = rng.uniform(.3, 1., count)
        offset = velocity[None, :, None]-centers[None, None]-shift[:, None, None]
        forest = np.sum(weights[None, None]*np.exp(-.5*(offset/sigma)**2), axis=-1)
        signal[b] = -spec["depth"]*data["attenuation"][b]*forest
    return signal


def paired_mask_metrics(raw, expected, response, old_mask, new_mask):
    """Measure fixed post-fit masking with the original denominator retained."""
    old = np.broadcast_to(old_mask[:, None], raw.shape)
    new = np.broadcast_to(new_mask[:, None], raw.shape)
    original_power = float(np.sum(expected[old]**2))
    retained_power = float(np.sum(expected[new]**2))
    original_numerator = float(expected[old] @ response[old])
    retained_numerator = float(expected[new] @ response[new])
    response_power = float(np.sum(response[new]**2))
    return dict(original_q5_template_power=original_power, retained_q5_template_power=retained_power,
        raw_signal_power_retention=float(np.sum(raw[new]**2)/np.sum(raw[old]**2)),
        q5_template_power_retention=retained_power/original_power,
        original_pipeline_gain=original_numerator/original_power,
        masked_gain_original_domain=retained_numerator/original_power,
        masked_gain_retained_domain=retained_numerator/retained_power,
        retained_shape_correlation=retained_numerator/np.sqrt(retained_power*response_power),
        original_shape_correlation=original_numerator/np.sqrt(original_power*np.sum(response[old]**2)),
        retained_response_norm_relative_to_original_template=np.sqrt(response_power/original_power))


def main():
    protected = OUT/"wavenumber_mask_protected.npz"
    frozen_hash = hashlib.sha256(protected.read_bytes()).hexdigest()
    with np.load(protected) as source:
        names = source["case_names"].tolist()
        included = source["included_masks"]
        scoring = source["score_masks"]
    data, _, _ = build_template_bank()
    specs = [dict(name=f"water_holdout_w{w:g}_k{k:g}_v{v:+g}", pattern="water", fwhm_kms=w,
                  kp_kms=k, systemic_kms=v, depth=1e-4, role="heldout_geometry")
             for w in [6., 10.] for k in [100., 200.] for v in [-40., -20., -10., 10., 20., 40.]]
    specs += [dict(name=f"extended_gaussian_s{seed}_w{w:g}_k{k:g}", pattern="extended_gaussian", seed=seed,
                   fwhm_kms=w, kp_kms=k, systemic_kms=0., depth=1e-4, role="heldout_geometry")
              for seed in [20261113, 20261127, 20261209] for w in [6., 10.] for k in [100., 200.]]
    signals = np.array([make_signal(data, s) if s["pattern"] == "water" else extended_forest(data, s) for s in specs])
    old = np.array([project(s, data["wavelength_um"], data["included_mask"]) for s in signals])
    suites, arrays = {}, {}
    for name, keep_included, keep_score in zip(names, included, scoring):
        new = np.array([project(s, data["wavelength_um"], keep_included) for s in signals])
        arrays[f"new_q5_{name}"] = new
        records = []
        for spec, raw, previous, changed in zip(specs, signals, old, new):
            record = dict(spec=spec,
                evaluation_score=compare_signal(raw[:, 5:], previous[:, 5:], changed[:, 5:], data["score_mask"], keep_score),
                evaluation_included=compare_signal(raw[:, 5:], previous[:, 5:], changed[:, 5:], data["included_mask"], keep_included),
                per_exposure={}, per_order={})
            for e in range(5, 9):
                record["per_exposure"][str(int(data["exposures"][e]))] = compare_signal(
                    raw[:, e:e+1], previous[:, e:e+1], changed[:, e:e+1], data["score_mask"], keep_score)
            for b, order in enumerate(data["orders"]):
                record["per_order"][str(int(order))] = compare_signal(
                    raw[b:b+1, 5:], previous[b:b+1, 5:], changed[b:b+1, 5:], data["score_mask"][b:b+1], keep_score[b:b+1])
            assert record["evaluation_included"]["new_q5_power_ratio_original"] <= 1+1e-10
            records.append(record)
        summary = {}
        for pattern in ["water", "extended_gaussian"]:
            subset = [r for r in records if r["spec"]["pattern"] == pattern]
            summary[pattern] = {
                scope: {metric: range_summary([r[scope][metric] for r in subset])
                        for metric in ["raw_power_retention", "original_q5_power_retention", "new_q5_power_ratio_original",
                                       "retained_pixel_gain", "retained_pixel_shape_correlation"]}
                for scope in ["evaluation_score", "evaluation_included"]}
            summary[pattern]["worst_exposure_raw_retention"] = min(
                r["raw_power_retention"] for parent in subset for r in parent["per_exposure"].values())
            summary[pattern]["worst_raw_template"] = min(subset, key=lambda r: r["evaluation_score"]["raw_power_retention"])["spec"]["name"]
        suites[name] = dict(cases=records, summary=summary)
        print(name, {pattern: dict(raw_min=s["evaluation_score"]["raw_power_retention"]["minimum"],
                                  included_info_min=s["evaluation_included"]["new_q5_power_ratio_original"]["minimum"],
                                  epoch_raw_min=s["worst_exposure_raw_retention"]) for pattern, s in summary.items()}, flush=True)
    # Reuse already tightly converged paired injections; no fitting or opacity call.
    paired = {}
    paired_meta = json.loads((OUT/"line_shape_precision_arrays.json").read_text())
    with np.load(OUT/"line_shape_precision_arrays.npz") as source:
        for key, expected in [("wave", data["wavelength_um"]), ("score", data["score_mask"]), ("allowed", data["included_mask"])]:
            np.testing.assert_array_equal(source[key], expected)
        for name, keep_score in zip(names, scoring):
            records = []
            for case in paired_meta["cases"]["strength_control"]:
                label = case["label"]
                raw, expected, response = [source[f"{prefix}_{label}"] for prefix in ["signal", "expected", "response"]]
                record = dict(case=case,
                    evaluation_score=paired_mask_metrics(raw[:, 5:], expected[:, 5:], response[:, 5:], data["score_mask"], keep_score))
                metric = record["evaluation_score"]
                np.testing.assert_allclose(metric["masked_gain_original_domain"],
                    metric["masked_gain_retained_domain"]*metric["q5_template_power_retention"], atol=1e-14, rtol=0)
                records.append(record)
            paired[name] = records
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == frozen_hash
    sources = [Path(__file__), ROOT/"check_wavenumber_mask_signal.py", protected, protected.with_suffix(".json"),
        OUT/"adopted_telluric_fit.npz", OUT/"quadratic_strength_fit.npz", OUT/"line_shape_fit.json",
        ROOT/"data/20201214_info.csv", OUT/"line_shape_precision_arrays.npz", OUT/"line_shape_precision_arrays.json"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), masks=suites, templates=specs,
        reused_strength_pipeline=paired, frozen_mask_sha256=frozen_hash,
        selection="All mask files were frozen before this separate geometry/forest stress test; no mask is retuned here",
        differences_from_design_bank="Water systemic offsets -40,-20,-10,+10,+20,+40 km/s were absent from design; Gaussian seeds differ and rest centers extend220 km/s beyond both boundaries",
        pure_postfit_definition="Keep original Q5 and original strength fit, then apply the new score mask; retained data and paired responses are unchanged",
        limitations=["These are finite artificial line-pattern stress tests, not guarantees for real planetary spectra.",
            "Raw template-power retention assumes equal independent noise and a known baseline; score Q5 powers remain conditional proxies.",
            "The included new-Q5 ratio is the conditional white-noise information ratio after refitting polynomial nuisance terms; no physical nuisance fit is repeated.",
            "The reused strength-only paired injection is at FWHM6,Kp200,zero systemic velocity; it is not an injection/refit under the new masks.",
            "For the narrow order34 mask the reused H2O pattern has no signal there, so exact retained gain is a bank coverage limitation rather than universal preservation.",
            "Masking can entirely remove an individual line crossing an excluded interval even when a multi-line template retains most of its power."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_protected_mask_stress.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"wavenumber_protected_mask_stress.npz", signals=signals, original_q5=old,
        case_names=np.array(names), template_names=np.array([s["name"] for s in specs]), included_masks=included,
        score_masks=scoring, included_mask=data["included_mask"], score_mask=data["score_mask"],
        wavelength_um=data["wavelength_um"], orders=data["orders"], exposures=data["exposures"], **arrays)
    lines = ["Independent geometry stress tests for frozen wavenumber masks", "",
        "Masks were frozen before this new template grid. No tuning follows these results.",
        "Pure post-fit masking keeps original Q5 and strength fits; retained data stay unchanged.", "",
        "Mask | Pattern | Minimum raw score power retained (%) | Minimum included Q5 information (%) | Worst exposure raw score power retained (%)"]
    for name, suite in suites.items():
        for pattern, summary in suite["summary"].items():
            lines.append(f"{name} | {pattern} | {100*summary['evaluation_score']['raw_power_retention']['minimum']:.6f} | "
                         f"{100*summary['evaluation_included']['new_q5_power_ratio_original']['minimum']:.6f} | {100*summary['worst_exposure_raw_retention']:.6f}")
    lines += ["", "Reused original-pipeline moving H2O injection, depth1e-4:",
              "Mask | Original gain (%) | Masked gain, original denominator (%) | Masked gain, retained denominator (%) | Retained shape correlation"]
    for name, records in paired.items():
        m = next(r["evaluation_score"] for r in records if r["case"]["depth"] == 1e-4)
        lines.append(f"{name} | {100*m['original_pipeline_gain']:.6f} | {100*m['masked_gain_original_domain']:.6f} | "
                     f"{100*m['masked_gain_retained_domain']:.6f} | {m['retained_shape_correlation']:.6f}")
    lines += ["", "Limitations:"]+result["limitations"]+[""]
    (OUT/"WAVENUMBER_PROTECTED_MASK_STRESS_REPORT.txt").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
