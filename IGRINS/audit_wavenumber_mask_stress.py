"""Independently verify held-out template geometry and post-fit mask responses."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from audit_wavenumber_masks import project
from audit_wavenumber_mask_signal import summary

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def main():
    path = OUT/"wavenumber_protected_mask_stress.json"
    report = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as saved:
        a = {key: saved[key] for key in saved.files}
    with np.load(OUT/"quadratic_strength_fit.npz") as saved:
        attenuation = saved["component_baseline"]
    records = json.loads((OUT/"line_shape_fit.json").read_text())["shape_records"]
    info = np.genfromtxt(ROOT/"data/20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][a["exposures"]]
    wave, included, score = [a[key] for key in ["wavelength_um", "included_mask", "score_mask"]]
    reconstructed = []
    for spec in report["templates"]:
        spectrum = np.zeros_like(a["signals"][0])
        for b, order in enumerate(a["orders"]):
            reference = np.exp(np.mean(np.log(wave[b])))
            velocity = C_KMS*np.log(wave[b]/reference)
            if spec["pattern"] == "extended_gaussian":
                count = int(np.ceil((np.ptp(velocity)+440)/80))
                rng = np.random.default_rng(spec["seed"]+int(order))
                centers = np.linspace(velocity.min()-220, velocity.max()+220, count)+rng.uniform(-20,20,count)
                weights = rng.uniform(.3,1.,count)
            else:
                local = [r for r in records if r["order"] == order]
                centers = [C_KMS*np.log(1e4/r["nu"]/reference) for r in local]
                weights = [r["original_contrast"]/max(r["original_contrast"] for r in records) for r in local]
            offset = velocity[None]-spec["kp_kms"]*np.sin(2*np.pi*phase)[:,None]-spec["systemic_kms"]
            for center, weight in zip(centers,weights):
                spectrum[b] += weight*np.exp(-4*np.log(2)*((offset-center)/spec["fwhm_kms"])**2)
            spectrum[b] *= -spec["depth"]*attenuation[b]
        reconstructed.append(spectrum)
    raw = np.array(reconstructed)
    template_error = float(np.max(abs(raw-a["signals"])))
    np.testing.assert_allclose(raw, a["signals"], atol=1e-14, rtol=0)
    old = np.array([project(s,wave,included) for s in raw])
    np.testing.assert_allclose(old, a["original_q5"], atol=1e-13, rtol=0)
    metric_error, paired_error = 0., 0.
    for i, name in enumerate(a["case_names"].tolist()):
        keep, kept_score = a["included_masks"][i], a["score_masks"][i]
        changed = np.array([project(s,wave,keep) for s in raw])
        np.testing.assert_allclose(changed,a[f"new_q5_{name}"],atol=1e-13,rtol=0)
        for j, case in enumerate(report["masks"][name]["cases"]):
            for scope, original, retained in [("evaluation_score",score,kept_score),
                                              ("evaluation_included",included,keep)]:
                measured = summary(raw[j,:,5:],old[j,:,5:],changed[j,:,5:],original,retained)
                for key,value in measured.items():
                    metric_error = max(metric_error,abs(value-case[scope][key]))
                    np.testing.assert_allclose(value,case[scope][key],atol=1e-10,rtol=0)
            for e in range(5,9):
                measured = summary(raw[j,:,e:e+1],old[j,:,e:e+1],changed[j,:,e:e+1],score,kept_score)
                expected = case["per_exposure"][str(int(a["exposures"][e]))]
                for key,value in measured.items():
                    np.testing.assert_allclose(value,expected[key],atol=1e-10,rtol=0)
        with np.load(OUT/"line_shape_precision_arrays.npz") as saved:
            for record in report["reused_strength_pipeline"][name]:
                label = record["case"]["label"]
                s,p,y = [saved[f"{prefix}_{label}"][:,5:] for prefix in ["signal","expected","response"]]
                before = np.broadcast_to(score[:,None],s.shape)
                after = np.broadcast_to(kept_score[:,None],s.shape)
                denominator = float(p[before] @ p[before])
                retained_denominator = float(p[after] @ p[after])
                numerator = float(p[after] @ y[after])
                measured = dict(
                    original_q5_template_power=denominator,
                    retained_q5_template_power=retained_denominator,
                    raw_signal_power_retention=float((s[after] @ s[after])/(s[before] @ s[before])),
                    q5_template_power_retention=retained_denominator/denominator,
                    original_pipeline_gain=float((p[before] @ y[before])/denominator),
                    masked_gain_original_domain=numerator/denominator,
                    masked_gain_retained_domain=numerator/retained_denominator,
                    retained_shape_correlation=numerator/(np.linalg.norm(p[after])*np.linalg.norm(y[after])),
                    original_shape_correlation=float((p[before] @ y[before])/(np.linalg.norm(p[before])*np.linalg.norm(y[before]))),
                    retained_response_norm_relative_to_original_template=float(np.linalg.norm(y[after])/np.sqrt(denominator)))
                for key,value in measured.items():
                    paired_error = max(paired_error,abs(value-record["evaluation_score"][key]))
                    np.testing.assert_allclose(value,record["evaluation_score"][key],atol=1e-12,rtol=0)
    assert hashlib.sha256((OUT/"wavenumber_mask_protected.npz").read_bytes()).hexdigest() == report["frozen_mask_sha256"]
    for key,expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected,key
    sources = [Path(__file__),ROOT/"audit_wavenumber_masks.py",ROOT/"audit_wavenumber_mask_signal.py",
               path,path.with_suffix(".npz"),OUT/"wavenumber_mask_protected.npz",OUT/"line_shape_precision_arrays.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        template_max_discrepancy=template_error,maximum_metric_discrepancy=metric_error,
        paired_original_and_retained_domain_metric_discrepancy=paired_error,
        frozen_mask_hash_verified=True,
        interpretation="The original design-bank constraints do not guarantee the same loss limits for new line positions. Paired post-fit masks preserve retained values but still discard information.",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_mask_stress_independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Stress audit passed",template_error,metric_error,paired_error)


if __name__ == "__main__":
    main()
