"""Independently audit bounded background/wavelength fits and data separation."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    meta = json.loads((OUT/"background_wavelength_fit.json").read_text())
    with np.load(OUT/"background_wavelength_fit.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "component", "residual_filtered"]:
            np.testing.assert_array_equal(arrays[key], saved[key])
    for relative, digest in meta["source_sha256"].items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    reconstructed = np.empty_like(arrays["design_raw"])
    derivative_errors = []
    for b, wave in enumerate(arrays["wavelength_um"]):
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        polynomial = np.polynomial.chebyshev.chebvander(x, 5)
        included = arrays["included_mask"][b]
        raw = arrays["design_raw"][b]
        flat = raw.transpose(1, 0, 2).reshape(len(wave), -1)
        coefficients = np.linalg.lstsq(polynomial[included], flat[included], rcond=None)[0]
        reconstructed[b] = (flat-polynomial@coefficients).reshape(len(wave), 9, 18).transpose(1, 0, 2)
        velocity = 299792.458*np.log(wave)
        sort = np.argsort(velocity)
        spline = CubicSpline(velocity[sort], arrays["component"][b][:, sort], axis=1)
        derivative = (spline(velocity+.0001)-spline(velocity-.0001))/.0002
        analytic = raw[..., 6*b+3]
        derivative_errors.append(float(np.max(abs(derivative-analytic))))
    np.testing.assert_allclose(reconstructed, arrays["design_filtered"], atol=2e-14, rtol=0)
    assert max(derivative_errors) < 2e-8
    keep = np.broadcast_to(arrays["included_mask"][:, None], arrays["residual_filtered"][:, :5].shape)
    target = arrays["residual_filtered"][:, :5][keep]
    all_design = reconstructed[:, :5][keep]
    case_audits = {}
    for name, case in meta["cases"].items():
        theta = arrays[f"theta_{name}"]
        active = np.array([record["name"] in meta["plan"]["cases"][name]
                           for record in meta["plan"]["parameter_records"]])
        assert np.all(theta[~active] == 0)
        assert np.all(abs(theta) <= arrays["bounds"]+1e-13)
        prediction = reconstructed @ theta
        np.testing.assert_allclose(prediction, arrays[f"correction_filtered_{name}"], atol=2e-14, rtol=0)
        residual = arrays["residual_filtered"]-prediction
        np.testing.assert_allclose(residual, arrays[f"residual_{name}"], atol=2e-14, rtol=0)
        keep_eval = np.broadcast_to(arrays["score_mask"][:, None], residual[:, 5:].shape)
        score = 100*np.sqrt(np.mean(residual[:, 5:][keep_eval]**2))
        np.testing.assert_allclose(score, case["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12, rtol=0)
        gradient = arrays["bounds"]*(all_design.T @ (all_design @ theta-target))
        upper = theta >= arrays["bounds"]-1e-12
        lower = theta <= -arrays["bounds"]+1e-12
        free = active&~upper&~lower
        violation = max(float(np.max(abs(gradient[free]), initial=0)),
                        float(np.max(gradient[upper], initial=0)),
                        float(np.max(-gradient[lower], initial=0)))
        assert violation < 1e-10
        cv_sse = 0.
        for held in range(5):
            cv_theta = arrays[f"cv_theta_{name}_{held}"]
            error = arrays["residual_filtered"][:, held]-reconstructed[:, held] @ cv_theta
            cv_sse += float(np.sum(error[arrays["included_mask"]]**2))
        np.testing.assert_allclose(cv_sse, meta["selection"]["cross_validation"][name]["sse"], atol=1e-12, rtol=0)
        case_audits[name] = dict(kkt_violation=violation, evaluation_rms_percent=float(score), cv_sse=cv_sse)
    ranked = sorted(case_audits, key=lambda name: case_audits[name]["cv_sse"])
    assert ranked == meta["selection"]["ranking"]
    from fit_background_wavelength import build_context, fit_case
    context = build_context()
    poisoning = {}
    for name in meta["cases"]:
        target = context["residual_filtered"].copy()
        target[:, 5:] += 1000.
        theta, _ = fit_case(context, name, target=target)
        np.testing.assert_allclose(theta, arrays[f"theta_{name}"], atol=1e-13, rtol=0)
        poisoning[name] = float(np.max(abs(theta-arrays[f"theta_{name}"])))
    try:
        fit_case(context, "wavelength2", training_indices=[0, 1, 2, 3, 5])
    except ValueError:
        pass
    else:
        raise AssertionError("Evaluation index was accepted for training")
    files = [Path(__file__), ROOT/"fit_background_wavelength.py", OUT/"background_wavelength_fit.json",
             OUT/"background_wavelength_fit.npz", OUT/"background_wavelength_selection.json"]
    audit = dict(passed=True, created_utc=datetime.now(timezone.utc).isoformat(),
                 independent_projection_max_error=float(np.max(abs(reconstructed-arrays["design_filtered"]))),
                 finite_difference_derivative_max_errors=derivative_errors,
                 evaluation_poisoning_parameter_max_difference=poisoning, cases=case_audits,
                 source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files})
    (OUT/"background_wavelength_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    baseline = meta["cases"]["baseline"]["metrics"]["evaluation_score"]
    selected = meta["selection"]["selected_case"]
    chosen = meta["cases"][selected]
    metric = chosen["metrics"]["evaluation_score"]
    report = ["Conditional background and wavelength residual tests", "===================================================", "",
              "Baseline: adopted fixed_contrast10. Original IP, strengths, TP, columns,",
              "continuum, masks, and degree-five filter remain fixed. No adopted file changed.", "",
              "Five training epochs alone determine each order-shared correction.",
              "The four evaluation epochs are used only to report frozen predictions.",
              "Conditional training CV selected: "+selected, "",
              "Case                                      CV RMS (%)  Evaluation RMS (%)"]
    for name, case in meta["cases"].items():
        report.append(f"{name:42s} {meta['selection']['cross_validation'][name]['rms_percent']:.9f} {case['metrics']['evaluation_score']['rms_percent']:.9f}")
    report += ["", f"Selected evaluation improvement: {baseline['rms_percent']-metric['rms_percent']:.9f} percentage points.",
               f"Selected evaluation SSE reduction: {100*(1-metric['sse']/baseline['sse']):.6f}%.",
               "The selected background offsets reach all three exploratory +/-3% bounds.",
               "Their high-frequency contribution is tiny after Q5; these are not measured sky levels.",
               "The leakage-only fit has interior coefficients of approximately +0.0108%,",
               "-0.0477%, +0.0501% in orders 13, 34, 40. Larger leakage is disfavored",
               "conditional on the already-fitted line strengths and fixed molecular columns.", "",
               "Definitions", "-----------",
               "Leakage: F -> F+f(x)*(1-F), equivalent to (F+b)/(1+b) for f=b/(1+b).",
               "Signed f includes an over-subtracted continuum/background possibility.",
               "Public-cube flux offset: delta F = a*median(training continuum)/continuum.",
               "Wavelength: delta F = delta v(x)*dF/dv, with order-shared Chebyshev degree 0-2.",
               "The derivative is taken from the saved observed-resolution prediction.",
               "The finite shift of the same sampled model confirms the linear result:",
               f"selected exact-shift evaluation RMS {chosen['exact_saved_model_shift_score']['rms_percent']:.9f}%.", "",
               "Interpretation", "--------------",
               "Simple shared additive backgrounds and smooth subpixel wavelength errors",
               "do not explain the remaining percent-level localized residuals in this test.",
               "This does not exclude pixel-dependent extraction/flat-field errors, variable",
               "background structure, interpolation-kernel errors, or errors in raw nod spectra.",
               "The available cube has no raw counts, sky/nod spectra, uncertainties or",
               "detector interpolation phase map; those possibilities cannot be identified here.",
               "There is no arbitrary learned narrow residual template. Planet response",
               "can be tested by refitting the saved design on injected training rows.", "",
               "Audit: independent Q5 reconstruction, bounded least-squares KKT conditions,",
               "finite-difference derivative, saved CV score, fixed-mask score, source hashes,",
               "and evaluation-data poisoning checks all passed.", ""]
    (OUT/"BACKGROUND_WAVELENGTH_REPORT.txt").write_text("\n".join(report))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
