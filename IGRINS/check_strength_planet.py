"""Inject moving artificial lines and refit shared telluric line strengths.

This measures the added strength-fit response relative to polynomial filtering
alone. Injection is at observed resolution; it is not a full planet retrieval.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from check_smooth_signal import forest_signal
from smooth_residuals import smoother_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def projection_metrics(raw_signal, filtered_signal, response, mask):
    """Compare injected-minus-uninjected output with the same filtered template."""
    keep = np.broadcast_to(mask[:, None], raw_signal.shape)
    raw = raw_signal[keep]
    expected = filtered_signal[keep]
    actual = response[keep]
    raw_power = float(raw @ raw)
    power = float(expected @ expected)
    error = actual-expected
    return dict(n_samples=int(keep.sum()),
                incremental_gain=float(expected @ actual / power),
                additional_loss_fraction=float(1-expected @ actual / power),
                combined_raw_gain=float(raw @ actual / raw_power),
                polynomial_only_raw_gain=float(raw @ expected / raw_power),
                response_norm_ratio=float(np.linalg.norm(actual)/np.sqrt(power)),
                response_correlation=float(expected @ actual /
                                           np.sqrt(power * (actual @ actual))),
                change_rms_percent=float(100*np.sqrt(np.mean(error**2))),
                expected_rms_percent=float(100*np.sqrt(np.mean(expected**2))))


def run_suite(fit_callback, predict_callback, baseline_theta, *, source_arrays,
              prior_scale=1.0, include_low_depth=False):
    """Run controlled injection/refit cases using explicit callable interfaces."""
    wave = np.asarray(source_arrays["wave"])
    allowed = np.asarray(source_arrays["allowed"], dtype=bool)
    scoring = np.asarray(source_arrays["score"], dtype=bool)
    orders = np.asarray(source_arrays["orders"])
    exposures = np.asarray(source_arrays["exposures"])
    observed = np.asarray(source_arrays["y_over_c"])
    component = np.asarray(source_arrays["reference_component"])
    assert exposures.tolist() == [0, 19, 39, 59, 78, 10, 30, 50, 70]
    info = np.genfromtxt(ROOT / "data" / "20201214_info.csv", delimiter=",", names=True)
    phase = info["Phase"][exposures]
    velocity = C_KMS*np.log(wave/np.exp(np.mean(np.log(wave), axis=1))[:, None])
    operators = np.array([smoother_matrix(w, m, dict(kind="polynomial", degree=5))
                          for w, m in zip(wave, allowed)])

    def project(values):
        return values-np.einsum("bij,bej->bei", operators, values)

    baseline_prediction = np.asarray(predict_callback(baseline_theta))
    if baseline_prediction.shape != observed.shape:
        raise ValueError("Prediction and observation shapes differ")
    baseline_residual = project(observed-baseline_prediction)
    cases = [dict(fwhm_kms=width, kp_kms=kp, depth=1e-3)
             for width in [6.0, 10.0] for kp in [100.0, 200.0]]
    cases += [dict(fwhm_kms=10.0, kp_kms=0.0, depth=1e-3),
              dict(fwhm_kms=10.0, kp_kms=200.0, depth=-1e-3)]
    if include_low_depth:
        cases.append(dict(fwhm_kms=10.0, kp_kms=200.0, depth=1e-4))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), prior_scale=prior_scale,
                  train_exposures=exposures[:5].tolist(), evaluation_exposures=exposures[5:].tolist(),
                  definitions={
                      "incremental_gain": "dot(Qs, injected-minus-uninjected output) / dot(Qs,Qs)",
                      "baseline": "Strength refit on the uninjected training spectra, plus the same polynomial Q",
                      "additional_loss": "One minus incremental_gain; relative to polynomial filtering alone",
                      "combined_raw_gain": "Projection on the unfiltered signal, including polynomial baseline removal",
                      "fit_target": "Only training exposures receive injected data in the fit callback",
                  },
                  limitations=[
                      "Observed-resolution artificial Gaussian forests, not physical planetary molecular spectra.",
                      "Kp=0/100/200 km/s are hypothetical controls, not measured orbital parameters.",
                      "The saved transmission/stellar component approximates detector-resolution attenuation.",
                      "TP, stellar, wavelength, IP, abundances and original continuum are held fixed by this experiment.",
                      "The test refits only the added shared line strengths; it does not cover the earlier nuisance-fit loss.",
                      "No retention value or residual-RMS improvement is a detection significance.",
                      "Finite-depth nonlinear refits with active bounds need not be exactly linear in injected depth.",
                  ], cases=[])
    for case in cases:
        started = time.monotonic()
        signal = np.array([forest_signal(v, phase, part, case["fwhm_kms"], case["kp_kms"],
                                         seed=20261111+int(order))*case["depth"]/1e-3
                           for v, part, order in zip(velocity, component, orders)])
        target = observed.copy()
        target[:, :5] += signal[:, :5]
        # Evaluation data never enter selection of the injected strength vector.
        np.testing.assert_array_equal(target[:, 5:], observed[:, 5:])
        theta, diagnostics = fit_callback(target, initial=np.asarray(baseline_theta).copy())
        theta = np.asarray(theta)
        if not all(item["success"] for item in diagnostics):
            raise RuntimeError("An injected strength fit did not converge")
        prediction = np.asarray(predict_callback(theta))
        response = project(observed+signal-prediction)-baseline_residual
        expected = project(signal)
        exact = expected-project(prediction-baseline_prediction)
        np.testing.assert_allclose(response, exact, rtol=1e-9, atol=1e-13)
        record = dict(**case,
                      training=projection_metrics(signal[:, :5], expected[:, :5], response[:, :5], allowed),
                      evaluation_score=projection_metrics(signal[:, 5:], expected[:, 5:], response[:, 5:], scoring),
                      evaluation_included=projection_metrics(signal[:, 5:], expected[:, 5:], response[:, 5:], allowed),
                      theta_change_rms=float(np.sqrt(np.mean((theta-baseline_theta)**2))),
                      theta_change_max=float(np.max(np.abs(theta-baseline_theta))),
                      active_bound_count=int(np.sum(np.abs(theta) > 1-1e-5)),
                      fitted_theta=theta.tolist(),
                      per_order={str(order): dict(
                          training=projection_metrics(signal[b:b+1, :5], expected[b:b+1, :5],
                                                      response[b:b+1, :5], allowed[b:b+1]),
                          evaluation_score=projection_metrics(signal[b:b+1, 5:], expected[b:b+1, 5:],
                                                              response[b:b+1, 5:], scoring[b:b+1]))
                          for b, order in enumerate(orders)},
                      fit_diagnostics=diagnostics,
                      injected_minus_uninjected_identity_max_error=float(np.max(np.abs(response-exact))),
                      seconds=time.monotonic()-started)
        result["cases"].append(record)
        print(f"FWHM={case['fwhm_kms']:g} Kp={case['kp_kms']:g} depth={case['depth']:g}: "
              f"evaluation gain={record['evaluation_score']['incremental_gain']:.8f}, "
              f"train gain={record['training']['incremental_gain']:.8f}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-scale", type=float, default=1.0)
    parser.add_argument("--low-depth", action="store_true")
    args = parser.parse_args()
    from fit_line_strength import build_context, fit_strengths, predict_components

    context = build_context()
    baseline_paths = []
    if args.prior_scale == 1.0 and (OUT / "line_strength_fit.npz").exists():
        with np.load(OUT / "line_strength_fit.npz") as saved:
            baseline = saved["theta_primary"].copy()
            np.testing.assert_allclose(saved["y_over_c"], context["y_over_c"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(predict_components(context, baseline), saved["component_primary"],
                                       rtol=0, atol=1e-12)
        saved_meta = json.loads((OUT / "line_strength_fit.json").read_text())
        baseline_diagnostics = saved_meta["cases"]["primary"]["fit"]
        baseline_paths = [OUT / "line_strength_fit.npz", OUT / "line_strength_fit.json"]
    else:
        baseline, baseline_diagnostics = fit_strengths(context, prior_scale=args.prior_scale)
    result = run_suite(
        lambda target, initial: fit_strengths(context, target=target, prior_scale=args.prior_scale, initial=initial),
        lambda theta: predict_components(context, theta), baseline,
        source_arrays=context, prior_scale=args.prior_scale, include_low_depth=args.low_depth)
    result["baseline_fit_diagnostics"] = baseline_diagnostics
    paths = [Path(__file__), ROOT / "fit_line_strength.py", ROOT / "line_strength_support.py",
             ROOT / "smooth_residuals.py", ROOT / "check_smooth_signal.py",
             ROOT / "data" / "20201214_info.csv", OUT / "motion_clean_control.npz"]
    paths += [OUT / f"nonstellar_control_e{exposure}.npz" for exposure in [10, 30, 50, 70]]
    paths += baseline_paths
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}
    (OUT / "strength_planet_injection.json").write_text(json.dumps(result, indent=2)+"\n")
    cases = result["cases"]
    labels = [f"{case['fwhm_kms']:g} km/s, Kp={case['kp_kms']:g}\ndepth={case['depth']:g}" for case in cases]
    x = np.arange(len(cases))
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.plot(x, [100*case["training"]["incremental_gain"] for case in cases], "o-", label="Training spectra")
    ax.plot(x, [100*case["evaluation_score"]["incremental_gain"] for case in cases], "o-", label="Evaluation scoring pixels")
    ax.axhline(100, color="0.5", ls="--", lw=1)
    ax.set(xticks=x, xticklabels=labels, ylabel="Additional strength-fit response (%)",
           title="Artificial lines after strength refitting, relative to polynomial filtering alone")
    ax.tick_params(axis="x", labelsize=8)
    ax.legend()
    fig.savefig(OUT / "strength_planet_injection.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
