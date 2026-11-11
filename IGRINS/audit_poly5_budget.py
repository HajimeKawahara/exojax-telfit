"""Independently audit the adopted degree-five residual filter using QR."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EXPOSURES = [10, 30, 50, 70]


def load_npz(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def qr_filter(residual, wave, allowed):
    """Fit a degree-five polynomial without importing the primary operator."""
    correction = np.empty_like(residual)
    max_orthogonality_error = 0.
    for b in range(len(wave)):
        velocity = 299792.458 * np.log(wave[b] / np.exp(np.log(wave[b]).mean()))
        x = 2 * (velocity - velocity.min()) / np.ptp(velocity) - 1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, triangular = np.linalg.qr(basis[allowed[b]], mode="reduced")
        coefficients = np.linalg.solve(triangular, q.T @ residual[b][:, allowed[b]].T)
        correction[b] = (basis @ coefficients).T
        remaining = residual[b] - correction[b]
        orthogonality = q.T @ remaining[:, allowed[b]].T
        max_orthogonality_error = max(max_orthogonality_error, float(np.max(np.abs(orthogonality))))
    assert max_orthogonality_error < 1e-12
    return residual-correction, correction, max_orthogonality_error


def statistics(values):
    return dict(samples=int(values.size), rms_percent=float(100*np.sqrt(np.mean(values**2))),
                bias_percent=float(100*values.mean()),
                p95_absolute_percent=float(100*np.percentile(np.abs(values), 95)))


def temporal_budget(residual, mask):
    keep = np.broadcast_to(mask[:, None], residual.shape)
    shared = np.broadcast_to(residual.mean(axis=1, keepdims=True), residual.shape)
    varying = residual-shared
    total = float(np.mean(residual[keep]**2))
    common_power = float(np.mean(shared[keep]**2))
    variable_power = float(np.mean(varying[keep]**2))
    cross = float(2*np.mean(shared[keep]*varying[keep]))
    np.testing.assert_allclose(total, common_power+variable_power, rtol=1e-12, atol=1e-16)
    np.testing.assert_allclose(cross, 0., atol=1e-16)
    return dict(total=statistics(residual[keep]), common=statistics(shared[keep]),
                variable=statistics(varying[keep]), common_sse_fraction=common_power/total,
                variable_sse_fraction=variable_power/total,
                twice_cross_product=cross, mse_closure=common_power+variable_power-total)


def template_budget(residual, prediction, mask):
    prediction = np.broadcast_to(prediction, residual.shape)
    keep = np.broadcast_to(mask[:, None], residual.shape)
    remaining = residual-prediction
    total = float(np.mean(residual[keep]**2))
    predicted_power = float(np.mean(prediction[keep]**2))
    remainder_power = float(np.mean(remaining[keep]**2))
    cross = float(2*np.mean(prediction[keep]*remaining[keep]))
    np.testing.assert_allclose(predicted_power+remainder_power+cross, total, rtol=1e-12, atol=1e-16)
    return dict(before=statistics(residual[keep]), after=statistics(remaining[keep]),
                total_mse=total, template_mse=predicted_power, remainder_mse=remainder_power,
                twice_cross_product=cross, scoring_sse_reduction=1-remainder_power/total)


def main():
    inputs = [OUT / "motion_clean_control.npz"]
    training = load_npz(inputs[0])
    wave, allowed = training["wavelength_um"], training["included_mask"]
    score = training["score_mask"][:, -1]
    residuals, continua = [], []
    for exposure in EXPOSURES:
        path = OUT / f"nonstellar_control_e{exposure}.npz"
        inputs.append(path)
        data = load_npz(path)
        for key in ["wavelength_um", "included_mask"]:
            np.testing.assert_array_equal(data[key], training[key])
        np.testing.assert_array_equal(data["score_mask"][:, -1], score)
        np.testing.assert_array_equal(data["model"][:, :-1], training["model"][:, :-1])
        residuals.append(((data["observed"]-data["model"])/data["continuum"])[:, -1])
        continua.append(data["continuum"][:, -1])
    residual = np.stack(residuals, axis=1)
    continuum = np.stack(continua, axis=1)
    filtered, correction, orthogonality = qr_filter(residual, wave, allowed)
    twice_filtered, second_correction, _ = qr_filter(filtered, wave, allowed)
    np.testing.assert_allclose(twice_filtered, filtered, rtol=0, atol=1e-12)
    np.testing.assert_allclose(second_correction, 0., atol=1e-12)
    training_residual = ((training["observed"]-training["model"])/training["continuum"])[:, :-1]
    filtered_training, _, _ = qr_filter(training_residual, wave, allowed)
    training_template = filtered_training.mean(axis=1, keepdims=True)
    original_template = training_residual.mean(axis=1, keepdims=True)
    filtered_mean, _, _ = qr_filter(original_template, wave, allowed)
    np.testing.assert_allclose(filtered_mean, training_template, rtol=0, atol=1e-12)
    saved = load_npz(OUT / "smooth_results.npz")
    np.testing.assert_allclose(correction, saved["correction_poly5_all_included"], rtol=0, atol=1e-12)
    inputs.append(OUT / "smooth_results.npz")
    scopes = {}
    for name, mask in [("score", score), ("included", allowed)]:
        scopes[name] = dict(
            continuum=temporal_budget(filtered, mask), fixed=temporal_budget(filtered*continuum, mask),
            training_template_continuum=template_budget(filtered, training_template, mask),
            training_template_fixed=template_budget(filtered*continuum, training_template*continuum, mask),
            per_order={str(order): dict(continuum=temporal_budget(filtered[b:b+1], mask[b:b+1]),
                                       fixed=temporal_budget((filtered*continuum)[b:b+1], mask[b:b+1]))
                       for b, order in enumerate(training["orders"])})
    primary_json = OUT / "poly5_residual_budget.json"
    primary_npz = OUT / "poly5_residual_budget.npz"
    primary_checks = {}
    if primary_json.exists() and primary_npz.exists():
        primary = json.loads(primary_json.read_text())
        arrays = load_npz(primary_npz)
        expected_arrays = dict(residual_before=residual, residual_continuum=filtered,
                               smooth_removed=correction, continuum=continuum,
                               residual_fixed_flux=filtered*continuum,
                               common=filtered.mean(axis=1),
                               temporal=filtered-filtered.mean(axis=1, keepdims=True),
                               training_template=training_template[:, 0],
                               diagnostic_template_remainder=filtered-training_template)
        for key, expected in expected_arrays.items():
            np.testing.assert_allclose(arrays[key], expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose((arrays["observed"]-arrays["model"])/continuum,
                                   filtered, rtol=0, atol=1e-12)
        np.testing.assert_allclose((arrays["observed"]-arrays["physical_model"])/continuum,
                                   residual, rtol=0, atol=1e-12)
        for scope in ["score", "included"]:
            for own_unit, main_unit in [("continuum", "continuum"), ("fixed", "fixed_flux")]:
                own = scopes[scope][own_unit]
                main = primary["budgets"][scope][main_unit]
                for own_part, main_part in [("total", "total"), ("common", "common"), ("variable", "temporal")]:
                    np.testing.assert_allclose(own[own_part]["rms_percent"], main[main_part]["rms_percent"], atol=1e-11)
                np.testing.assert_allclose(own["common_sse_fraction"], main["common"]["squared_error_fraction"], atol=1e-11)
                np.testing.assert_allclose(own["variable_sse_fraction"], main["temporal"]["squared_error_fraction"], atol=1e-11)
                transfer = scopes[scope][f"training_template_{own_unit}"]
                other = primary["template_transfer"][scope][main_unit]
                for key in ["before", "after"]:
                    np.testing.assert_allclose(transfer[key]["rms_percent"], other[key]["rms_percent"], atol=1e-11)
                np.testing.assert_allclose(transfer["twice_cross_product"], other["twice_cross_product"], atol=1e-14)
                for order in training["orders"]:
                    own_order = scopes[scope]["per_order"][str(order)][own_unit]
                    main_order = primary["per_order"][scope][str(order)][main_unit]
                    for own_part, main_part in [("total", "total"), ("common", "common"), ("variable", "temporal")]:
                        np.testing.assert_allclose(own_order[own_part]["rms_percent"], main_order[main_part]["rms_percent"], atol=1e-11)
        manifest_path = OUT / "adopted_residual_filter.json"
        manifest = json.loads(manifest_path.read_text())
        assert manifest["filter"] == dict(kind="polynomial", degree=5)
        assert manifest["temporal_template_subtraction"] is False
        assert manifest["product_sha256"] == hashlib.sha256(primary_npz.read_bytes()).hexdigest()
        primary_checks = dict(passed=True, compared_arrays=list(expected_arrays),
                              budget_scopes=["score", "included"], units=["continuum", "fixed_flux"],
                              product_manifest_hash_verified=True)
        inputs += [primary_json, primary_npz, manifest_path]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  primary_artifacts_checked=bool(primary_checks), primary_checks=primary_checks,
                  exposures=EXPOSURES,
                  scopes=scopes, maximum_polynomial_orthogonality_error=orthogonality,
                  maximum_filter_idempotency_error=float(np.max(np.abs(twice_filtered-filtered))),
                  maximum_saved_correction_difference=float(np.max(np.abs(correction-saved["correction_poly5_all_included"]))),
                  near_point73_metrics=dict(
                      order13_score_continuum_rms_percent=scopes["score"]["per_order"]["13"]["continuum"]["total"]["rms_percent"],
                      score_fixed_common_rms_percent=scopes["score"]["fixed"]["common"]["rms_percent"],
                      included_fixed_common_rms_percent=scopes["included"]["fixed"]["common"]["rms_percent"]),
                  caveats=["No current all-order total RMS equals 0.73%; nearby metrics differ in scope or component.",
                           "Shared/time-variable fractions are descriptive, not physical cause fractions.",
                           "The four-exposure mean includes finite-sample noise.",
                           "All-included smoothing uses scoring pixels, so RMS is descriptive post-filter scatter.",
                           "Training-template subtraction is diagnostic only and is not the adopted planet-analysis filter.",
                           "A training residual template can contain moving planet signal and stellar mismatch; subtraction needs a full injection-recovery study."],
                  sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in inputs+[Path(__file__)]})
    (OUT / "poly5_independent_audit.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(passed=report["passed"], scopes=scopes), indent=2))


if __name__ == "__main__":
    main()
