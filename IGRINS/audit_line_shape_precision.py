"""Independently audit paired injection spectra with NumPy-only projections."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="line_shape_precision")
    args = parser.parse_args()
    precision_path = OUT / f"{args.source}.json"
    precision = json.loads(precision_path.read_text())
    arrays_path = OUT / f"{args.source}_arrays.npz"
    export = json.loads(arrays_path.with_suffix(".json").read_text())
    assert hashlib.sha256(arrays_path.read_bytes()).hexdigest() == export["arrays_sha256"]
    hash_checks = {}
    for document in [precision, export]:
        for name, expected in document["source_sha256"].items():
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            hash_checks[name] = actual == expected
    assert all(hash_checks.values())
    data = np.load(arrays_path)
    original = np.load(OUT / "quadratic_strength_fit.npz")
    for left, right in [("wave", "wavelength_um"), ("allowed", "included_mask"),
                        ("score", "score_mask"), ("exposures", "exposures"),
                        ("y_over_c", "y_over_c"), ("reference_component", "component_baseline")]:
        np.testing.assert_allclose(data[left], original[right], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(data["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    shape = json.loads((OUT / "line_shape_fit.json").read_text())["shape_records"]
    info = np.genfromtxt(ROOT / "data/20201214_info.csv", delimiter=",", names=True)
    phases = info["Phase"][data["exposures"]]
    bases = []
    for wave, mask in zip(data["wave"], data["allowed"]):
        x = np.log(wave)
        x = 2 * (x - x.min()) / np.ptp(x) - 1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, upper = np.linalg.qr(basis[mask], mode="reduced")
        bases.append((basis, q, upper, mask))

    def project(values):
        return np.array([row - (basis @ np.linalg.solve(upper, q.T @ row[:, mask].T)).T
                         for row, (basis, q, upper, mask) in zip(values, bases)])

    def signal(case):
        result = np.zeros_like(data["y_over_c"])
        sigma = case["fwhm_kms"] / np.sqrt(8 * np.log(2))
        motion = case["kp_kms"] * np.sin(2 * np.pi * phases)
        normalization = max(line["original_contrast"] for line in shape)
        for b, order in enumerate(data["orders"]):
            lines = [line for line in shape if line["order"] == order]
            if not lines:
                continue
            centers = np.array([1e4 / line["nu"] for line in lines])
            weights = np.array([line["original_contrast"] / normalization for line in lines])
            velocities = C_KMS * np.log(data["wave"][b, :, None] / centers[None])
            offset = velocities[None] - motion[:, None, None]
            forest = np.sum(weights[None, None] * np.exp(-.5 * (offset / sigma)**2), axis=-1)
            result[b] = -case["depth"] * data["reference_component"][b] * forest
        return result

    def metrics(raw, expected, response, es, mask):
        keep = np.broadcast_to(mask[:, None], raw[:, es].shape)
        r, e, a = [values[:, es][keep] for values in [raw, expected, response]]
        dot = float(e @ a)
        epower, rpower, apower = float(e @ e), float(r @ r), float(a @ a)
        gain = dot / epower
        return dict(n_samples=int(keep.sum()), incremental_gain=gain,
            additional_loss_fraction=1 - gain, combined_raw_gain=float(r @ a / rpower),
            polynomial_only_raw_gain=float(r @ e / rpower), response_norm_ratio=float(np.sqrt(apower / epower)),
            response_correlation=float(dot / np.sqrt(epower * apower)),
            change_rms_percent=float(100 * np.sqrt(np.mean((a - e)**2))),
            expected_rms_percent=float(100 * np.sqrt(np.mean(e**2))))

    scopes = [("training", slice(0, 5), data["allowed"]),
              ("evaluation_score", slice(5, None), data["score"]),
              ("evaluation_included", slice(5, None), data["allowed"])]
    weak = project(signal(dict(fwhm_kms=6., kp_kms=200., depth=1e-4)))
    settings, errors = {}, []
    maxima = dict(raw_signal=0., projected_signal=0., paired_response=0., paired_identity=0., null_response=0.)
    for name, setting in precision["settings"].items():
        baseline = data[f"baseline_{name}"]
        null = project(baseline - data[f"null_{name}"])
        maxima["null_response"] = max(maxima["null_response"], float(np.max(abs(null - data[f"null_response_{name}"]))))
        baseline_residual = project(data["y_over_c"] - baseline)
        null_metrics = {}
        for scope, es, mask in scopes:
            keep = np.broadcast_to(mask[:, None], null[:, es].shape)
            r, expected = null[:, es][keep], weak[:, es][keep]
            actual = dict(rms_continuum=float(np.sqrt(np.mean(r*r))), maximum_absolute=float(np.max(abs(r))),
                projection_as_weak_signal_gain=float(expected @ r / (expected @ expected)),
                norm_relative_to_weak_signal=float(np.linalg.norm(r) / np.linalg.norm(expected)))
            for key, value in actual.items():
                errors.append(abs(value - setting["null_history"][-1]["metrics"][scope][key]))
            null_metrics[scope] = actual
        null_pass = all(value["norm_relative_to_weak_signal"] < .001 for value in null_metrics.values())
        assert null_pass == setting["null_precision_passed"]
        cases = []
        for index, case in enumerate(setting["water"]["cases"]):
            label = f"{name}_case{index}"
            raw = signal(case)
            expected = project(raw)
            prediction = data[f"prediction_{label}"]
            response = project(data["y_over_c"] + raw - prediction) - baseline_residual
            exact = project(raw + baseline - prediction)
            for key, difference in [
                ("raw_signal", raw - data[f"signal_{label}"]),
                ("projected_signal", expected - data[f"expected_{label}"]),
                ("paired_response", response - data[f"response_{label}"]),
                ("paired_identity", response - exact)]:
                maxima[key] = max(maxima[key], float(np.max(abs(difference))))
            record = {key: case[key] for key in ["fwhm_kms", "kp_kms", "depth"]}
            for scope, es, mask in scopes:
                actual = metrics(raw, expected, response, es, mask)
                for key, value in actual.items():
                    errors.append(abs(value - case[scope][key]))
                corrected = metrics(raw, expected, response - null, es, mask)
                record[scope] = actual
                record[scope]["diagnostic_null_subtracted_gain"] = corrected["incremental_gain"]
                record[scope]["orthogonal_response_norm_relative_to_expected"] = float(np.sqrt(max(
                    0., actual["response_norm_ratio"]**2 - actual["incremental_gain"]**2)))
            cases.append(record)
        settings[name] = dict(null_passed=null_pass, null_metrics=null_metrics, cases=cases)
    assert max(maxima.values()) < 1e-11
    assert max(errors) < 1e-7
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        passed=all(setting["null_passed"] for setting in settings.values()),
        method="Independent NumPy Gaussian-pattern reconstruction, degree-five QR, and paired residual algebra; no JAX, opacity or optimization",
        maximum_array_differences=maxima, maximum_metric_difference=max(errors), settings=settings,
        source_hash_checks=hash_checks, arrays_sha256=export["arrays_sha256"],
        code_review=["Only the first five exposures are injected in optimizer targets; fit_shapes restricts fitting to indices 0 through 4.",
            "Each injected fit starts from its matching tightly fitted uninjected baseline under unchanged scientific bounds, priors, masks and configuration.",
            "The paired response subtracts the corresponding uninjected residual, cancelling original data residuals.",
            "A separate zero-injection refit quantifies numerical drift. Published responses do not subtract it; the audit also reports null-subtracted gains diagnostically.",
            "Q5 coefficients are profiled independently per order and exposure on the included mask; retention is measured relative to the same filtered injected template.",
            "Original attenuation and frozen 35-line weights are identical between configurations; the template is an artificial observed-resolution water pattern.",
            "The precision script records a failed null but does not abort; this audit requires every reported null to pass.",
            "No injection outcomes select a configuration or prior scale. These tests do not cover prior stellar/continuum/column/wavelength fitting or demonstrate physical-planet preservation."],
        audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / f"{args.source}_independent_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(passed=report["passed"], maximum_array_differences=maxima,
                          maximum_metric_difference=max(errors)), indent=2))


if __name__ == "__main__":
    main()
