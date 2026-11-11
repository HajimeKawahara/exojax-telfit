"""Independently audit residual identities for the current frozen model.

This is a descriptive residual audit, not a physical source attribution.
All fits and masks are read-only; the five-exposure template uses training
residuals only and is applied without any refit to four scoring exposures.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


OUT = Path(__file__).resolve().parent / "results"
SOURCE = "motion_clean_control"
TARGETS = [f"nonstellar_control_e{e}" for e in (10, 30, 50, 70)]


def read_fit(tag):
    """Read a saved fit without importing any forward-model implementation."""
    with np.load(OUT / f"{tag}.npz") as saved:
        arrays = {key: saved[key] for key in saved.files}
    meta = json.loads((OUT / f"{tag}.json").read_text())
    return arrays, meta


def mse(value):
    return float(np.mean(np.asarray(value) ** 2))


def transfer_statistics(residual, template):
    """Keep the cross term in an arbitrary, nonorthogonal template split."""
    remainder = residual - template
    base, common, rest = mse(residual), mse(template), mse(remainder)
    cross = float(2 * np.mean(template * remainder))
    closure = base - common - rest - cross
    assert abs(closure) < 1e-16
    return {
        "baseline_rms_percent": 100 * np.sqrt(base),
        "template_rms_percent": 100 * np.sqrt(common),
        "remainder_rms_percent": 100 * np.sqrt(rest),
        "cross_term_percent_squared": 1e4 * cross,
        "fractional_sse_reduction": 1 - rest / base,
        "identity_absolute_error": abs(closure),
    }


def orthogonal_statistics(residual, mask):
    """Separate time mean, cubic projection, and zero-mean time residuals."""
    # Input shape is (order, exposure, wavelength); the mask is time invariant.
    mean = residual.mean(axis=1)
    smooth = np.empty_like(mean)
    basis = np.polynomial.chebyshev.chebvander(
        np.linspace(-1, 1, residual.shape[-1]), 3
    )
    for b, included in enumerate(mask):
        q, _ = np.linalg.qr(basis[included], mode="reduced")
        smooth[b] = np.nan
        smooth[b, included] = q @ (q.T @ mean[b, included])
    localized = mean - smooth
    varying = residual - mean[:, None]
    allowed = np.broadcast_to(mask[:, None], residual.shape)
    total = mse(residual[allowed])
    parts = {
        "smooth_time_common": mse(smooth[mask]),
        "localized_time_common": mse(localized[mask]),
        "temporal": mse(varying[allowed]),
    }
    closure = total - sum(parts.values())
    assert abs(closure) < 1e-16
    return {
        "rms_percent": 100 * np.sqrt(total),
        "components": {
            key: {"rms_percent": 100 * np.sqrt(value), "sse_fraction": value / total}
            for key, value in parts.items()
        },
        "identity_absolute_error": abs(closure),
    }


def main():
    source, meta = read_fit(SOURCE)
    indices = [int(np.flatnonzero(source["exposures"] == e).item())
               for e in meta["train_exposures"]]
    assert meta["train_exposures"] == [0, 19, 39, 59, 78]
    train_residual = ((source["observed"][:, indices] - source["model"][:, indices])
                      / source["continuum"][:, indices])
    template = np.mean(train_residual, axis=1)
    residuals, continua, score_masks = [], [], []
    for tag in TARGETS:
        data, target_meta = read_fit(tag)
        for key in ("orders", "wavelength_um", "included_mask"):
            np.testing.assert_array_equal(source[key], data[key])
        for key in ("model", "observed", "continuum", "residual_continuum"):
            np.testing.assert_array_equal(source[key][:, indices], data[key][:, indices])
        exposure = target_meta["heldout_exposure"]
        assert exposure not in meta["train_exposures"]
        e = int(np.flatnonzero(data["exposures"] == exposure).item())
        raw = data["observed"][:, e] - data["model"][:, e]
        continuum = data["continuum"][:, e]
        assert np.all(continuum > 0)
        residuals.append(raw / continuum)
        continua.append(continuum)
        score_masks.append(data["score_mask"][:, e])
    for mask in score_masks[1:]:
        np.testing.assert_array_equal(score_masks[0], mask)
    mask = score_masks[0]
    residual = np.stack(residuals, axis=1)
    continuum = np.stack(continua, axis=1)
    allowed = np.broadcast_to(mask[:, None], residual.shape)
    trained = np.broadcast_to(template[:, None], residual.shape)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE,
        "targets": TARGETS,
        "included_wavelength_count": int(source["included_mask"].sum()),
        "score_wavelength_count": int(mask.sum()),
        "score_sample_count": int(allowed.sum()),
        "score_pixels_per_order": dict(zip(map(str, source["orders"]), map(int, mask.sum(axis=1)))),
        "continuum_normalized": orthogonal_statistics(residual, mask),
        "training_template_transfer": {
            "continuum_normalized": transfer_statistics(residual[allowed], trained[allowed]),
            "fixed_observed_scale": transfer_statistics(
                (residual * continuum)[allowed], (trained * continuum)[allowed]
            ),
        },
        "per_order": {},
        "limitations": [
            "The exact orthogonal split describes residual morphology, not physical causes.",
            "The four-exposure time mean includes finite-sample noise and uses scoring data.",
            "The training-template transfer is independent of the four scoring residuals, but is not a physical model correction.",
            "Shared fit parameters and temporal evolution prevent identifying photon noise from split-training residuals alone.",
            "Weak predicted stellar absorption does not guarantee absence of unmodeled stellar blends.",
        ],
        "sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__)] + [OUT / f"{tag}.npz" for tag in [SOURCE] + TARGETS]
        },
    }
    for b, order in enumerate(source["orders"]):
        report["per_order"][str(order)] = orthogonal_statistics(residual[b:b + 1], mask[b:b + 1])
    # This sensitivity diagnostic deliberately does not estimate a noise floor.
    # Alternating sets span the night but share fitted physical parameters.
    split_a = train_residual[:, [0, 2, 4]].mean(axis=1)
    split_b = train_residual[:, [1, 3]].mean(axis=1)
    report["training_split_reproducibility"] = {
        "exposures_a": [0, 39, 78], "exposures_b": [19, 59],
        "correlation_on_score_wavelengths": float(np.corrcoef(split_a[mask], split_b[mask])[0, 1]),
        "difference_rms_percent": 100 * np.sqrt(mse((split_a - split_b)[mask])),
        "cross_product_percent_squared": float(1e4 * np.mean(split_a[mask] * split_b[mask])),
        "interpretation": "Reproducibility only; atmosphere/IP/continuum changes and shared-fit errors remain confounded with noise.",
    }
    primary_path = OUT / "nonstellar_residual_budget.json"
    if primary_path.exists():
        primary = json.loads(primary_path.read_text())
        np.testing.assert_allclose(
            100 * primary["pooled"]["rms"],
            report["continuum_normalized"]["rms_percent"], rtol=1e-13,
        )
        for primary_key, audit_key in (
            ("smooth_common", "smooth_time_common"),
            ("localized_common", "localized_time_common"),
            ("time_variable", "temporal"),
        ):
            actual = primary["decomposition"]["parts"][primary_key]
            expected = report["continuum_normalized"]["components"][audit_key]
            np.testing.assert_allclose(
                [100 * actual["rms"], actual["mse_fraction"]],
                [expected["rms_percent"], expected["sse_fraction"]], rtol=1e-13,
            )
        with np.load(primary_path.with_suffix(".npz")) as saved:
            np.testing.assert_array_equal(saved["residual_continuum"], residual)
            np.testing.assert_array_equal(saved["scoring_mask"], mask)
            np.testing.assert_array_equal(saved["training_template"], template)
        report["primary_report_validation"] = {
            "passed": True,
            "json_sha256": hashlib.sha256(primary_path.read_bytes()).hexdigest(),
            "checked": "Residual arrays, mask, training template, pooled RMS, and all orthogonal component RMS/SSE fractions.",
        }
    path = OUT / "nonstellar_budget_independent_audit.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in (
        "score_sample_count", "continuum_normalized", "training_template_transfer",
        "training_split_reproducibility"
    )}, indent=2))


if __name__ == "__main__":
    main()
