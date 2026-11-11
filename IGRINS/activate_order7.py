"""Select the requested Order 7 analysis while retaining prior benchmarks."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    product = OUT / "order7_stellar_fit.npz"
    result = json.loads(product.with_suffix(".json").read_text())
    product_hash = hashlib.sha256(product.read_bytes()).hexdigest()
    assert result["product_sha256"] == product_hash
    audit = json.loads((OUT / "order7_independent_audit.json").read_text())
    assert audit["passed"]
    assert audit["source_sha256"]["results/order7_stellar_fit.npz"] == product_hash
    assert all(row["success"] for rows in result["fits"].values() for row in rows)
    refinements = result.get("null_refinements", [])
    null_drift = refinements[-1]["component_max_drift"] if refinements else None
    assert null_drift is None or np.isfinite(null_drift)
    with np.load(product) as data:
        assert data["orders"].tolist() == [5]
        assert data["display_order"].tolist() == [7]
        assert data["included_mask"].shape == (1, 1848)
        assert data["included_mask"].all()
        rms = float(100 * np.sqrt(np.mean(data["residual_filtered"][:, 5:] ** 2)))
        np.testing.assert_allclose(
            rms, result["metrics"]["refit"]["evaluation_included"]["rms_percent"], atol=1e-12
        )
    history = OUT / "model_history" / "three_order_blase"
    history.mkdir(parents=True, exist_ok=True)
    pointer = ROOT / "CURRENT_MODEL.txt"
    if pointer.exists() and not (history / "CURRENT_MODEL.txt").exists():
        (history / "CURRENT_MODEL.txt").write_bytes(pointer.read_bytes())
    manifest = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        status="Current requested analysis scope; conditional fit under evaluation",
        case="order7_blase_strength_full_region",
        product="results/order7_stellar_fit.npz", product_sha256=product_hash,
        fit_metadata="results/order7_stellar_fit.json",
        scope=json.loads((OUT / "order7_scope_audit.json").read_text()),
        selection="User requested Order 7 and inclusion of strong absorption regions",
        primary_metric="RMS across all 1848 wavelengths and four evaluation frames",
        metrics=result["metrics"]["refit"],
        numerical_checks=dict(
            strict_null_threshold=1e-8,
            strict_null_threshold_passed=result["numerically_stabilized"],
            last_maximum_prediction_change=null_drift,
            independent_audit="results/order7_independent_audit.json",
        ),
        planet_validation="Not yet performed for this order and expanded region",
        historical_manifest=("results/adopted_telluric_model.json"
                             if (OUT / "adopted_telluric_model.json").exists() else None),
    )
    (OUT / "current_order7_model.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pointer.write_text(
        "Current analysis: Order 7, including strong absorption\n"
        "====================================================\n\n"
        "Active manifest: results/current_order7_model.json\n"
        "Fit product: results/order7_stellar_fit.npz\n"
        "Requested Order 7 is released cube index 5, identified by wavelength.\n"
        "Use all 1848 available pixels: 2.2662216709-2.2945839498 microns.\n"
        "No stellar-depth, telluric-depth or residual-amplitude exclusion.\n"
        "Shared corrections use frames 0,19,39,59,78; evaluation uses 10,30,50,70.\n"
        "Primary evaluation uses all pixels; the fixed spatial score gate is secondary.\n"
        f"All-pixel evaluation RMS: {rms:.9f}% of continuum.\n"
        "PHOENIX with blase amplitudes, weak-line-regularized HITRAN strengths, Q5.\n"
        f"Last null-refit maximum prediction change: {null_drift!s} in continuum units (None if not evaluated).\n"
        f"Strict 1e-8 null threshold passed: {result['numerically_stabilized']}.\n"
        "New-order planet-signal preservation has not yet been tested.\n\n"
        "Prior three-order products, when present, are historical benchmarks\n"
        "outside the active analysis scope.\n"
        "Recreate this pointer: python IGRINS/activate_order7.py\n"
        "Code is tracked; observational data, caches and generated arrays remain Git-ignored.\n"
    )
    print(f"Selected Order 7 / cube index 5; all-pixel RMS {rms:.6f}%")


if __name__ == "__main__":
    main()
