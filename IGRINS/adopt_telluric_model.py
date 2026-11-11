"""Record the user-selected fixed-IP model with weak-line regularization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CASE = "fixed_contrast10"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    metadata_path = OUT / "quadratic_strength_fit.json"
    data_path = OUT / "quadratic_strength_fit.npz"
    audit_path = OUT / "quadratic_strength_independent_audit.json"
    metadata = json.loads(metadata_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert audit["passed"]
    for path in [metadata_path, data_path]:
        assert audit["source_sha256"][str(path.relative_to(ROOT))] == sha256(path)
    selected = metadata["cases"][CASE]
    assert selected["config"] == dict(name=CASE, ip_family="fixed",
                                      strength_prior="contrast", prior_scale=10.)
    with np.load(data_path) as saved:
        arrays = {key: saved[key] for key in ["orders", "exposures", "wavelength_um", "included_mask",
                  "score_mask", "y_over_c", "continuum", "line_parameter_indices", "ip_parameter_indices", "line_contrast"]}
        arrays["parameters"] = saved[f"theta_{CASE}"]
        arrays["component"] = saved[f"component_{CASE}"]
        arrays["residual_filtered"] = saved[f"residual_{CASE}"]
    np.testing.assert_array_equal(arrays["parameters"][arrays["ip_parameter_indices"]], 0.)
    arrays["residual_unfiltered"] = arrays["y_over_c"]-arrays["component"]
    # Independent QR check of the saved degree-five residual filter.
    for b, wave in enumerate(arrays["wavelength_um"]):
        mask = arrays["included_mask"][b]
        coordinate = np.log(wave)
        x = 2*(coordinate-coordinate.min())/np.ptp(coordinate)-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, r = np.linalg.qr(basis[mask], mode="reduced")
        coefficients = np.linalg.solve(r, q.T @ arrays["residual_unfiltered"][b][:, mask].T)
        filtered = arrays["residual_unfiltered"][b]-(basis @ coefficients).T
        np.testing.assert_allclose(filtered, arrays["residual_filtered"][b], atol=1e-11, rtol=0)
    keep = np.broadcast_to(arrays["score_mask"][:, None], arrays["residual_filtered"][:, 5:].shape)
    rms = float(100*np.sqrt(np.mean(arrays["residual_filtered"][:, 5:][keep]**2)))
    np.testing.assert_allclose(rms, selected["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12)
    product = OUT / "adopted_telluric_fit.npz"
    np.savez_compressed(product, **arrays)
    manifest = dict(
        status="adopted by explicit user instruction", created_utc=datetime.now(timezone.utc).isoformat(),
        case=CASE, config=selected["config"], physical_source="motion_clean_control",
        ip="Keep the previous symmetric two-Gaussian IP and existing epoch width factors; quadratic IP is deferred",
        strength_prior=dict(kind="contrast-weighted", maximum_multiplier=10., contrast_scale=.005,
                            formula="s_i=1+9*d_i^2/(d_i^2+0.005^2); penalty=sum((z_i/s_i)^2)",
                            contrast="Frozen original-model maximum absolute dF/d(delta S/S) on training included pixels, before poly5",
                            positivity="S_i=S_i,HITRAN*(1+u_i*z_i)>=0", uncertainty_note="Bin upper edges are chosen scales, not calibrated Gaussian sigmas"),
        filter=dict(kind="polynomial", degree=5, coordinate="normalized log wavelength", per_order=True,
                    per_exposure=True, fit_mask="all included pixels", temporal_template_subtraction=False,
                    template_operation="Apply the same degree-five projection separately to every shifted planetary template"),
        scope=dict(orders=arrays["orders"].tolist(), available_exposures=arrays["exposures"].tolist(),
                   training_exposures=arrays["exposures"][:5].tolist(), evaluation_exposures=arrays["exposures"][5:].tolist()),
        product=str(product.relative_to(ROOT)), product_sha256=sha256(product), metrics=selected["metrics"],
        definitions=dict(component="Stellar times telluric spectrum after the original IP, in original continuum units",
                         residual_filtered="Q*(observed/original_continuum-component)",
                         parameters="Stored experiment layout; the nine new IP coefficients are fixed to zero"),
        supersedes="The physical-model choice in results/adopted_residual_filter.json; that historical artifact and its audit remain unchanged",
        selection="User choice of weak-line suppression with old IP, not the experiment's CV-selected quadratic IP",
        limits="Conditional predictions for three orders and nine exposures; not a new reduction of all orders or all frames",
        source_sha256={str(path.relative_to(ROOT)):sha256(path) for path in
                       [Path(__file__), metadata_path, data_path, audit_path, OUT/"adopted_residual_filter.json"]},
    )
    (OUT/"adopted_telluric_model.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (ROOT/"CURRENT_MODEL.txt").write_text(
        "Current adopted telluric model\n"
        "==============================\n\n"
        "Authoritative manifest: results/adopted_telluric_model.json\n"
        "Predictions and filtered residuals: results/adopted_telluric_fit.npz\n"
        "Case: fixed_contrast10 from fit_quadratic_strength.py.\n"
        "Keep the previous IP; use contrast-dependent line-strength regularization.\n"
        "Keep degree-five spatial filtering; do not subtract a temporal template.\n"
        "The earlier adopted_residual_filter.json is a frozen historical artifact.\n"
        "The combined manifest above supersedes its physical-model choice.\n\n"
        "Recreate this selection with: python IGRINS/adopt_telluric_model.py\n"
        "Refit source cases with: python IGRINS/fit_quadratic_strength.py\n"
        "A refit does not automatically change this user-selected model.\n"
        "All files in IGRINS remain outside Git tracking.\n")
    print(f"Adopted {CASE}: evaluation RMS {rms:.9f}%")


if __name__ == "__main__":
    main()
