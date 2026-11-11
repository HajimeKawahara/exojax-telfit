"""Calibrate the TransformerPayne star on model-selected telluric-clear pixels."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fit_transformerpayne_labels import (
    CONTEXT, OUT, ROOT, Problem, STELLAR_NAMES, fit_case, plot_result,
    project, statistics,
)


def write_plot(data, components, residuals, stars):
    """Reuse the comparison layout without overwriting the original fit plot."""
    import fit_transformerpayne_labels as fitting

    # The imported plot helper writes a fixed basename; use a private output directory.
    directory = OUT / "transformerpayne_stellar_control_figures"
    directory.mkdir(exist_ok=True)
    original_out = fitting.OUT
    fitting.OUT = directory
    try:
        plot_result(data, components, residuals, stars)
    finally:
        fitting.OUT = original_out
    source = directory / "transformerpayne_label_fit.png"
    source.replace(OUT / "transformerpayne_stellar_control.png")
    directory.rmdir()


def main(args):
    started = time.monotonic()
    context_metadata = json.loads(CONTEXT.with_suffix(".json").read_text())
    assert hashlib.sha256(CONTEXT.read_bytes()).hexdigest() == context_metadata["product_sha256"]
    with np.load(CONTEXT) as saved:
        data = {key: saved[key] for key in saved.files}
    original = Problem(data)
    calibration_mask = data["included_mask"].copy()
    telluric = np.ones_like(data["y_over_c"])
    mask_report = []
    for block in original.blocks:
        b, order, sl = block["index"], block["order"], block["sl"]
        raw_ones = np.ones_like(data[f"o{order}_grid"])
        telluric[b] = np.asarray(block["component"](
            raw_ones, original.initial[4:6], original.initial[sl]))
        minimum_transmission = np.min(telluric[b, :5], axis=0)
        calibration_mask[b] = minimum_transmission > .95
        strong = np.max(1-data["phoenix_stellar"][b, :5], axis=0) > .02
        count = int(calibration_mask[b].sum())
        if count < 30 or int(np.sum(calibration_mask[b] & strong)) < 10:
            raise ValueError(f"Insufficient independent stellar calibration pixels in order {order}")
        mask_report.append(dict(order=order, calibration_pixels=count,
                                strong_stellar_calibration_pixels=int(np.sum(calibration_mask[b] & strong)),
                                original_included_pixels=int(data["included_mask"][b].sum()),
                                newly_used_pixels=int(np.sum(calibration_mask[b] & ~data["included_mask"][b])),
                                original_score_overlap=int(np.sum(calibration_mask[b] & data["score_mask"][b])),
                                transmission_range=[float(minimum_transmission.min()), float(minimum_transmission.max())]))
    print("Stellar calibration masks", json.dumps(mask_report), flush=True)
    calibration_data = dict(data, included_mask=calibration_mask)
    stellar_problem = Problem(calibration_data)
    free = np.zeros(len(original.initial), dtype=bool)
    free[:6] = True
    calibrated, stellar_fit = fit_case(stellar_problem, original.initial, free,
                                       args.max_nfev, "telluric_clear_stellar_calibration")
    np.savez_compressed(OUT / "transformerpayne_stellar_control_checkpoint.npz",
                        parameters=calibrated, calibration_mask=calibration_mask)
    (OUT / "transformerpayne_stellar_control_checkpoint.json").write_text(
        json.dumps(dict(fit=stellar_fit, mask=mask_report), indent=2) + "\n")
    strength_free = ~free
    final, strength_fit = fit_case(original, calibrated, strength_free, args.max_nfev,
                                   "original_mask_strength_refit")
    components = dict(phoenix=data["adopted_component"], initial=data["initial_tp_component"])
    stars = dict(phoenix=data["phoenix_stellar"], initial=data["initial_tp_stellar"])
    components["stellar_only"], stars["stellar_only"] = original.predictions(calibrated)
    components["joint"], stars["joint"] = original.predictions(final)
    residuals = {name: project(data["y_over_c"]-value, data["wavelength_um"], data["included_mask"])
                 for name, value in components.items()}
    np.testing.assert_allclose(residuals["phoenix"], data["adopted_residual"], rtol=0, atol=1e-10)
    np.testing.assert_array_equal(final[:6], calibrated[:6])
    for value in components.values():
        np.testing.assert_array_equal(value[0], data["adopted_component"][0])
    metrics = {name: statistics(data, residual) for name, residual in residuals.items()}
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
                  method="Fit six stellar parameters on fixed training telluric-clear pixels, then freeze the star and refit119telluric strengths on the original included pixels",
                  calibration_mask_rule="Minimum original adopted model telluric-only transmission over five training frames >0.95; no flux/residual-dependent selection",
                  evaluation_rule="Original included-mask Q5 projection and original score pixels for all cases",
                  training_frames=data["exposures"][:5].tolist(), evaluation_frames=data["exposures"][5:].tolist(),
                  mask_counts=mask_report, fits=dict(stellar=stellar_fit, strengths=strength_fit),
                  fitted_stellar=dict(zip(STELLAR_NAMES, calibrated[:6].tolist())),
                  stellar_lower=original.lower[:6].tolist(), stellar_upper=original.upper[:6].tolist(),
                  metrics=metrics, wall_seconds=time.monotonic()-started,
                  history=dict(stellar=stellar_problem.history, strengths=original.history),
                  caveats=["The stellar calibration mask deliberately changes the training pixels to include strong stellar absorption; the primary evaluation mask remains unchanged.",
                           "Stage1 profiles Q5 on the new training calibration mask; every reported comparison residual is recomputed with the original included-mask Q5.",
                           "The telluric-clear mask uses the adopted PHOENIX-era telluric model with its fitted line strengths; it is conditional, not independent physical truth.",
                           "Atmospheric columns, TP, IP, wavelength calibration, original continua, and rotation remain fixed.",
                           "Stellar parameters have broad model-support bounds and no external priors; fitted values are effective conditional labels, not a calibrated stellar retrieval.",
                           "No planet injection/recovery or adoption is performed. Order13 remains PHOENIX outside the public TransformerPayne wavelength range."])
    sources = [Path(__file__), ROOT / "fit_transformerpayne_labels.py", ROOT / "transformerpayne_labels.py",
               CONTEXT, CONTEXT.with_suffix(".json"), OUT / "adopted_telluric_fit.npz"]
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in sources}
    (OUT / "transformerpayne_stellar_control.json").write_text(json.dumps(result, indent=2) + "\n")
    arrays = {key: data[key] for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c"]}
    arrays.update(calibration_mask=calibration_mask, calibration_telluric_only=telluric,
                  parameters_calibrated=calibrated, parameters_strength_refit=final)
    for name in components:
        arrays.update({f"component_{name}": components[name], f"residual_{name}": residuals[name], f"stellar_only_{name}": stars[name]})
    np.savez_compressed(OUT / "transformerpayne_stellar_control.npz", **arrays)
    write_plot(data, components, residuals, stars)
    lines = ["TransformerPayne stellar-calibration control", "",
             result["method"], result["calibration_mask_rule"], result["evaluation_rule"], "",
             "Case | H score RMS | Order34 | Order40 | Hybrid3 RMS | H full-window | H strong-stellar diagnostic"]
    for name, entry in metrics.items():
        e = entry["evaluation"]; score = e["score"]
        lines.append(f"{name} | {score['h_orders']['rms_percent']:.8f} | {score['per_order']['34']['rms_percent']:.8f} | {score['per_order']['40']['rms_percent']:.8f} | {score['hybrid_all_orders']['rms_percent']:.8f} | {e['full_window']['h_orders']['rms_percent']:.8f} | {e['strong_stellar_diagnostic']['h_orders']['rms_percent']:.8f}")
    lines += ["", "Stellar calibration: " + json.dumps(stellar_fit),
              "Strength refit: " + json.dumps(strength_fit), "", "Interpretation limits:"] + result["caveats"]
    (OUT / "TRANSFORMERPAYNE_STELLAR_CONTROL_REPORT.txt").write_text("\n".join(lines) + "\n")
    print("Stellar control finished", result["wall_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=120)
    main(parser.parse_args())
