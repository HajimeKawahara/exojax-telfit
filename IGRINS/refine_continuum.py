"""Refit smooth continua with frozen spectra and export observed CCF residuals."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from log_continuum import fit_log_continuum, filter_log_continuum_template
from smooth_continuum import (
    cubic_spline_basis,
    filter_continuum_template,
    fit_positive_continuum,
)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def rms(values):
    return float(100 * np.sqrt(np.mean(np.asarray(values) ** 2)))


def project(values, basis):
    """Remove the same six smooth modes independently from each frame."""
    coefficients = np.linalg.lstsq(basis, values.T, rcond=None)[0].T
    smooth = coefficients @ basis.T
    return values - smooth, smooth, coefficients


def filter_ccf_template(template, product, order, frame):
    """Apply the local continuum-plus-Q5 response to a native-flux template.

    The template must already include the IP, sampling and observational flux
    scale. This response holds physical parameters and the active NNLS bounds
    fixed; it is not an end-to-end response of the original physical fit.
    """
    mode = (str(product["continuum_parameterization"][order])
            if "continuum_parameterization" in product else "positive_bspline")
    if mode == "log_bspline":
        filtered = filter_log_continuum_template(
            template, product["observed_fixed_scale"][order, frame],
            product["full_model"][order, frame], product["continuum_basis_matrix"],
            product["continuum_fixed_scale"][order, frame],
            active_mask=(product["continuum_active_lower_bounds"][order, frame]
                         | product["continuum_active_upper_bounds"][order, frame]))
    elif mode == "positive_bspline":
        filtered = filter_continuum_template(
            template, product["full_model"][order, frame],
            product["continuum_basis_matrix"],
            active_lower_mask=product["continuum_active_lower_bounds"][order, frame],
            regularization_strength=float(product["continuum_regularization"]))
    else:
        raise ValueError(f"Unknown continuum parameterization: {mode}")
    basis = product["q5_basis_matrix"][order]
    coefficients = np.linalg.lstsq(basis, filtered.reshape(-1, basis.shape[0]).T,
                                  rcond=None)[0].T
    return (filtered.reshape(-1, basis.shape[0]) - coefficients @ basis.T).reshape(filtered.shape)


def main(args):
    started = time.monotonic()
    if (len(set(args.log_continuum_orders)) != len(args.log_continuum_orders)
            or any(order not in range(44) for order in args.log_continuum_orders)):
        raise ValueError("Use unique log-continuum cube indices in 0..43")
    if args.regularization and args.log_continuum_orders:
        raise ValueError("Log continua use no curvature penalty; pass --log-continuum-orders for linear-only trials")
    source_hash = digest(args.source)
    if args.source.resolve() == args.output.resolve():
        raise ValueError("Preserve the original continuum product as a separate source")
    with np.load(args.source, allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in saved.files}
    assert not arrays["artificial_injection"]
    assert not arrays["time_common_subtraction"]
    shape = arrays["observed_fixed_scale"].shape
    assert shape == (44, 79, 1848)
    assert arrays["included_mask"].all()
    old_residual = arrays["residual_fixed_scale_q5_independent"].copy()
    # Preserve the physical-stage provenance; only the continuum is revised.
    for name in ["continuum_degree", "continuum_coefficients", "continuum_fixed_scale",
                 "continuum_scope_per_frame", "fit_success", "frame_status",
                 "application_method", "baseline_model_fixed_scale"]:
        arrays[f"source_{name}"] = arrays.pop(name)
    basis, knots = cubic_spline_basis(shape[-1], spacing=args.spacing)
    arrays.update(
        application_method=np.array("positive_spline_after_frozen_spectra"),
        continuum_basis=np.array("clamped_cubic_bspline"),
        continuum_parameterization=np.array([
            "log_bspline" if order in args.log_continuum_orders else "positive_bspline"
            for order in range(shape[0])]),
        continuum_log_coefficient_bounds=np.array([-40., 20.]),
        continuum_basis_matrix=basis, continuum_knots=knots,
        continuum_degree=np.full(shape[0], 3),
        continuum_coefficient_floor=np.array(args.floor),
        continuum_regularization=np.array(args.regularization),
        continuum_refitted_after_corrections=np.array(True),
        continuum_scope_per_frame=np.full(shape[:2], "all", dtype="U11"),
        continuum_coefficients=np.empty((*shape[:2], basis.shape[1])),
        continuum_active_lower_bounds=np.zeros((*shape[:2], basis.shape[1]), bool),
        continuum_active_upper_bounds=np.zeros((*shape[:2], basis.shape[1]), bool),
        continuum_optimizer_success=np.zeros(shape[:2], bool),
        continuum_design_condition=np.empty(shape[:2]),
        q5_basis_matrix=np.empty((shape[0], shape[-1], 6)),
        continuum_fixed_scale=np.empty(shape),
        fit_success=np.zeros(shape[:2], bool),
        frame_status=np.full(shape[:2], "not_processed", dtype="U40"),
        source_combined_product_sha256=np.array(source_hash),
    )
    records = []
    for order in range(shape[0]):
        log_diagnostics = []
        for frame in range(shape[1]):
            if order in args.log_continuum_orders:
                fit = fit_log_continuum(
                    arrays["observed_fixed_scale"][order, frame],
                    arrays["full_model"][order, frame], basis,
                    initial_continuum=arrays["source_continuum_fixed_scale"][order, frame])
                if (not fit["diagnostics"]["success"] or not fit["diagnostics"]["stationary"]
                        or np.any(fit["active_mask"])):
                    raise ValueError(f"Log continuum failed stationarity or reached a numerical guard: order {order}, frame {frame}")
                arrays["continuum_active_lower_bounds"][order, frame] = fit["active_mask"] < 0
                arrays["continuum_active_upper_bounds"][order, frame] = fit["active_mask"] > 0
                log_diagnostics.append(dict(frame=frame, **fit["diagnostics"]))
            else:
                fit = fit_positive_continuum(
                    arrays["observed_fixed_scale"][order, frame],
                    arrays["full_model"][order, frame], basis,
                    coefficient_floor=args.floor,
                    **({"regularization": args.regularization} if args.regularization else {}))
                arrays["continuum_active_lower_bounds"][order, frame] = fit["coefficients"] <= args.floor
            arrays["continuum_coefficients"][order, frame] = fit["coefficients"]
            arrays["continuum_fixed_scale"][order, frame] = fit["continuum"]
            arrays["continuum_optimizer_success"][order, frame] = True
            arrays["continuum_design_condition"][order, frame] = fit["diagnostics"]["design_condition"]
        continuum = arrays["continuum_fixed_scale"][order]
        model = continuum * arrays["full_model"][order]
        observed = arrays["observed_fixed_scale"][order]
        unfiltered = observed - model
        normalized = unfiltered / continuum
        wave = arrays["wavelength_um"][order]
        coordinate = 2 * (np.log(wave) - np.log(wave).min()) / np.ptp(np.log(wave)) - 1
        q5_basis = np.polynomial.chebyshev.chebvander(coordinate, 5)
        arrays["q5_basis_matrix"][order] = q5_basis
        filtered, smooth, coefficients = project(normalized, q5_basis)
        fixed_filtered, _, fixed_coefficients = project(unfiltered, q5_basis)
        for name, value in dict(
            full_model_fixed_scale=model, observed_continuum=observed / continuum,
            residual_fixed_scale_unfiltered=unfiltered,
            residual_continuum_unfiltered=normalized, residual_continuum_q5=filtered,
            residual_fixed_scale_q5=filtered * continuum,
            residual_fixed_scale_q5_independent=fixed_filtered,
            smooth_continuum_q5=smooth, q5_continuum_coefficients=coefficients,
            q5_fixed_scale_coefficients=fixed_coefficients,
        ).items():
            assert np.isfinite(value).all(), (order, name)
            arrays[name][order] = value
        arrays["valid_math"][order] = True
        arrays["valid_fixed_scale"][order] = True
        arrays["q5_valid_frame"][order] = True
        arrays["fixed_scale_q5_valid_frame"][order] = True
        arrays["q5_valid_pixel_count"][order] = shape[-1]
        arrays["fixed_scale_q5_valid_pixel_count"][order] = shape[-1]
        arrays["fit_success"][order] = arrays["optimizer_success"][order]
        active = arrays["continuum_active_lower_bounds"][order].any(axis=1)
        arrays["frame_status"][order] = np.where(active, "continuum_bound_active", "complete")
        record = dict(
            cube_index=order, continuum_parameterization=str(arrays["continuum_parameterization"][order]),
            before_all_rms_percent=rms(old_residual[order]),
            after_all_rms_percent=rms(fixed_filtered),
            before_right128_rms_percent=rms(old_residual[order, :, -128:]),
            after_right128_rms_percent=rms(fixed_filtered[:, -128:]),
            active_bound_frames=np.flatnonzero(active).tolist(),
            minimum_continuum=float(continuum.min()),
        )
        if log_diagnostics:
            record["log_continuum_fits"] = log_diagnostics
        records.append(record)
        print(json.dumps(record), flush=True)
    # These arrays must remain byte-identical to the source scientific data/model.
    immutable = ["observed_raw", "observed_fixed_scale", "wavelength_um", "wavenumber_cm1",
                 "time_bjd", "phase", "full_model", "stellar_only", "telluric_only",
                 "nuisance_parameters", "total_calibration_rv_kms", "ip_sigma_kms"]
    with np.load(args.source, allow_pickle=False) as saved:
        for name in immutable:
            np.testing.assert_array_equal(arrays[name], saved[name])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(args.output)
    with np.load(args.output, allow_pickle=False) as saved:
        for name in immutable + ["continuum_fixed_scale", "residual_fixed_scale_q5_independent"]:
            np.testing.assert_array_equal(saved[name], arrays[name])
    assert source_hash == digest(args.source)
    metadata = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), shape=list(shape),
        source=str(args.source.resolve()), source_sha256=source_hash,
        product=str(args.output.resolve()), product_sha256=digest(args.output),
        script_sha256=digest(__file__), helper_sha256=digest(ROOT / "smooth_continuum.py"),
        artificial_injection=False, time_common_subtraction=False, telluric_division=False,
        shared_parameters_refitted=False, physical_nuisance_parameters_refitted=False,
        continuum="Positive cubic B-spline multiplying the frozen corrected spectrum; selected orders fit its logarithm instead",
        log_continuum_orders=args.log_continuum_orders,
        continuum_scope="All original pixels, fitted independently per frame in frame-p95 flux units",
        requested_spacing_pixels=args.spacing, knots=knots.tolist(),
        continuum_coefficients_per_frame=basis.shape[1], coefficient_floor=args.floor,
        curvature_regularization=args.regularization, smooth_residual_degree=5,
        residual_units="Fractions; multiply by 100 for percent. Fixed scale means each frame's original all-pixel p95 flux.",
        before_all_rms_percent=rms(old_residual),
        after_all_rms_percent=rms(arrays["residual_fixed_scale_q5_independent"]),
        before_right128_rms_percent=rms(old_residual[:, :, -128:]),
        after_right128_rms_percent=rms(arrays["residual_fixed_scale_q5_independent"][:, :, -128:]),
        immutable_arrays_verified=immutable, rows=records,
        limitations=[
            "This is a conditional continuum revision; shared and per-frame spectral parameters remain frozen.",
            "All pixels are retained, including deep absorption and negative measured flux. No temporal subtraction is performed.",
            "Active positive-coefficient floors are reported. A small positive continuum still amplifies normalized noise.",
            "Smooth fits change narrow-signal throughput, especially at order boundaries. Process CCF templates with the same continuum and Q5 response.",
            "A continuum-only response check is not a full planet injection-recovery validation of the physical fitting pipeline.",
            "The original shared fits did not meet the strict repeated-refit stabilization threshold.",
        ], seconds=time.monotonic() - started,
    )
    path = args.output.with_suffix(".json")
    temporary = path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(metadata, indent=2) + "\n")
    temporary.replace(path)
    readme = args.output.with_name(args.output.stem + "_README.txt")
    readme.write_text("\n".join([
        "IGRINS observed CCF series with revised smooth continuum", "",
        f"Shape: {shape}; axes: zero-based stored order, original frame, reversed original pixel.",
        "No artificial planet signal is injected. All original observations, wavelengths and times are unchanged.",
        "No temporal mean, common template, low-rank component or telluric division is applied.", "",
        "Use residual_fixed_scale_q5_independent for the no-division residual in frame-p95 flux units.",
        "Its definition is Q5(observed_fixed_scale - continuum_fixed_scale * full_model).",
        "Q5 removes degrees 0..5 in normalized log wavelength separately for every frame.",
        "full_model, stellar_only and telluric_only are unchanged from the source product.",
        "The smooth continuum is fitted against the frozen corrected full_model.",
        f"Stored orders {args.log_continuum_orders} use exp(B @ a): a cubic spline of log continuum, fitted in original flux units.",
        "This prevents a linear spline's endpoint zero from contaminating the full-order normalized Q5 fit.",
        "No stellar, atmospheric, line-strength, velocity or IP parameters are refitted in this revision.", "",
        "Continuum representation:",
        "  linear_predictor = continuum_coefficients @ continuum_basis_matrix.T",
        "  continuum_fixed_scale = exp(linear_predictor) for continuum_parameterization='log_bspline'; otherwise linear_predictor",
        "  continuum_knots and continuum_degree describe the clamped cubic B-spline on pixel coordinates.",
        "  continuum_coefficient_floor applies only to positive_bspline coefficients.",
        "  Log coefficients have broad numerical guards continuum_log_coefficient_bounds; an active guard rejects the export.",
        "  continuum_active_lower_bounds/continuum_active_upper_bounds record coefficient bounds.",
        "  continuum_regularization is the squared second-coefficient-difference penalty weight.",
        "  q5_basis_matrix contains the six subsequent smooth residual modes for each order.",
        "  source_continuum_* retain the previous polynomial representation and fitted continuum.",
        "  source_frame_status/source_fit_success retain the previous normalization diagnostics.",
        "  optimizer_success describes the original physical fit; continuum_optimizer_success describes this revision.",
        "  frame_status='continuum_bound_active' is a caution about identifiability, not a removed frame.", "",
        "Other residuals:",
        "  residual_fixed_scale_unfiltered = observed_fixed_scale - full_model_fixed_scale",
        "  residual_continuum_unfiltered = residual_fixed_scale_unfiltered / continuum_fixed_scale",
        "  residual_continuum_q5 = Q5(residual_continuum_unfiltered)",
        "  residual_fixed_scale_q5 = continuum_fixed_scale * residual_continuum_q5",
        "The two Q5 choices are different filters. All arrays are fractions, not percentages.", "",
        "For model sampling, use target_wavenumber_cm1[:,None,:] *",
        "(1 + total_calibration_rv_kms[:,:,None] / speed_of_light_kms).",
        "This is the fitted model coordinate, not an independently verified absolute/barycentric solution.",
        "IP response parameters and original RV metadata retain the definitions in the source README.", "",
        "CCF template response (template_flux is already sampled/IP-convolved in frame-p95 units):",
        "  from refine_continuum import filter_ccf_template  # add IGRINS to Python's import path",
        "  d = np.load('allorders_ccf_continuum_refined.npz')",
        "  processed = filter_ccf_template(template_flux, d, order=8, frame=30)",
        "This differentiates only the revised continuum and Q5 fits, with the physical model and active bounds fixed.",
        "For log continua the template response uses the exact residual-dependent Hessian, not the Gauss-Newton approximation.",
        "It is not an injection-recovery result for the original physical/stellar/telluric optimization.", "",
        *metadata["limitations"], "",
        f"Before/after full-series fixed-scale Q5 RMS: {metadata['before_all_rms_percent']:.6f}% / {metadata['after_all_rms_percent']:.6f}%.",
        f"Before/after last-128-pixel RMS: {metadata['before_right128_rms_percent']:.6f}% / {metadata['after_right128_rms_percent']:.6f}%.",
        f"Source and product hashes, settings and per-order metrics: {path.name}", "",
    ]))
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=OUT / "allorders_ccf_no_injection.npz")
    parser.add_argument("--output", type=Path, default=OUT / "allorders_ccf_continuum_refined.npz")
    parser.add_argument("--spacing", type=float, default=128.)
    parser.add_argument("--floor", type=float, default=1e-8)
    parser.add_argument("--regularization", type=float, default=0.)
    parser.add_argument("--log-continuum-orders", nargs="*", type=int, default=[39],
                        help="Fit log continuum on these stored orders to avoid endpoint collapse; default: 39")
    main(parser.parse_args())
