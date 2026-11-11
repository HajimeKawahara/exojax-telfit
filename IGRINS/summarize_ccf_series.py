"""Independently audit and combine the observed, uninjected 79-frame series."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from zipfile import BadZipFile

import numpy as np

ROOT = Path(__file__).resolve().parent
N_ORDERS, N_FRAMES, N_PIXELS = 44, 79, 1848
TRAIN_FRAMES = np.array([0, 19, 39, 59, 78])
KNOWN_FRAMES = np.array([0, 19, 39, 59, 78, 10, 30, 50, 70])
APPLICATION_METHOD = "baseline_first_frozen_corrections"
SPEED_OF_LIGHT_KMS = 299792.458
RESPONSE_UNITS = {
    "speed_of_light_kms": "Speed of light used by ExoJAX sampling, km/s",
    "wavelength_stretch_kms": "Shared linear wavenumber correction coefficient, km/s, multiplying pixel coordinate [-1,1]",
    "order_calibration_rv_kms": "Shared order sampling offset, km/s; added to nuisance_parameters[...,1]",
    "total_calibration_rv_kms": "Order plus epoch sampling offset, km/s; no stellar, barycentric or planet-frame shift",
    "target_wavenumber_cm1": "Original wavenumber times (1 + wavelength_stretch_kms*x/c), inverse cm; ascending",
    "base_ip_sigma_kms": "Shared Gaussian IP core standard deviation in velocity, km/s; not FWHM",
    "ip_sigma_kms": "Per-frame Gaussian IP core sigma: base_ip_sigma_kms * exp(nuisance_parameters[...,2]), km/s",
    "ip_wing_fraction": "Shared dimensionless mixture weight of the broad Gaussian; core weight is one minus this value",
    "ip_wing_width_ratio": "Shared broad/core Gaussian sigma ratio; two in this fiducial model",
}
COMMON = {
    "exposures": "Zero-based original frame index",
    "time_bjd": "Original CSV Time (BJD), without timescale reinterpretation",
    "phase": "Original CSV orbital phase",
    "airmass": "Dimensionless original CSV airmass",
    "rv_metadata_kms": "Original CSV RV, km/s; not an applied wavelength correction",
    "median_snr": "Dimensionless original CSV median S/N",
}
PIXEL = {
    "wavelength_um": "Microns, descending; original cube pixel axis reversed",
    "wavenumber_cm1": "Inverse cm, ascending; 10000 / wavelength_um",
    "included_mask": "Boolean; all 1848 stored pixels retained",
    "nuisance_calibration_mask": "Boolean; alternating 80-pixel blocks with five-pixel guards",
}
SPECTRAL = {
    "observed_raw": "Original cube flux units, unchanged",
    "observed_fixed_scale": "Original flux / each frame's all-pixel 95th percentile",
    "continuum_fixed_scale": "Polynomial continuum in the fixed observational scale; degree recorded per order",
    "observed_continuum": "observed_fixed_scale / continuum_fixed_scale where valid_math",
    "full_model": "Continuum-unit convolved star times transmission; excludes continuum and Q5",
    "full_model_fixed_scale": "continuum_fixed_scale * full_model",
    "baseline_model": "Uncorrected physical star-times-transmission model used for frame nuisance and continuum fits",
    "baseline_model_fixed_scale": "continuum_fixed_scale * baseline_model; before frozen stellar/telluric corrections",
    "stellar_only": "Continuum-unit stellar model convolved with the IP",
    "telluric_only": "Dimensionless transmission convolved with the IP",
    "residual_fixed_scale_unfiltered": "observed_fixed_scale - full_model_fixed_scale",
    "residual_continuum_unfiltered": "residual_fixed_scale_unfiltered / continuum_fixed_scale where valid_math",
    "residual_continuum_q5": "Q5(residual_continuum_unfiltered); fraction of continuum, not percent",
    "residual_fixed_scale_q5": "continuum_fixed_scale * residual_continuum_q5; same continuum-unit projection",
    "residual_fixed_scale_q5_independent": "Q5(residual_fixed_scale_unfiltered); separate projection without continuum division",
    "smooth_continuum_q5": "Degree-five fitted smooth residual in continuum units",
    "valid_math": "Boolean; finite fixed-scale residual and finite strictly positive continuum give finite normalization",
    "valid_fixed_scale": "Boolean; finite fixed-scale residual, independently of continuum positivity",
}
FRAME = {
    "normalization_scalar": "95th percentile of all 1848 original observed samples; original cube units",
    "physical_normalization_scalar": "Original physical-fit normalization: all-pixel p95 for training five, calibration-block p95 otherwise",
    "continuum_scale_conversion": "physical_normalization_scalar / normalization_scalar; continuum unit conversion only",
    "reused_physical_frame": "Boolean; original five training and four evaluation frames reuse saved nuisance parameters and continuum",
    "nuisance_optimization_executed": "Boolean; true only for the additional 70 baseline-first frame fits",
    "nuisance_source": "Provenance of frame nuisance parameters and continuum",
    "continuum_scope_per_frame": "Pixels used in original baseline continuum fit: all for training five, adopted held-out scope otherwise",
    "nuisance_scope_per_frame": "Pixels used in original baseline nuisance fit: all for training five, calibration blocks otherwise",
    "optimizer_success": "Boolean; optimizer termination flag, independent of mathematical validity",
    "fit_success": "Boolean; optimizer_success and all valid_math and q5_valid_frame",
    "q5_valid_frame": "Boolean; continuum-unit Q5 has six independent fitted modes",
    "q5_valid_pixel_count": "Number of valid_math pixels, not a residual or absorption selection",
    "fixed_scale_q5_valid_frame": "Boolean; independent fixed-scale Q5 is mathematically defined",
    "fixed_scale_q5_valid_pixel_count": "Number of finite fixed-scale residual samples",
    "frame_status": "Explicit producer status; failed or undefined frames remain present",
}
COEFFICIENT = {
    "nuisance_initial_parameters": (3, "Initial [log10_water, epoch RV km/s, natural-log IP-width correction]"),
    "nuisance_parameters": (3, "Fitted [log10_water, epoch RV km/s, natural-log IP-width correction]"),
    "continuum_coefficients": (None, "Chebyshev coefficients on linear pixel coordinate [-1,1]; combined artifact zero-pads beyond continuum_degree"),
    "q5_continuum_coefficients": (6, "Chebyshev coefficients on normalized log wavelength [-1,1]"),
    "q5_fixed_scale_coefficients": (6, "Independent fixed-scale Chebyshev coefficients on normalized log wavelength [-1,1]"),
}
BOOLEAN = {"included_mask", "nuisance_calibration_mask", "valid_math", "valid_fixed_scale", "optimizer_success",
           "fit_success", "q5_valid_frame", "fixed_scale_q5_valid_frame", "reused_physical_frame",
           "nuisance_optimization_executed"}
NO_ACTION_FLAGS = ("artificial_injection", "time_common_subtraction", "shared_parameters_refitted",
                   "telluric_division", "enuma_used", "new_absorption_or_residual_masks")


def digest(path):
    with Path(path).open("rb") as stream:
        checksum = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def close(actual, expected, label, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=atol, equal_nan=True,
                               err_msg=label)


def finite_metrics(values):
    valid = np.isfinite(values)
    count = int(valid.sum())
    total = float(np.sum(values[valid] ** 2))
    return dict(finite_samples=count, undefined_samples=int(values.size-count), sum_squared=total,
                rms_percent=100 * np.sqrt(total/count) if count else None)


def project_q5(values, basis, valid):
    """Use QR independently of the producer's least-squares implementation."""
    result = np.full(values.shape, np.nan)
    smooth = np.full(values.shape, np.nan)
    coefficients = np.full(6, np.nan)
    if valid.sum() < 6 or np.linalg.matrix_rank(basis[valid]) != 6:
        return result, smooth, coefficients, False
    orthogonal, triangular = np.linalg.qr(basis[valid], mode="reduced")
    coefficients = np.linalg.solve(triangular, orthogonal.T @ values[valid])
    smooth = basis @ coefficients
    result[valid] = values[valid] - smooth[valid]
    return result, smooth, coefficients, True


def read_info(path):
    with path.open() as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == N_FRAMES, "Expected all 79 metadata rows"
    columns = dict(time_bjd="Time (BJD)", phase="Phase", airmass="Air Mass",
                   rv_metadata_kms="RV [km/s]", median_snr="Med. SNR")
    result = {name: np.array([float(record[column]) for record in records])
              for name, column in columns.items()}
    result["exposures"] = np.arange(N_FRAMES)
    return result


def read_product(path, order, original_wave, original_flux, info, input_hashes, root=ROOT):
    """Audit original data, frozen inputs, residual identities and every frame."""
    metadata_path = path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text())
    product_hash = digest(path)
    assert metadata["product_sha256"] == product_hash, "Series product hash mismatch"
    for flag in NO_ACTION_FLAGS:
        assert metadata[flag] is False, f"Required explicit false provenance flag: {flag}"
    assert metadata["application_method"] == APPLICATION_METHOD, "Expected the canonical baseline-first application"
    assert metadata["corrections_applied_after_calibration"] is True
    assert metadata["cube_index"] == order
    assert metadata["total_frames"] == N_FRAMES
    assert metadata["retained_wavelength_pixels"] == N_PIXELS
    assert metadata["continuum_scope"] in {"all", "calibration"}
    degree = metadata["continuum_degree"]
    assert type(degree) is int and 0 <= degree < N_PIXELS, "Invalid declared continuum degree"
    assert metadata["q5_degree"] == 5
    assert metadata["nuisance_parameter_names"] == ["log10_water", "calibration_rv_epoch_kms", "log_ip_width_epoch"]
    for key, checksum in input_hashes.items():
        assert metadata["input_sha256"][key] == checksum, f"Original input changed: {key}"
    sources = {}
    for name in ["source_fit_product", "source_fit_metadata"]:
        relative = Path(metadata[name])
        assert not relative.is_absolute() and ".." not in relative.parts, "Expected a local relative source path"
        source = root / relative
        assert digest(source) == metadata[f"{name}_sha256"], f"Frozen input changed: {name}"
        sources[name] = source
    source_metadata = json.loads(sources["source_fit_metadata"].read_text())
    assert source_metadata["product_sha256"] == metadata["source_fit_product_sha256"]
    physical_sources, physical_hashes = {}, {}
    for frame in KNOWN_FRAMES[5:]:
        physical_path = sources["source_fit_product"].parent / f"{metadata['physical_prefix']}{frame}.json"
        for input_path in [physical_path, physical_path.with_suffix(".npz")]:
            key = str(input_path.relative_to(root))
            physical_hashes[key] = digest(input_path)
            assert source_metadata["source_sha256"][key] == physical_hashes[key], f"Frozen physical input changed: {key}"
        physical = json.loads(physical_path.read_text())
        assert physical["continuum_degree"] == degree
        assert physical.get("continuum_scope", "calibration") == metadata["continuum_scope"]
        physical_sources[int(frame)] = physical
    response_keys = dict(wavelength_stretch_kms=f"stretch_o{order}_kms",
                         order_calibration_rv_kms=f"calibration_rv_o{order}_kms",
                         base_ip_sigma_kms=f"ip_sigma_o{order}_kms",
                         ip_wing_fraction="ip_wing_fraction")
    response_parameters = {name: physical_sources[10]["parameters"][key]
                           for name, key in response_keys.items()}
    response_parameters["ip_wing_width_ratio"] = physical_sources[10]["ip_wing_width_ratio"]
    assert all(np.isfinite(value) for value in response_parameters.values())
    assert response_parameters["base_ip_sigma_kms"] > 0
    assert 0 <= response_parameters["ip_wing_fraction"] <= 1
    assert response_parameters["ip_wing_width_ratio"] == 2
    for physical in physical_sources.values():
        assert physical["ip"] == "mixture"
        assert not any(physical.get(key, False) for key in
                       ["quadratic_wavelength", "direct_ip", "ip_asymmetry", "ip_wavelength"])
        for name, key in response_keys.items():
            assert physical["parameters"][key] == response_parameters[name], f"Shared response changed: {key}"
        assert physical["ip_wing_width_ratio"] == response_parameters["ip_wing_width_ratio"]
    with np.load(sources["source_fit_product"], allow_pickle=False) as source:
        np.testing.assert_array_equal(source["orders"], [order])
        np.testing.assert_array_equal(source["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
        np.testing.assert_array_equal(source["wavelength_um"][0], original_wave[::-1])
        assert source["included_mask"].all()
        known_reference = {key: source[key][0] for key in ["component", "component_baseline", "continuum", "y_over_c",
                                                           "residual_unfiltered", "residual_filtered"]}
    with np.load(path, allow_pickle=False) as saved:
        arrays = {key: saved[key] for key in saved.files}
    assert arrays["cube_index"].shape == () and int(arrays["cube_index"]) == order
    expected_shapes = {name: (N_FRAMES,) for name in COMMON}
    expected_shapes.update({name: (N_PIXELS,) for name in PIXEL})
    expected_shapes.update({name: (N_FRAMES, N_PIXELS) for name in SPECTRAL})
    expected_shapes.update({name: (N_FRAMES,) for name in FRAME})
    expected_shapes.update({name: (N_FRAMES, degree + 1 if name == "continuum_coefficients" else width)
                            for name, (width, _) in COEFFICIENT.items()})
    for name, shape in expected_shapes.items():
        assert arrays[name].shape == shape, f"Unexpected shape: {name}"
        if name in BOOLEAN:
            assert arrays[name].dtype == bool, f"Expected a Boolean array: {name}"
        if np.issubdtype(arrays[name].dtype, np.number):
            assert not np.isinf(arrays[name]).any(), f"Infinite values in {name}"
    for name in COMMON:
        if name == "exposures":
            np.testing.assert_array_equal(arrays[name], info[name])
        else:
            # pandas and Python's float parser can differ by a few ULPs.
            tolerance = 4 * np.finfo(float).eps * max(1., abs(info[name]).max())
            np.testing.assert_allclose(arrays[name], info[name], rtol=0, atol=tolerance,
                                       err_msg=f"Changed original metadata: {name}")
    wave = arrays["wavelength_um"]
    np.testing.assert_array_equal(wave, original_wave[::-1])
    assert np.all(np.diff(wave) < 0)
    close(arrays["wavenumber_cm1"], 1e4 / wave, "Wavenumber conversion")
    np.testing.assert_array_equal(arrays["observed_raw"], original_flux[:, ::-1], err_msg="Observed raw data were modified")
    assert np.isfinite(arrays["observed_raw"]).all() and arrays["included_mask"].all()
    scalar = np.percentile(original_flux[:, ::-1], 95, axis=1)
    np.testing.assert_array_equal(arrays["normalization_scalar"], scalar)
    assert np.all(scalar > 0) and np.isfinite(scalar).all()
    observed = arrays["observed_fixed_scale"]
    close(observed, arrays["observed_raw"] / scalar[:, None], "Fixed observational normalization")
    pixel = np.arange(N_PIXELS)
    calibration = ((pixel // 80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    np.testing.assert_array_equal(arrays["nuisance_calibration_mask"], calibration)
    training = np.isin(np.arange(N_FRAMES), TRAIN_FRAMES)
    reused = np.isin(np.arange(N_FRAMES), KNOWN_FRAMES)
    np.testing.assert_array_equal(arrays["reused_physical_frame"], reused)
    np.testing.assert_array_equal(arrays["nuisance_optimization_executed"], ~reused)
    physical_scalar = np.percentile(arrays["observed_raw"][:, calibration], 95, axis=1)
    physical_scalar[training] = scalar[training]
    assert np.isfinite(physical_scalar).all() and (physical_scalar > 0).all()
    np.testing.assert_array_equal(arrays["physical_normalization_scalar"], physical_scalar)
    close(arrays["continuum_scale_conversion"], physical_scalar / scalar, "Physical-to-output continuum scale conversion")
    expected_continuum_scope = np.where(training, "all", metadata["continuum_scope"])
    np.testing.assert_array_equal(arrays["continuum_scope_per_frame"], expected_continuum_scope)
    expected_nuisance_scope = np.where(training, "all", "calibration")
    np.testing.assert_array_equal(arrays["nuisance_scope_per_frame"], expected_nuisance_scope)
    expected_nuisance_source = np.where(training, "saved_training", np.where(reused, "saved_heldout", "new_heldout_fit"))
    np.testing.assert_array_equal(arrays["nuisance_source"], expected_nuisance_source)
    continuum, model = arrays["continuum_fixed_scale"], arrays["full_model"]
    for name in ["nuisance_parameters", "nuisance_initial_parameters"]:
        assert np.isfinite(arrays[name]).all(), f"Nonfinite nuisance parameters: {name}"
    for name in ["full_model", "stellar_only", "telluric_only"]:
        finite = arrays[name][np.isfinite(arrays[name])]
        assert finite.min(initial=0) >= -1e-10, f"Negative physical model: {name}"
        if name == "telluric_only":
            assert finite.max(initial=0) <= 1 + 1e-10
    with np.errstate(over="ignore", invalid="ignore"):
        fixed_model = continuum * model
    fixed_model[~np.isfinite(fixed_model)] = np.nan
    close(arrays["full_model_fixed_scale"], fixed_model, "Full model scale")
    close(arrays["baseline_model_fixed_scale"], continuum * arrays["baseline_model"], "Baseline model scale")
    fixed_residual = observed - fixed_model
    close(arrays["residual_fixed_scale_unfiltered"], fixed_residual, "Signed data-minus-model residual")
    fixed_valid_all = np.isfinite(fixed_residual)
    np.testing.assert_array_equal(arrays["valid_fixed_scale"], fixed_valid_all)
    valid = np.isfinite(continuum) & (continuum > 0) & fixed_valid_all
    normalized, normalized_residual = np.full(observed.shape, np.nan), np.full(observed.shape, np.nan)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        normalized[valid] = observed[valid] / continuum[valid]
        normalized_residual[valid] = fixed_residual[valid] / continuum[valid]
    valid &= np.isfinite(normalized) & np.isfinite(normalized_residual)
    normalized[~valid], normalized_residual[~valid] = np.nan, np.nan
    np.testing.assert_array_equal(arrays["valid_math"], valid)
    close(arrays["observed_continuum"], normalized, "Continuum normalization and undefined samples")
    close(arrays["residual_continuum_unfiltered"], normalized_residual, "Continuum-unit signed residual")
    lower, upper = np.asarray(metadata["nuisance_bounds"])
    np.testing.assert_array_equal(lower, [-1.5, -2., -.15])
    np.testing.assert_array_equal(upper, [1.5, 2., .15])
    assert np.all(arrays["nuisance_parameters"] >= lower - 1e-10)
    assert np.all(arrays["nuisance_parameters"] <= upper + 1e-10)
    np.testing.assert_array_equal(arrays["nuisance_parameters"][0, 1:], [0., 0.])
    continuum_basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, N_PIXELS), degree)
    close(continuum, arrays["continuum_coefficients"] @ continuum_basis.T, "Declared-degree continuum reconstruction")
    known_errors = {}
    for key, reference_key in [("full_model", "component"), ("baseline_model", "component_baseline"),
                               ("observed_continuum", "y_over_c"), ("residual_continuum_unfiltered", "residual_unfiltered"),
                               ("residual_continuum_q5", "residual_filtered")]:
        actual, expected = arrays[key][KNOWN_FRAMES], known_reference[reference_key]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10,
                                   err_msg=f"The original nine-frame fiducial product changed: {key}")
        known_errors[key] = float(np.max(abs(actual - expected)))
    expected_known_continuum = known_reference["continuum"] * arrays["continuum_scale_conversion"][KNOWN_FRAMES, None]
    close(continuum[KNOWN_FRAMES], expected_known_continuum, "Reuse of saved nine-frame continuum after scale conversion")
    for frame in KNOWN_FRAMES:
        physical = physical_sources[10 if frame in TRAIN_FRAMES else int(frame)]
        parameters = physical["parameters"]
        expected = [parameters[f"log10_water_e{frame}"], parameters.get(f"calibration_rv_e{frame}_kms", 0.),
                    parameters.get(f"log_ip_width_e{frame}", 0.)]
        np.testing.assert_array_equal(arrays["nuisance_parameters"][frame], expected)
    baseline_continuum_error = []
    for frame in np.flatnonzero(~reused):
        weight = np.ones(N_PIXELS) if expected_continuum_scope[frame] == "all" else calibration.astype(float)
        design = arrays["baseline_model"][frame, :, None] * continuum_basis
        physical_y = arrays["observed_raw"][frame] / physical_scalar[frame]
        coefficients = np.linalg.solve(design.T @ (weight[:, None] * design) + 1e-12 * np.eye(degree + 1),
                                       design.T @ (weight * physical_y))
        expected = continuum_basis @ coefficients * arrays["continuum_scale_conversion"][frame]
        close(continuum[frame], expected, f"Frame {frame} continuum was fitted to the physical baseline", atol=1e-8)
        difference = abs(continuum[frame] - expected)
        if np.isfinite(difference).any():
            baseline_continuum_error.append(float(difference[np.isfinite(difference)].max()))
    x = 2 * (np.log(wave) - np.log(wave).min()) / np.ptp(np.log(wave)) - 1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    assert len(metadata["frames"]) == N_FRAMES
    q5_errors = []
    for frame in range(N_FRAMES):
        residual, smooth, coefficients, defined = project_q5(normalized_residual[frame], basis, valid[frame])
        assert bool(arrays["q5_valid_frame"][frame]) == defined
        assert arrays["q5_valid_pixel_count"][frame] == valid[frame].sum()
        close(arrays["residual_continuum_q5"][frame], residual, f"Frame {frame} continuum Q5", atol=1e-9)
        close(arrays["smooth_continuum_q5"][frame], smooth, f"Frame {frame} smooth modes", atol=1e-9)
        close(arrays["q5_continuum_coefficients"][frame], coefficients, f"Frame {frame} Q5 coefficients", atol=1e-9)
        close(arrays["residual_fixed_scale_q5"][frame], residual * continuum[frame], f"Frame {frame} Q5 rescaling", atol=1e-9)
        fixed_valid = np.isfinite(fixed_residual[frame])
        independent, _, fixed_coefficients, fixed_defined = project_q5(fixed_residual[frame], basis, fixed_valid)
        assert bool(arrays["fixed_scale_q5_valid_frame"][frame]) == fixed_defined
        assert arrays["fixed_scale_q5_valid_pixel_count"][frame] == fixed_valid.sum()
        close(arrays["residual_fixed_scale_q5_independent"][frame], independent, f"Frame {frame} independent fixed-scale Q5")
        close(arrays["q5_fixed_scale_coefficients"][frame], fixed_coefficients, f"Frame {frame} fixed-scale Q5 coefficients")
        success = bool(arrays["optimizer_success"][frame]) and bool(valid[frame].all()) and defined
        assert bool(arrays["fit_success"][frame]) == success
        record = metadata["frames"][frame]
        assert record["frame"] == frame and record["optimizer_success"] == bool(arrays["optimizer_success"][frame])
        assert record["status"] == arrays["frame_status"][frame]
        assert record["valid_math_pixels"] == valid[frame].sum()
        assert record["valid_fixed_scale_pixels"] == fixed_valid.sum()
        assert record["q5_valid_frame"] is defined
        assert record["fixed_scale_q5_valid_frame"] is fixed_defined
        assert record["optimizer_executed"] == bool(~reused[frame])
        assert record["nuisance_source"] == expected_nuisance_source[frame]
        assert record["continuum_scope"] == expected_continuum_scope[frame]
        assert record["nuisance_scope"] == expected_nuisance_scope[frame]
        if reused[frame]:
            physical = physical_sources[10 if training[frame] else frame]
            original_fit = physical["training_fit" if training[frame] else "heldout_calibration_fit"]
            assert record["optimizer_success"] == original_fit["success"]
            assert record["nfev"] == 0
        if success:
            assert record["status"] == ("reused_physical_fit" if reused[frame] else "success")
        elif not fixed_defined:
            assert record["status"] == "undefined_fixed_scale_q5"
        elif not defined:
            assert record["status"] == "undefined_q5_projection"
        elif not valid[frame].all():
            assert record["status"] == "undefined_normalization"
        else:
            assert record["status"] in {"optimizer_exception", "optimizer_not_converged", "source_optimizer_not_converged"}
        expected_rms = finite_metrics(residual)["rms_percent"]
        if expected_rms is None:
            assert record["continuum_q5_rms_percent"] is None
        else:
            close(record["continuum_q5_rms_percent"], expected_rms, f"Frame {frame} reported RMS", atol=1e-8)
        if defined:
            q5_errors.append(float(np.max(abs(arrays["residual_continuum_q5"][frame, valid[frame]] - residual[valid[frame]]))))
    assert metadata["frames_with_successful_fit"] == int(arrays["fit_success"].sum())
    assert metadata["frames_with_optimizer_success"] == int(arrays["optimizer_success"].sum())
    assert metadata["undefined_normalization_samples"] == int((~valid).sum())
    assert metadata["undefined_fixed_scale_samples"] == int((~fixed_valid_all).sum())
    assert metadata["reused_physical_frames"] == KNOWN_FRAMES.tolist()
    assert metadata["newly_calibrated_frames"] == 70
    residual_keys = [key for key in SPECTRAL if key.startswith("residual_")]
    metrics = {key: finite_metrics(arrays[key]) for key in residual_keys}
    frame_rms = {key: [finite_metrics(values)["rms_percent"] for values in arrays[key]]
                 for key in residual_keys}
    positive = continuum[valid]
    row = dict(cube_index=order, status="complete" if arrays["fit_success"].all() else "complete_with_flagged_frames",
               product_path=str(path), product_sha256=product_hash, metadata_sha256=digest(metadata_path),
               source_fit_product=str(sources["source_fit_product"]), source_fit_product_sha256=metadata["source_fit_product_sha256"],
               source_fit_metadata=str(sources["source_fit_metadata"]), source_fit_metadata_sha256=metadata["source_fit_metadata_sha256"],
               application_code_sha256=metadata["application_code_sha256"],
               frozen_nine_frame_reconstruction_max_error=metadata["frozen_nine_frame_reconstruction_max_error"],
               known_nine_frame_max_errors=known_errors,
               baseline_continuum_reconstruction_max_error=max(baseline_continuum_error, default=0.),
               source_physical_sha256=physical_hashes, application_method=APPLICATION_METHOD,
               response_parameters=response_parameters,
               wavelength_min_um=float(wave.min()), wavelength_max_um=float(wave.max()),
               frames=N_FRAMES, included_wavelength_pixels=N_PIXELS,
               optimizer_failed_frames=np.flatnonzero(~arrays["optimizer_success"]).tolist(),
               flagged_frames=np.flatnonzero(~arrays["fit_success"]).tolist(),
               undefined_normalization_samples=int((~valid).sum()),
               undefined_fixed_scale_samples=int((~fixed_valid_all).sum()),
               nonpositive_observed_samples=int(np.count_nonzero(arrays["observed_raw"] <= 0)),
               continuum_minimum=float(continuum[np.isfinite(continuum)].min()) if np.isfinite(continuum).any() else None,
               continuum_minimum_positive=float(positive.min()) if positive.size else None,
               frame_status_counts=dict(Counter(arrays["frame_status"].tolist())),
               q5_reconstruction_max_error=max(q5_errors, default=None), metrics=metrics,
               per_frame_rms_percent=frame_rms,
               per_frame_undefined_normalization_samples=(~valid).sum(axis=1).tolist(),
               conditional_fit=True, reused_physical_frames=KNOWN_FRAMES.tolist(), new_conditional_frames=70,
               continuum_scope=metadata["continuum_scope"], continuum_degree=degree,
               continuum_degree_deviation=degree != 3, nuisance_scope=metadata["nuisance_scope"],
               source_numerically_stabilized=source_metadata.get("numerically_stabilized", False))
    return row


def combine_products(rows, target, info):
    """Preserve all frames and flags, writing only a fully audited 44-order set."""
    blocked = [row["cube_index"] for row in rows
               if row["status"] not in {"complete", "complete_with_flagged_frames"}]
    result = dict(created=False, path=str(target), blocked_orders=blocked,
                  existing_file_unchanged=target.exists())
    if len(rows) != N_ORDERS or blocked:
        result["reason"] = "All 44 products must exist and pass independent audits; flagged frames are retained, but missing or invalid products prevent combination."
        return result
    combined = {"orders": np.arange(N_ORDERS), **info,
                "artificial_injection": np.array(False), "time_common_subtraction": np.array(False),
                "telluric_division": np.array(False), "shared_parameters_refitted": np.array(False),
                "application_method": np.array(APPLICATION_METHOD),
                "corrections_applied_after_calibration": np.array(True)}
    combined["continuum_degree"] = np.array([row["continuum_degree"] for row in rows])
    maximum_degree = int(combined["continuum_degree"].max())
    combined["continuum_coefficients"] = np.zeros((N_ORDERS, N_FRAMES, maximum_degree + 1))
    keys = [*PIXEL, *SPECTRAL, *FRAME, *COEFFICIENT]
    for order, row in enumerate(rows):
        assert row["cube_index"] == order
        path = Path(row["product_path"])
        for name, source in [("product_sha256", path), ("metadata_sha256", path.with_suffix(".json")),
                             ("source_fit_product_sha256", Path(row["source_fit_product"])),
                             ("source_fit_metadata_sha256", Path(row["source_fit_metadata"]))]:
            if digest(source) != row[name]:
                result.update(blocked_orders=[order], reason="An input changed after its independent audit.")
                return result
        with np.load(path, allow_pickle=False) as source:
            for key in COMMON:
                if order == 0:
                    combined[key] = source[key]
                else:
                    np.testing.assert_array_equal(source[key], combined[key])
            for key in keys:
                value = source[key]
                if key == "continuum_coefficients":
                    assert value.shape[-1] == row["continuum_degree"] + 1
                    combined[key][order, :, :value.shape[-1]] = value
                    continue
                if order == 0:
                    combined[key] = np.empty((N_ORDERS, *value.shape), dtype=value.dtype)
                combined[key][order] = value
    combined["source_fit_product_sha256"] = np.array([row["source_fit_product_sha256"] for row in rows])
    combined["source_fit_metadata_sha256"] = np.array([row["source_fit_metadata_sha256"] for row in rows])
    combined["source_series_product_sha256"] = np.array([row["product_sha256"] for row in rows])
    combined["speed_of_light_kms"] = np.array(SPEED_OF_LIGHT_KMS)
    for name in rows[0]["response_parameters"]:
        combined[name] = np.array([row["response_parameters"][name] for row in rows])
    pixel_coordinate = np.linspace(-1, 1, N_PIXELS)
    combined["target_wavenumber_cm1"] = (1e4 / combined["wavelength_um"]) * (
        1 + combined["wavelength_stretch_kms"][:, None] * pixel_coordinate / SPEED_OF_LIGHT_KMS)
    combined["total_calibration_rv_kms"] = combined["order_calibration_rv_kms"][:, None] + combined["nuisance_parameters"][:, :, 1]
    combined["ip_sigma_kms"] = combined["base_ip_sigma_kms"][:, None] * np.exp(combined["nuisance_parameters"][:, :, 2])
    assert np.all(np.diff(combined["target_wavenumber_cm1"], axis=1) > 0)
    assert (combined["ip_sigma_kms"] > 0).all()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".allorders_ccf_no_injection.", suffix=".npz",
                                         dir=target.parent, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **combined)
        with np.load(temporary, allow_pickle=False) as saved:
            assert set(saved.files) == set(combined)
            for name, values in combined.items():
                np.testing.assert_array_equal(saved[name], values)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result.update(created=True, sha256=digest(target), existing_file_unchanged=False,
                  shapes={name: list(value.shape) for name, value in combined.items()},
                  continuum_degree=combined["continuum_degree"].tolist(),
                  continuum_coefficient_padding="Zero coefficients beyond each order's declared degree; all other arrays retain their original values and shapes.",
                  reason="All 44 observed series passed independent audits. Any flagged frames and undefined continuum-unit samples remain explicitly represented.")
    return result


def plot_overview(rows, path):
    """Show per-frame residual scale and flags without changing any samples."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    keys = ["residual_continuum_q5", "residual_fixed_scale_q5", "residual_fixed_scale_q5_independent"]
    labels = ["Continuum-unit Q5", "Same continuum Q5, rescaled to frame-p95 flux units",
              "Independent Q5 in frame-p95 flux units; no continuum division"]
    colorbar_labels = ["RMS (% of fitted continuum)", "RMS (% of frame-p95 flux)",
                       "RMS (% of frame-p95 flux)"]
    matrices = {key: np.full((N_ORDERS, N_FRAMES), np.nan) for key in keys}
    failed, undefined = [], []
    for row in rows:
        order = row["cube_index"]
        if "metrics" not in row:
            continue
        for key in keys:
            matrices[key][order] = row["per_frame_rms_percent"][key]
        failed += [(frame, order) for frame in row["optimizer_failed_frames"]]
        undefined += [(frame, order) for frame, count in enumerate(row["per_frame_undefined_normalization_samples"]) if count]
    if not any(np.any(np.isfinite(value) & (value > 0)) for value in matrices.values()):
        return
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(".85")
    fig, axes = plt.subplots(3, 1, figsize=(13, 13), sharex=True, sharey=True)
    fig.subplots_adjust(top=.93, bottom=.10, left=.07, right=.91, hspace=.25)
    for ax, key, label, units in zip(axes, keys, labels, colorbar_labels):
        positive = matrices[key][np.isfinite(matrices[key]) & (matrices[key] > 0)]
        low, high = (positive.min(), positive.max()) if positive.size else (1., 10.)
        norm = LogNorm(vmin=low, vmax=max(high, low * 1.01))
        artist = ax.imshow(matrices[key], origin="lower", aspect="auto", interpolation="nearest",
                           cmap=cmap, norm=norm, extent=(-.5, N_FRAMES-.5, -.5, N_ORDERS-.5))
        colorbar = fig.colorbar(artist, ax=ax, fraction=.025, pad=.02)
        ticks = np.geomspace(norm.vmin, norm.vmax, 5)
        colorbar.set_ticks(ticks)
        colorbar.set_ticklabels([f"{value:.3g}" for value in ticks])
        colorbar.set_label(units + "; independent log scale", fontsize=9)
        if failed:
            ax.scatter(*np.array(failed).T, s=10, marker="x", color="red", linewidths=.6)
        if undefined:
            ax.scatter(*np.array(undefined).T, s=14, marker="s", facecolors="none", edgecolors="white", linewidths=.65)
        ax.set(title=label, ylabel="Stored cube index", yticks=np.arange(0, N_ORDERS, 5))
    axes[-1].set_xlabel("Original frame index; all 79 frames retained")
    fig.suptitle("IGRINS observed residual series: no artificial planet injection", weight="bold")
    fig.text(.5, .02, "Each panel has its own logarithmic color limits; no values are clipped. Compare numbers, not colors, across panels.\n"
             "Red crosses: optimizer failure. White squares: undefined continuum normalization. Gray: unavailable/undefined RMS.\n"
             "All samples remain in the products. These are conditional fitted residuals; no temporal common-mode subtraction or telluric division.",
             ha="center", fontsize=9)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    with args.cube.open("rb") as stream:
        original_wave, original_flux = pickle.load(stream)
    assert original_wave.shape == (N_ORDERS, N_PIXELS)
    assert original_flux.shape == (N_ORDERS, N_FRAMES, N_PIXELS)
    info = read_info(args.info)
    input_hashes = {"data/cube14v3.pic": digest(args.cube), "data/20201214_info.csv": digest(args.info)}
    rows = []
    for order in range(N_ORDERS):
        path = out / f"{args.prefix}_o{order:02d}_ccf_no_injection.npz"
        row = dict(cube_index=order, status="missing", message="Completed NPZ/JSON pair is not available.")
        if path.exists() and path.with_suffix(".json").exists():
            try:
                row = read_product(path, order, original_wave[order], original_flux[order], info,
                                   input_hashes, args.source_root)
            except (AssertionError, ValueError, KeyError, OSError, BadZipFile, json.JSONDecodeError) as error:
                row.update(status="invalid_product", message=f"{type(error).__name__}: {error}")
        rows.append(row)
        if row["status"] != "missing":
            print(json.dumps({key: row[key] for key in ["cube_index", "status"]}), flush=True)
    pooled = {}
    for row in rows:
        for name, metrics in row.get("metrics", {}).items():
            accumulator = pooled.setdefault(name, dict(finite_samples=0, undefined_samples=0, sum_squared=0.))
            for key in accumulator:
                accumulator[key] += metrics[key]
    for metrics in pooled.values():
        count = metrics["finite_samples"]
        metrics["rms_percent"] = 100 * np.sqrt(metrics["sum_squared"] / count) if count else None
    combined = (combine_products(rows, out / f"{args.prefix}_ccf_no_injection.npz", info)
                if args.combine else dict(created=False, requested=False, reason="Combination disabled by --no-combine."))
    units = {**COMMON, **PIXEL, **SPECTRAL, **FRAME, **RESPONSE_UNITS,
             **{name: description for name, (_, description) in COEFFICIENT.items()}}
    units["continuum_degree"] = "Per-order continuum polynomial degree; cubic fiducial unless explicitly flagged"
    units["application_method"] = "baseline_first_frozen_corrections; physical calibration precedes fixed stellar/telluric corrections"
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), expected_shape=[N_ORDERS, N_FRAMES, N_PIXELS],
                    status_counts=dict(Counter(row["status"] for row in rows)), rows=rows,
                    all_products_audited=all("metrics" in row for row in rows), combined_product=combined,
                    artificial_injection=False, time_common_subtraction=False, telluric_division=False,
                    shared_parameters_refitted=False, enuma_used=False,
                    application_method=APPLICATION_METHOD, corrections_applied_after_calibration=True,
                    model_sampling_coordinates="nu_eff = target_wavenumber_cm1[:,None,:] * (1 + total_calibration_rv_kms[:,:,None] / speed_of_light_kms). These are the interpolation coordinates used by the fitted model, not an independently verified absolute wavelength solution. Original wavelength arrays and residual samples are unchanged; no resampling or additional stellar, barycentric or planet-frame correction is applied.",
                    model_sampling_target="target_nu = (1e4 / wavelength_um) * (1 + wavelength_stretch_kms[:,None] * linspace(-1,1,1848) / c). ExoJAX sampling interpolates the convolved spectrum at target_nu*(1+RV/c); retain the product of factors rather than replacing it with an approximate summed velocity.",
                    ip_response="At each frame, convolve the intrinsic spectrum on the model velocity grid with Gaussian sigmas ip_sigma_kms and ip_wing_width_ratio*ip_sigma_kms, sample both at nu_eff, and mix with weights 1-ip_wing_fraction and ip_wing_fraction. Stellar broadening is a separate operation and is not included in these IP sigmas.",
                    signed_residual="Observed minus full stellar-times-terrestrial model; absorption signals are negative.",
                    transmission="Telluric transmission is retained in the observed residual signal and is never divided out. Separate convolved stellar/telluric components do not multiply to full_model.",
                    q5="Six smooth modes, degrees zero through five, projected separately for each frame in normalized log wavelength. No temporal common-mode subtraction.",
                    normalization_validity="Undefined continuum normalization is represented by NaN and valid_math=False. Original and fixed-scale samples remain present; negative observed flux is retained.",
                    conditional_fit="Shared corrections use frames 0,19,39,59,78. These five training and four evaluation physical fits are reused without optimization. Only the other 70 frames receive new water, wavelength offset, IP-width and continuum fits against the uncorrected physical baseline. Frozen stellar/telluric corrections and Q5 are applied afterward with the baseline continuum fixed. Reported RMS is not an independent 79-frame prediction test.",
                    baseline_fit_scope="Saved training fits use all pixels. Saved held-out and new frame nuisance fits use calibration blocks; their continuum uses the adopted physical scope. Each frame records its scopes and whether optimization was executed. Continuum degree follows the saved physical fit.",
                    physical_output_normalization="Physical fits use all-pixel p95 for training and calibration-block p95 otherwise. Output observed_fixed_scale always uses all-pixel p95. The saved/profiled continuum is multiplied by physical_p95/output_p95; this changes units only. The original nine component, normalized-data and residual arrays agree with their saved fiducial product within 1e-10 absolute tolerance, with measured differences reported per order.",
                    continuum_degree_exceptions={str(row["cube_index"]): row["continuum_degree"] for row in rows if row.get("continuum_degree_deviation")},
                    unchanged_benchmark="The separate nine-frame validation products and report are unchanged.",
                    axis_order="Order, exposure, stored pixel reversed for spectral arrays; common time/metadata axes have length 79.",
                    order_numbering="Released cube indices 0..43; only stored index 5 is identified as user Order 7.",
                    rms_scope="All finite residual samples in audited products, retaining failed-optimizer frames and reporting undefined counts separately; fractions multiplied by 100 for RMS percent.",
                    metadata_comparison="Original CSV numeric values independently checked within four float64 eps times max(1, column absolute maximum) to allow parser roundoff; raw flux and wavelengths must match exactly.",
                    pooled_metrics=pooled, array_units=units, input_sha256=input_hashes,
                    script_sha256=digest(__file__))
    manifest_path = out / f"{args.prefix}_ccf_no_injection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    plot_overview(rows, out / f"{args.prefix}_ccf_no_injection_overview.png")
    lines = ["IGRINS observed 79-frame residual series for CCF analysis", "",
             f"Status counts: {json.dumps(manifest['status_counts'])}",
             f"Combined product created: {combined['created']}. {combined['reason']}", "",
             "All 44 stored orders and all 79 original frames are required for the combined artifact.",
             "No artificial planet signal is injected. No time-common mean, template, or low-rank temporal component is subtracted.",
             "The original observations are preserved exactly, with the stored pixel axis reversed.",
             "No absorption or residual masks are added; deep telluric absorption and negative measured flux remain present.", "",
             "To avoid dividing by a small fitted continuum, use residual_fixed_scale_q5_independent: Q5(observed_fixed_scale - full_model_fixed_scale). Its units are fractions of each frame's all-pixel p95 observed flux; multiply by 100 for percent.",
             "This option changes the residual scale and projection. It does not correct the underlying physical model, establish planet-signal throughput, or provide injection-recovery validation.", "",
             manifest["signed_residual"], manifest["transmission"], manifest["conditional_fit"],
             manifest["baseline_fit_scope"], manifest["physical_output_normalization"],
             manifest["q5"], manifest["normalization_validity"], manifest["unchanged_benchmark"], "",
             "The per-order continuum polynomial degree is separate from residual Q5; Q5 always uses degree five. Continuum degree exceptions appear in each order's audit row and the combined continuum_degree array.",
             "residual_continuum_q5 is in fractions of fitted continuum and can become very large where that continuum is small.",
             "residual_fixed_scale_q5 returns that same filtered residual to frame-p95 flux units; rescaling does not undo the continuum division used in its Q5 projection.",
             "Keep fit_success, optimizer_success, valid_math, valid_fixed_scale, both Q5 validity flags and frame_status with any downstream analysis.",
             "The flags describe mathematical/optimization validity, not planet-signal validation. No injection recovery or CCF detection significance is reported.",
             "Very small positive continua can amplify continuum-unit residuals; this does not alter raw/fixed-scale observations.", "",
             "The RMS overview uses a separate logarithmic colorbar for each panel, with explicit continuum or frame-p95 flux units. Every finite value is shown without clipping; colors are not directly comparable across panels.", "",
             manifest["axis_order"], manifest["order_numbering"], "",
             "Model sampling coordinates and IP response:", manifest["model_sampling_coordinates"],
             manifest["model_sampling_target"], manifest["ip_response"], "",
             "NumPy example (the original residual arrays are not interpolated):",
             "  d = np.load('allorders_ccf_no_injection.npz')",
             "  nu_eff = d['target_wavenumber_cm1'][:, None, :] * (",
             "      1 + d['total_calibration_rv_kms'][:, :, None] / d['speed_of_light_kms'])",
             "  wavelength_eff_um = 1e4 / nu_eff",
             "The original RV metadata column remains available separately; its stellar shift is not part of nu_eff.", "",
             "Array units and definitions:"]
    lines.extend(f"  {name}: {description}" for name, description in units.items())
    lines += ["", "Per-order audit status:"]
    for row in rows:
        lines.append(f"  {row['cube_index']:02d}: {row['status']}" +
                     (f"; continuum degree={row['continuum_degree']}; flagged frames={row['flagged_frames']}; undefined normalization samples={row['undefined_normalization_samples']}"
                      if "metrics" in row else f"; {row.get('message', '')}"))
    lines += ["", "RMS scope: " + manifest["rms_scope"], json.dumps(pooled, indent=2), "",
              f"Full hashes, per-order residual metrics and source provenance: {manifest_path.name}"]
    (out / f"{args.prefix}_ccf_no_injection_README.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(dict(status_counts=manifest["status_counts"], combined_product=combined), indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--source-root", type=Path, default=ROOT, help="Root for relative frozen-fit paths in metadata")
    parser.add_argument("--cube", type=Path, default=ROOT / "data/cube14v3.pic")
    parser.add_argument("--info", type=Path, default=ROOT / "data/20201214_info.csv")
    parser.add_argument("--prefix", default="allorders")
    parser.add_argument("--combine", action="store_true", default=True, help="Atomically combine only after all 44 observed series pass independent audits (default)")
    parser.add_argument("--no-combine", action="store_false", dest="combine", help="Write the audit and overview without a combined NPZ")
    run(parser.parse_args())
