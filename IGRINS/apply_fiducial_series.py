"""Calibrate the physical baseline, then apply frozen corrections without injection."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from atmosphere_profile import load_profile
from blase_stellar import build_clone
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from fit_blase_stellar import FixedReferenceStar
from fit_joint import profile_tau
from fit_line_strength import build_context
from hitran_cache import load_mdb, make_opacity

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SPECIES = ("H2O", "CH4", "N2O", "CO", "CO2")
PARAMETER_NAMES = ["log10_water", "calibration_rv_epoch_kms", "log_ip_width_epoch"]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_immutable_dependencies(metadata):
    """Reject a frozen shared fit whose recorded physical inputs changed."""
    dependencies = {name: expected for name, expected in metadata.get("source_sha256", {}).items()
                    if name.startswith("results/")}
    if not dependencies:
        raise ValueError("Shared fit metadata has no immutable results dependencies")
    for name, expected in dependencies.items():
        try:
            actual = digest(ROOT/name)
        except OSError as error:
            raise ValueError(f"Missing shared fit dependency: {name}") from error
        if actual != expected:
            raise ValueError(f"Shared fit dependency changed: {name}; rebuild the shared fit first")


def cached_inputs_match(previous, source_metadata_hash):
    """Require the same metadata and original observations before reusing a series."""
    if previous.get("source_fit_metadata_sha256") != source_metadata_hash:
        return False
    expected = previous.get("input_sha256", {})
    try:
        return all(expected.get(name) == digest(ROOT/name) for name in
                   ("data/cube14v3.pic", "data/20201214_info.csv"))
    except OSError:
        return False


def rms(values):
    values = np.asarray(values)
    return float(100*np.sqrt(np.mean(values**2))) if values.size else None


def project_q5(values, basis, valid):
    """Remove only defined smooth modes, recording the mathematical domain."""
    residual = np.full(values.shape, np.nan)
    smooth = np.full(values.shape, np.nan)
    coefficients = np.full(6, np.nan)
    if np.count_nonzero(valid) < 6:
        return residual, smooth, coefficients, False
    coefficients, _, rank, _ = np.linalg.lstsq(basis[valid], values[valid], rcond=None)
    if rank != 6 or not np.isfinite(coefficients).all():
        return residual, smooth, np.full(6, np.nan), False
    smooth = basis @ coefficients
    residual[valid] = values[valid]-smooth[valid]
    return residual, smooth, coefficients, True


def main(args):
    started = time.monotonic()
    application_code_hash = digest(__file__)
    if not 0 <= args.order < 44:
        raise ValueError("Use a released cube index in 0..43")
    tag = args.tag or f"allorders_o{args.order:02d}"
    if Path(tag).name != tag:
        raise ValueError("Tag must be a filename stem")
    OUT.mkdir(parents=True, exist_ok=True)
    source = OUT/f"{tag}_stellar_fit.npz"
    source_metadata = source.with_suffix(".json")
    source_metadata_bytes = source_metadata.read_bytes()
    metadata = json.loads(source_metadata_bytes)
    source_metadata_hash = hashlib.sha256(source_metadata_bytes).hexdigest()
    source_hash = digest(source)
    if source_hash != metadata["product_sha256"]:
        raise ValueError("Shared fit product does not match its recorded hash")
    validate_immutable_dependencies(metadata)
    physical_prefix = (args.physical_prefix or metadata.get("physical_prefix") or
                       ("order7_physical_e" if metadata.get("reference_reused")
                        else f"{tag}_physical_e"))
    stem = f"{tag}_ccf_no_injection"
    output = OUT/f"{stem}.npz"
    result_path = OUT/f"{stem}.json"
    if output.exists() and result_path.exists() and not args.overwrite:
        previous = json.loads(result_path.read_text())
        if (previous.get("source_fit_product_sha256") == source_hash
                and previous.get("application_code_sha256") == application_code_hash
                and previous.get("product_sha256") == digest(output)
                and cached_inputs_match(previous, source_metadata_hash)):
            print(f"Reusing complete 79-frame product: {output}", flush=True)
            return previous
        raise ValueError("Existing series has different provenance; use --overwrite explicitly")
    with np.load(source) as saved:
        shared = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(shared["orders"], [args.order])
    assert shared["included_mask"].all()
    context = build_context(f"{physical_prefix}10", physical_prefix, SPECIES)
    assert context["records"] == metadata["line_records"]
    block = context["blocks"][0]
    physical = context["physical_parameters"]
    physical_metadata = context["physical_metadata"]
    grid, velocity, target = block["grid"], block["velocity"], block["target"]
    wave = context["wave"][0]
    np.testing.assert_array_equal(wave, shared["wavelength_um"][0])
    q, z = shared["q"], shared["z"]
    if q.size:
        clone = build_clone(np.asarray(grid), args.order)
        if clone.n_lines != len(q):
            raise ValueError("Frozen stellar amplitudes do not match the cached clone")
    else:
        clone = FixedReferenceStar(grid, "Frozen fiducial fit has no stellar amplitude corrections")
    raw_star = clone.corrected_raw(q)
    rotated = convolve_rigid_rotation(raw_star, velocity, 4., .6, 0.)
    broadened = ipgauss(rotated, velocity, physical["stellar_extra_sigma_kms"])
    baseline_star = clone.corrected_raw(np.ones_like(q))
    baseline_star = convolve_rigid_rotation(baseline_star, velocity, 4., .6, 0.)
    baseline_star = ipgauss(baseline_star, velocity, physical["stellar_extra_sigma_kms"])
    profile = load_profile(physical_metadata["layers"])
    species_tau, baseline_species_tau, rounding = {}, {}, {}
    for molecule in SPECIES:
        database = load_mdb(molecule, float(grid[0])-physical_metadata["line_margin_cm1"],
                            float(grid[-1])+physical_metadata["line_margin_cm1"],
                            physical_metadata["line_cutoff"])
        baseline = profile_tau(make_opacity(database, grid), grid, profile, molecule,
                               physical_metadata["pressure_shift"],
                               physical_metadata["reference_self_broadening"])[0]
        baseline_species_tau[molecule] = baseline
        selected = np.array([r["species"] == molecule for r in context["records"]])
        correction = (block["upper"][selected]*z[selected]) @ block["line_tau"][selected]
        corrected_tau = np.asarray(baseline)+correction
        if not np.isfinite(corrected_tau).all() or corrected_tau.min() < -1e-10:
            raise ValueError(f"Invalid corrected optical depth for {molecule}")
        rounding[molecule] = dict(negative_roundoff_samples=int(np.count_nonzero(corrected_tau < 0)),
                                  minimum_before_roundoff_guard=float(corrected_tau.min()))
        species_tau[molecule] = jnp.asarray(np.maximum(corrected_tau, 0.))
    water_tau = species_tau["H2O"]
    dry_tau = sum(10**physical[f"log10_scale_{s}"]*species_tau[s]
                  for s in SPECIES if s != "H2O")
    baseline_water_tau = baseline_species_tau["H2O"]
    baseline_dry_tau = sum(10**physical[f"log10_scale_{s}"]*baseline_species_tau[s]
                           for s in SPECIES if s != "H2O")
    width0 = physical[f"ip_sigma_o{args.order}_kms"]
    offset0 = physical[f"calibration_rv_o{args.order}_kms"]
    fraction = physical["ip_wing_fraction"]
    # All shared line and atmospheric corrections are collapsed before the
    # frame loop. Only three frame-specific nuisance coordinates remain free.
    del block["line_tau"]

    def response(row, width, offset):
        core = sampling(target, grid, ipgauss(row, velocity, width), offset)
        wing = sampling(target, grid, ipgauss(row, velocity, 2*width), offset)
        return (1-fraction)*core+fraction*wing

    def components(parameters, stellar, air):
        width = width0*jnp.exp(parameters[2])
        offset = offset0+parameters[1]
        transmission = jnp.exp(-air*(dry_tau+10**parameters[0]*water_tau))
        return (response(stellar*transmission, width, offset),
                response(stellar, width, offset), response(transmission, width, offset))

    forward = jax.jit(lambda parameters, stellar, air: components(parameters, stellar, air)[0])
    components = jax.jit(components)

    @jax.jit
    def baseline_forward(parameters, stellar, air):
        transmission = jnp.exp(-air*(baseline_dry_tau+10**parameters[0]*baseline_water_tau))
        return response(stellar*transmission, width0*jnp.exp(parameters[2]), offset0+parameters[1])

    info = pd.read_csv(ROOT/"data/20201214_info.csv")
    exposures = np.arange(len(info))
    assert len(exposures) == 79
    relative_rv = info["RV [km/s]"].to_numpy()-info.iloc[0]["RV [km/s]"]
    stars = jax.jit(jax.vmap(lambda rv: sampling(grid, grid, broadened, rv)))(
        jnp.asarray(physical["stellar_rv_reference_kms"]+relative_rv))
    baseline_stars = jax.jit(jax.vmap(lambda rv: sampling(grid, grid, baseline_star, rv)))(
        jnp.asarray(physical["stellar_rv_reference_kms"]+relative_rv))
    air = info["Air Mass"].to_numpy()
    known_frames = context["exposures"]
    known_parameters = []
    for frame in known_frames:
        record = (physical if frame in known_frames[:5] else
                  json.loads((OUT/f"{physical_prefix}{frame}.json").read_text())["parameters"])
        known_parameters.append([record[f"log10_water_e{frame}"],
                                 record.get(f"calibration_rv_e{frame}_kms", 0.),
                                 record.get(f"log_ip_width_e{frame}", 0.)])
    known_parameters = np.asarray(known_parameters)
    reconstructed = np.asarray([forward(parameters, stars[frame], air[frame])
                                for frame, parameters in zip(known_frames, known_parameters)])
    reconstruction_error = float(np.max(abs(reconstructed-shared["component"][0])))
    np.testing.assert_allclose(reconstructed, shared["component"][0], rtol=0, atol=1e-10)
    baseline_reconstructed = np.asarray([baseline_forward(parameters, baseline_stars[frame], air[frame])
                                         for frame, parameters in zip(known_frames, known_parameters)])
    baseline_reconstruction_error = float(np.max(abs(baseline_reconstructed-context["reference_component"][0])))
    np.testing.assert_allclose(baseline_reconstructed, context["reference_component"][0], rtol=0, atol=1e-10)
    ordering = np.argsort(known_frames)
    initial = np.array([np.interp(exposures, known_frames[ordering], known_parameters[ordering, i])
                        for i in range(3)]).T
    with (ROOT/"data/cube14v3.pic").open("rb") as stream:
        original_wave, cube = pickle.load(stream)
    np.testing.assert_array_equal(wave, original_wave[args.order, ::-1])
    observed_raw = cube[args.order, :, ::-1].copy()
    assert np.isfinite(observed_raw).all()
    scalar = np.percentile(observed_raw, 95, axis=1)
    if np.any(scalar <= 0) or not np.isfinite(scalar).all():
        raise ValueError("The fixed observational normalization is undefined")
    observed = observed_raw/scalar[:, None]
    continuum_degree = int(physical_metadata["continuum_degree"])
    continuum_basis = jnp.asarray(np.polynomial.chebyshev.chebvander(
        np.linspace(-1, 1, len(wave)), continuum_degree))
    pixel = np.arange(len(wave))
    calibration = ((pixel//80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    continuum_scope = physical_metadata.get("continuum_scope", "calibration")
    if continuum_scope not in {"all", "calibration"}:
        raise ValueError("Unknown adopted physical continuum scope")
    continuum_weight = jnp.asarray(np.ones(len(wave)) if continuum_scope == "all" else calibration.astype(float))
    physical_scalar = np.percentile(observed_raw[:, calibration], 95, axis=1)
    physical_scalar[known_frames[:5]] = scalar[known_frames[:5]]
    if np.any(physical_scalar <= 0) or not np.isfinite(physical_scalar).all():
        raise ValueError("The physical-stage observational normalization is undefined")
    physical_observed = observed_raw/physical_scalar[:, None]
    scale_conversion = physical_scalar/scalar
    known_index = {int(frame): i for i, frame in enumerate(known_frames)}
    known_continuum = context["continuum"][0]*scale_conversion[known_frames, None]
    np.testing.assert_allclose(physical_observed[known_frames],
                               context["y_over_c"][0]*context["continuum"][0], rtol=0, atol=1e-12)

    def evaluate(parameters, y, stellar, air):
        model = baseline_forward(parameters, stellar, air)
        design = model[:, None]*continuum_basis
        coefficients = jnp.linalg.solve(
            design.T@(continuum_weight[:, None]*design)+1e-12*jnp.eye(continuum_degree+1),
            design.T@(continuum_weight*y))
        continuum = continuum_basis@coefficients
        return model, continuum, coefficients

    def residual(parameters, y, stellar, air):
        model, continuum, _ = evaluate(parameters, y, stellar, air)
        return (y-model*continuum)[calibration]

    evaluate = jax.jit(evaluate)
    residual = jax.jit(residual)
    jacobian = jax.jit(jax.jacfwd(residual, 0))
    lower, upper = np.array([-1.5, -2., -.15]), np.array([1.5, 2., .15])
    shape = observed.shape
    arrays = dict(cube_index=np.array(args.order), wavelength_um=wave, wavenumber_cm1=1e4/wave,
                  exposures=exposures, time_bjd=info["Time (BJD)"].to_numpy(),
                  phase=info["Phase"].to_numpy(), median_snr=info["Med. SNR"].to_numpy(),
                  airmass=air, rv_metadata_kms=info["RV [km/s]"].to_numpy(),
                  included_mask=np.ones(len(wave), bool), nuisance_calibration_mask=calibration,
                  observed_raw=observed_raw, normalization_scalar=scalar,
                  physical_normalization_scalar=physical_scalar, continuum_scale_conversion=scale_conversion,
                  observed_fixed_scale=observed, nuisance_initial_parameters=initial,
                  nuisance_parameters=np.full((len(exposures), 3), np.nan),
                  continuum_coefficients=np.full((len(exposures), continuum_degree+1), np.nan),
                  valid_math=np.zeros(shape, bool), valid_fixed_scale=np.zeros(shape, bool),
                  optimizer_success=np.zeros(len(exposures), bool),
                  reused_physical_frame=np.isin(exposures, known_frames),
                  nuisance_optimization_executed=~np.isin(exposures, known_frames),
                  nuisance_source=np.full(len(exposures), "new_heldout_fit", dtype="U32"),
                  continuum_scope_per_frame=np.full(len(exposures), continuum_scope, dtype="U11"),
                  nuisance_scope_per_frame=np.full(len(exposures), "calibration", dtype="U11"),
                  fit_success=np.zeros(len(exposures), bool),
                  q5_valid_frame=np.zeros(len(exposures), bool),
                  q5_valid_pixel_count=np.zeros(len(exposures), int),
                  fixed_scale_q5_valid_frame=np.zeros(len(exposures), bool),
                  fixed_scale_q5_valid_pixel_count=np.zeros(len(exposures), int),
                  frame_status=np.full(len(exposures), "not_processed", dtype="U40"),
                  q5_continuum_coefficients=np.full((len(exposures), 6), np.nan),
                  q5_fixed_scale_coefficients=np.full((len(exposures), 6), np.nan))
    for key in ["continuum_fixed_scale", "observed_continuum", "full_model", "full_model_fixed_scale",
                "baseline_model", "baseline_model_fixed_scale",
                "stellar_only", "telluric_only", "residual_fixed_scale_unfiltered",
                "residual_continuum_unfiltered", "residual_continuum_q5", "residual_fixed_scale_q5",
                "residual_fixed_scale_q5_independent", "smooth_continuum_q5"]:
        arrays[key] = np.full(shape, np.nan)
    x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
    q5_basis = np.polynomial.chebyshev.chebvander(x, 5)
    records = []
    for frame in exposures:
        y, star, mass = jnp.asarray(physical_observed[frame]), baseline_stars[frame], float(air[frame])
        fixed = np.clip(initial[frame], lower+1e-10, upper-1e-10)
        free = np.array([0]) if frame == 0 else np.arange(3)
        if frame == 0:
            fixed[1:] = 0.
        if frame not in known_index and frame > 0 and arrays["optimizer_success"][frame-1]:
            previous = arrays["nuisance_parameters"][frame-1].copy()
            if np.sum(np.asarray(residual(previous, y, star, mass))**2) < np.sum(
                    np.asarray(residual(fixed, y, star, mass))**2):
                fixed = np.clip(previous, lower+1e-10, upper-1e-10)

        def expand(values):
            full = fixed.copy()
            full[free] = values
            return full

        if frame in known_index:
            index = known_index[frame]
            training = index < 5
            source_meta = (physical_metadata if training else
                           json.loads((OUT/f"{physical_prefix}{frame}.json").read_text()))
            fit_meta = source_meta["training_fit" if training else "heldout_calibration_fit"]
            parameters = known_parameters[index].copy()
            continuum = known_continuum[index].copy()
            coefficients = np.linalg.lstsq(np.asarray(continuum_basis), continuum, rcond=None)[0]
            baseline_model = np.asarray(baseline_forward(parameters, star, mass))
            success = bool(fit_meta["success"])
            status = "reused_physical_fit" if success else "source_optimizer_not_converged"
            arrays["nuisance_source"][frame] = "saved_training" if training else "saved_heldout"
            arrays["continuum_scope_per_frame"][frame] = "all" if training else source_meta.get("continuum_scope", "calibration")
            arrays["nuisance_scope_per_frame"][frame] = "all" if training else "calibration"
            record = dict(frame=int(frame), optimizer_success=success,
                          message="Reused saved physical-stage nuisance parameters and continuum", nfev=0,
                          physical_source_tag=f"{physical_prefix}{10 if training else frame}")
        else:
            try:
                fit = least_squares(lambda values: np.asarray(residual(expand(values), y, star, mass)),
                                    fixed[free], jac=lambda values: np.asarray(jacobian(expand(values), y, star, mass))[:, free],
                                    bounds=(lower[free], upper[free]), method="trf", x_scale="jac",
                                    ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=args.max_nfev)
                parameters = expand(fit.x)
                success = bool(fit.success)
                status = "success" if success else "optimizer_not_converged"
                record = dict(frame=int(frame), optimizer_success=success, message=str(fit.message),
                              nfev=int(fit.nfev), optimality=float(fit.optimality))
            except Exception as error:
                parameters, success, status = fixed, False, "optimizer_exception"
                record = dict(frame=int(frame), optimizer_success=False,
                              exception_type=type(error).__name__, message=str(error))
            baseline_model, continuum, coefficients = map(np.asarray, evaluate(parameters, y, star, mass))
            continuum = continuum*scale_conversion[frame]
            coefficients = coefficients*scale_conversion[frame]
        model, stellar_only, telluric_only = map(np.asarray, components(parameters, stars[frame], mass))
        # Preserve unavailable mathematical samples as NaN with explicit flags.
        # The original observed flux remains intact, including negative values.
        model, continuum, coefficients, stellar_only, telluric_only = (
            np.where(np.isfinite(value), value, np.nan) for value in
            (model, continuum, coefficients, stellar_only, telluric_only))
        fixed_model = model*continuum
        fixed_model[~np.isfinite(fixed_model)] = np.nan
        fixed_residual = observed[frame]-fixed_model
        fixed_valid = np.isfinite(fixed_residual)
        valid = np.isfinite(continuum) & (continuum > 0) & fixed_valid
        normalized = np.full(len(wave), np.nan)
        normalized_residual = np.full(len(wave), np.nan)
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            normalized[valid] = observed[frame, valid]/continuum[valid]
            normalized_residual[valid] = fixed_residual[valid]/continuum[valid]
        valid &= np.isfinite(normalized) & np.isfinite(normalized_residual)
        normalized[~valid] = np.nan
        normalized_residual[~valid] = np.nan
        filtered, smooth, coefficients_q5, q5_valid = project_q5(normalized_residual, q5_basis, valid)
        independent, _, fixed_coefficients, fixed_q5_valid = project_q5(fixed_residual, q5_basis, fixed_valid)
        if not valid.all():
            status = "undefined_normalization"
        if not q5_valid:
            status = "undefined_q5_projection"
        if not fixed_q5_valid:
            status = "undefined_fixed_scale_q5"
        for key, value in dict(nuisance_parameters=parameters, continuum_coefficients=coefficients,
                               continuum_fixed_scale=continuum, observed_continuum=normalized,
                               full_model=model, full_model_fixed_scale=fixed_model,
                               baseline_model=baseline_model,
                               baseline_model_fixed_scale=baseline_model*continuum,
                               stellar_only=stellar_only, telluric_only=telluric_only,
                               residual_fixed_scale_unfiltered=fixed_residual,
                               residual_continuum_unfiltered=normalized_residual,
                               residual_continuum_q5=filtered, residual_fixed_scale_q5=filtered*continuum,
                               residual_fixed_scale_q5_independent=independent,
                               smooth_continuum_q5=smooth, valid_math=valid, valid_fixed_scale=fixed_valid,
                               optimizer_success=success, fit_success=success and valid.all() and q5_valid,
                               q5_valid_frame=q5_valid, q5_valid_pixel_count=int(valid.sum()),
                               fixed_scale_q5_valid_frame=fixed_q5_valid,
                               fixed_scale_q5_valid_pixel_count=int(fixed_valid.sum()),
                               frame_status=status, q5_continuum_coefficients=coefficients_q5,
                               q5_fixed_scale_coefficients=fixed_coefficients).items():
            arrays[key][frame] = value
        record.update(status=status, valid_math_pixels=int(valid.sum()),
                      optimizer_executed=bool(arrays["nuisance_optimization_executed"][frame]),
                      nuisance_source=str(arrays["nuisance_source"][frame]),
                      continuum_scope=str(arrays["continuum_scope_per_frame"][frame]),
                      nuisance_scope=str(arrays["nuisance_scope_per_frame"][frame]),
                      physical_normalization_scalar=float(physical_scalar[frame]),
                      continuum_scale_conversion=float(scale_conversion[frame]),
                      valid_fixed_scale_pixels=int(fixed_valid.sum()),
                      q5_valid_frame=bool(q5_valid), fixed_scale_q5_valid_frame=bool(fixed_q5_valid),
                      continuum_minimum=float(np.min(continuum)) if np.isfinite(continuum).all() else None,
                      continuum_q5_rms_percent=rms(filtered[valid]) if q5_valid else None,
                      fixed_scale_q5_rms_percent=rms(independent[np.isfinite(independent)]),
                      nuisance_parameters=parameters.tolist(),
                      parameters_near_bounds=[PARAMETER_NAMES[i] for i in free
                          if min(parameters[i]-lower[i], upper[i]-parameters[i]) < 1e-5])
        records.append(record)
        progress = dict(cube_index=args.order, completed_frames=len(records), total_frames=len(exposures),
                        source_fit_product_sha256=source_hash, frames=records, seconds=time.monotonic()-started)
        temporary = OUT/f"{stem}_progress.tmp.json"
        temporary.write_text(json.dumps(progress, indent=2)+"\n")
        temporary.replace(OUT/f"{stem}_progress.json")
        print(json.dumps(record), flush=True)
    known_checks = {}
    for key, expected in [("observed_continuum", shared["y_over_c"][0]),
                          ("residual_continuum_unfiltered", shared["residual_unfiltered"][0]),
                          ("residual_continuum_q5", shared["residual_filtered"][0])]:
        actual = arrays[key][known_frames]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
        known_checks[key] = float(np.max(abs(actual-expected)))
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), cube_index=args.order,
                  status="complete", total_frames=len(exposures), retained_wavelength_pixels=len(wave),
                  frames_with_successful_fit=int(arrays["fit_success"].sum()),
                  frames_with_optimizer_success=int(arrays["optimizer_success"].sum()),
                  reused_physical_frames=known_frames.tolist(), newly_calibrated_frames=70,
                  application_method="baseline_first_frozen_corrections",
                  corrections_applied_after_calibration=True,
                  undefined_normalization_samples=int(np.count_nonzero(~arrays["valid_math"])),
                  undefined_fixed_scale_samples=int(np.count_nonzero(~arrays["valid_fixed_scale"])),
                  artificial_injection=False, time_common_subtraction=False, shared_parameters_refitted=False,
                  telluric_division=False, enuma_used=False, new_absorption_or_residual_masks=False,
                  nuisance_parameter_names=PARAMETER_NAMES, nuisance_bounds=[lower.tolist(), upper.tolist()],
                  frame_zero_anchor="Saved training frame zero is reused, including its zero epoch wavelength/IP corrections",
                  continuum_scope=continuum_scope, continuum_degree=continuum_degree,
                  nuisance_scope="Original physical stages: saved training/all pixels; saved or new heldout/calibration blocks",
                  calibration_model="Uncorrected PHOENIX and uncorrected HITRAN optical depths with frozen shared physical parameters; q/z applied only after nuisance/continuum calibration",
                  physical_normalization="Training: full-pixel 95th percentile. Heldout and new frames: calibration-mask 95th percentile. Saved/profiled continua converted to common full-pixel output units.",
                  q5_continuum_scope="All finite positive-continuum pixels; all 1848 normally. Undefined mathematical samples are flagged, not hidden.",
                  q5_fixed_scale_scope="All finite fixed-scale residual pixels, independently of continuum positivity. Fewer than six defined samples or deficient rank makes the frame projection undefined.",
                  q5_degree=5, q5_coordinates="Normalized log wavelength on the original stored grid",
                  normalization="Each frame's 95th percentile over all 1848 original pixels; retained observed flux can be negative",
                  orientation="Descending wavelength, ascending wavenumber; observed_raw is the original cube row with pixels reversed",
                  time_bjd="Original CSV Time (BJD), unchanged; no timescale reinterpretation",
                  source_fit_product=str(source.relative_to(ROOT)), source_fit_product_sha256=source_hash,
                  source_fit_metadata=str(source_metadata.relative_to(ROOT)), source_fit_metadata_sha256=source_metadata_hash,
                  physical_prefix=physical_prefix, frozen_nine_frame_reconstruction_max_error=reconstruction_error,
                  frozen_nine_frame_baseline_reconstruction_max_error=baseline_reconstruction_error,
                  frozen_nine_frame_normalized_reconstruction_max_errors=known_checks,
                  shared_coordinate_count=dict(stellar_amplitudes=len(q), telluric_strengths=len(z)),
                  reference_optical_depth_roundoff=rounding, frames=records,
                  residual_definitions={
                      "residual_continuum_unfiltered": "(observed_fixed_scale - full_model_fixed_scale) / continuum_fixed_scale",
                      "residual_continuum_q5": "Continuum-unit residual with six smooth polynomial modes removed; no telluric division",
                      "residual_fixed_scale_q5": "continuum_fixed_scale * residual_continuum_q5; same projection as the normalized residual",
                      "residual_fixed_scale_q5_independent": "Degree-five projection directly on observed_fixed_scale - full_model_fixed_scale; different filter, no continuum or telluric division"},
                  flux_units="Residual arrays are fractions, not percentages. Raw flux retains the source cube units.",
                  full_model_definition="Frozen corrected star times corrected terrestrial transmission, then IP convolution and sampling; excludes continuum and Q5",
                  limitations=["The nine saved physical fits are reused; only the other seventy frames receive new heldout-style physical nuisance/continuum fits. Shared q/z remain frozen.",
                               "Training frames retain their original all-pixel physical fits; heldout/new frames use calibration blocks and the adopted physical continuum scope, recorded per frame.",
                               "Planet light is neither injected nor modeled away explicitly; any remaining planet signal is still modulated by telluric transmission and the IP.",
                               "No time-common mean, template or low-rank temporal component is removed.",
                               "Normalization and fitting validity flags must accompany CCF use; positive but very small continua amplify continuum-unit residuals.",
                               "Separate convolved stellar and telluric components do not multiply to the convolved full model."],
                  application_code_sha256=application_code_hash,
                  input_sha256={str(p.relative_to(ROOT)): digest(p) for p in
                      [ROOT/"data/cube14v3.pic", ROOT/"data/20201214_info.csv"]},
                  product_sha256=digest(output), seconds=time.monotonic()-started)
    temporary = result_path.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(result_path)
    assert digest(source) == source_hash
    print(json.dumps({key: result[key] for key in ["cube_index", "total_frames", "frames_with_successful_fit", "undefined_normalization_samples", "seconds"]}), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True)
    parser.add_argument("--tag")
    parser.add_argument("--physical-prefix")
    parser.add_argument("--max-nfev", type=int, default=150)
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
