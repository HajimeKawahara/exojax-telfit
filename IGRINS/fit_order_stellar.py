"""Apply the fiducial stellar and telluric correction to one released cube order."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import jax
import jax.numpy as jnp
import numpy as np

from fit_blase_stellar import Problem
from fit_line_strength import build_context, make_forward, metrics
from solver_support import install_trf_svd_fallback

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SPECIES = ("H2O", "CH4", "N2O", "CO", "CO2")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def transmission(context, z):
    rows = []
    for block in context["blocks"]:
        unity = dict(block, shifted_star=np.ones_like(block["shifted_star"]))
        _, forward = make_forward(unity)
        rows.append(np.asarray(forward(z[block["parameter_slice"]])))
    return np.asarray(rows)


def selected_metrics(residual, mask):
    values = residual[:, 5:][np.broadcast_to(mask[:, None], residual[:, 5:].shape)]
    return dict(n_wavelengths=int(mask.sum()), n_samples=int(values.size),
                rms_percent=float(100*np.sqrt(np.mean(values**2))) if values.size else None)


def main(args):
    started = time.monotonic()
    svd_fallbacks = install_trf_svd_fallback()
    if not 0 <= args.order < 44:
        raise ValueError("Cube order must be between 0 and 43")
    if args.derivative_chunk_size < 1 or args.clone_retries < 0 or args.null_refits < 0:
        raise ValueError("Chunk size must be positive and retry counts nonnegative")
    tag = args.tag or f"allorders_o{args.order:02d}"
    if Path(tag).name != tag:
        raise ValueError("Tag must be a filename stem, not a path")
    physical_prefix = args.physical_prefix or f"{tag}_physical_e"
    output_stem = f"{tag}_stellar_fit"
    display_order = 7 if args.order == 5 else -1
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_paths = [OUT / name for name in ("adopted_telluric_model.json",
                    "adopted_blase_telluric_fit.npz", "adopted_telluric_fit.npz")]
    frozen_paths = [path for path in frozen_paths if path.exists()]
    frozen_hashes = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    context = build_context(f"{physical_prefix}10", physical_prefix, SPECIES)
    np.testing.assert_array_equal(context["orders"], [args.order])
    np.testing.assert_array_equal(context["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    assert context["allowed"].shape == (1, 1848) and context["allowed"].all()
    assert np.isfinite(context["y_over_c"]).all()
    problem = Problem(context, allow_sparse_calibration=True,
                      derivative_chunk_size=args.derivative_chunk_size,
                      clone_retries=args.clone_retries)
    problem.telluric_checkpoint_path = OUT/f"{tag}_stellar_checkpoint.npz"
    problem.telluric_solver = args.strength_solver
    problem.telluric_solver_options = dict(atol=args.lsmr_tolerance, btol=args.lsmr_tolerance,
                                          maxiter=args.lsmr_maxiter)
    derivative_checks = []
    for block in problem.blocks:
        q0 = jnp.asarray(problem.initial_stellar[block["stellar_slice"]])
        z0 = jnp.asarray(problem.adopted_z[block["strength_slice"]])
        y = jnp.asarray(context["y_over_c"][block["index"], :5])
        residual = block["original_functions"][0]
        rng = np.random.default_rng(20261111)
        for name, vector in [("stellar", q0), ("telluric", z0)]:
            if vector.size == 0:
                derivative_checks.append(dict(parameter=name, skipped=True,
                                              reason="No free model coordinates"))
                continue
            direction = rng.normal(size=vector.size)
            direction /= np.linalg.norm(direction)
            if name == "stellar":
                callback = lambda value: residual(value, z0, y)
            else:
                callback = lambda value: residual(q0, value, y)
            step = 1e-4
            difference = (np.asarray(callback(vector+step*direction))
                          -np.asarray(callback(vector-step*direction)))/(2*step)
            analytic = np.asarray(jax.jvp(callback, (vector,),
                                          (jnp.asarray(direction),))[1])
            norm = float(np.linalg.norm(analytic))
            absolute = float(np.linalg.norm(difference-analytic))
            error = absolute/norm if norm > 1e-12 else None
            assert (error < 1e-5 if error is not None else absolute < 1e-8), (name, error, absolute)
            derivative_checks.append(dict(parameter=name, relative_l2=error,
                                          absolute_l2=absolute, analytic_norm=norm,
                                          method="Directional JVP and central finite difference"))
    initial_q, initial_z = None, None
    if args.resume and problem.telluric_checkpoint_path.exists():
        with np.load(problem.telluric_checkpoint_path) as checkpoint:
            initial_q = checkpoint["q"].copy()
            initial_z = checkpoint["z"].copy() if "z" in checkpoint else None
        assert initial_q.shape == problem.initial_stellar.shape
        assert initial_z is None or initial_z.shape == problem.adopted_z.shape
    if args.polish:
        if initial_q is None or initial_z is None:
            raise ValueError("Polishing requires --resume and a q/z checkpoint")
        q = initial_q
        stellar_fit = [dict(order=int(order), reused=True, success=True, nfev=0)
                       for order in context["orders"]]
    else:
        try:
            q, stellar_fit = problem.fit_stellar(initial=initial_q, max_nfev=args.max_nfev)
        except Exception:
            if hasattr(problem, "last_stellar_state"):
                np.savez_compressed(problem.telluric_checkpoint_path,
                                    q=problem.last_stellar_state,
                                    z=problem.adopted_z if initial_z is None else initial_z)
            raise
    np.savez_compressed(OUT/f"{tag}_stellar_checkpoint.npz", q=q,
                        z=problem.adopted_z if initial_z is None else initial_z)
    z, strength_fit = problem.fit_telluric(q, initial=initial_z, max_nfev=args.max_nfev)
    component = problem.predict(q, z)
    null_checks, stable = [], 0
    for iteration in range(args.null_refits):
        q, stellar_fit = problem.fit_stellar(initial=q, max_nfev=args.max_nfev)
        z, strength_fit = problem.fit_telluric(q, initial=z, max_nfev=args.max_nfev)
        updated = problem.predict(q, z)
        drift = float(np.max(np.abs(updated-component)))
        null_checks.append(dict(iteration=iteration+1, component_max_drift=drift))
        stable = stable+1 if drift < 1e-8 else 0
        component = updated
        np.savez_compressed(OUT/f"{tag}_stellar_checkpoint.npz", q=q, z=z, component=component)
        print("Null refinement", null_checks[-1], flush=True)
        if stable >= 2:
            break
    components = dict(baseline=context["reference_component"],
                      stellar=problem.predict(q, problem.adopted_z), refit=component)
    stars = dict(baseline=problem.stellar(problem.initial_stellar), refit=problem.stellar(q))
    transmissions = dict(baseline=transmission(context, problem.adopted_z), refit=transmission(context, z))
    minimum_t = np.min(transmissions["baseline"][:, :5], axis=1)
    depth = np.max(1-stars["baseline"][:, :5], axis=1)
    masks = dict(strong_stellar=depth > .02, deep_telluric=minimum_t <= .15,
                 strong_telluric=minimum_t < .5, moderate_telluric=(minimum_t >= .5) & (minimum_t < .95),
                 clear_telluric=minimum_t >= .95)
    results, residuals = {}, {}
    arrays = dict(orders=context["orders"], display_order=np.array([display_order]),
                  exposures=context["exposures"], wavelength_um=context["wave"],
                  included_mask=context["allowed"], score_mask=context["score"],
                  y_over_c=context["y_over_c"], continuum=context["continuum"],
                  calibration_mask=problem.calibration_mask,
                  calibration_telluric=problem.calibration_telluric,
                  q=q, z=z, initial_stellar=problem.initial_stellar, initial_z=problem.adopted_z,
                  strength_prior=problem.strength_prior, strength_contrast=problem.strength_contrast,
                  stellar_free_mask=np.concatenate([b["free"] for b in problem.blocks]),
                  stellar_prior=np.concatenate([b["prior"] for b in problem.blocks]),
                  stellar_upper=np.concatenate([b["upper"] for b in problem.blocks]),
                  stellar_rest_centers_angstrom=np.concatenate([b["clone"].centers_angstrom for b in problem.blocks]))
    for name, value in components.items():
        assert np.isfinite(value).all() and value.min() >= -1e-12
        results[name], residuals[name] = metrics(context, value)
        results[name]["depth_subsets"] = {key: selected_metrics(residuals[name], mask) for key, mask in masks.items()}
        arrays[f"component_{name}"] = value
        arrays[f"residual_{name}"] = residuals[name]
    for name in ("baseline", "refit"):
        arrays[f"stellar_only_{name}"] = stars[name]
        arrays[f"telluric_only_{name}"] = transmissions[name]
    arrays.update(component=component, residual_unfiltered=context["y_over_c"]-component,
                  residual_filtered=residuals["refit"], stellar_only=stars["refit"],
                  telluric_only=transmissions["refit"])
    q5_errors = []
    for b, wave in enumerate(context["wave"]):
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        orthogonal, _ = np.linalg.qr(basis)
        raw = arrays["residual_unfiltered"][b]
        projected = raw-(raw@orthogonal)@orthogonal.T
        error = float(np.max(np.abs(projected-arrays["residual_filtered"][b])))
        assert error < 1e-11
        q5_errors.append(error)
    bounds = np.concatenate([block["upper"] for block in context["blocks"]])
    minimum_strength = float(np.min(1+bounds*z)) if len(z) else 1.
    assert minimum_strength >= -1e-10
    assert frozen_hashes == {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    sources = list(dict.fromkeys(context["source_files"] + [Path(__file__), ROOT/"fit_blase_stellar.py",
                ROOT/"fit_line_strength.py", ROOT/"blase_stellar.py", ROOT/"line_strength_support.py",
                ROOT/"solver_support.py"]))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
        display_order=None if display_order < 0 else display_order, cube_index=args.order, wavelength_um=[float(context["wave"].min()), float(context["wave"].max())],
        training_exposures=[0, 19, 39, 59, 78], evaluation_exposures=[10, 30, 50, 70],
        included_wavelengths=int(context["allowed"].sum()), score_wavelengths=int(context["score"].sum()),
        strong_absorption_excluded=False, species=list(SPECIES),
        plan=dict(star="PHOENIX times actual blase amplitude ratio; centers and widths fixed",
                  stellar_calibration="Minimum reference transmission in five training frames > 0.95",
                  strengths="S/HITRAN = 1 + uncertainty_upper*z >= 0; shared across five training frames",
                  weak_line_prior="1+9*d^2/(d^2+0.005^2), model-only original contrast on all included pixels",
                  smooth="Degree-five projection in normalized log wavelength, per frame on all 1848 pixels",
                  fixed="Dated T-P profile, physical columns, wavelength calibration, mixture IP, original continuum",
                  evaluation="Frozen alternating spatial score blocks; all included pixels also reported",
                  temporal_template_subtraction=False),
        metrics=results, fits=dict(stellar=stellar_fit, telluric=strength_fit), null_refinements=null_checks,
        solver_attempts=[], svd_fallbacks=svd_fallbacks,
        solver_options=problem.telluric_solver_options,
        strength_solver=args.strength_solver,
        numerically_stabilized=stable >= 2, stellar_candidate_features=problem.n_stellar,
        stellar_free_features=int(arrays["stellar_free_mask"].sum()),
        stellar_calibration_pixels=int(problem.calibration_mask.sum()),
        free_strength_count=context["n_parameters"], free_strength_species=dict(Counter(r["species"] for r in context["records"])),
        line_records=context["records"], selection_metadata=context["selection_metadata"],
        clone_metadata=[b["clone"].metadata for b in problem.blocks],
        audit=dict(physical_reconstruction_max_error=context["baseline_reconstruction_error"],
                   blase_identity_max_error=problem.reconstruction_error, derivative_checks=derivative_checks,
                   q5_independent_max_error=max(q5_errors), strength_minimum_multiplier=minimum_strength,
                   legacy_adopted_sources_unchanged=True),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        legacy_adopted_sha256=frozen_hashes, seconds=time.monotonic()-started,
        diagnostic_flags=problem.diagnostic_flags,
        physical_prefix=physical_prefix,
        observed_nonpositive_samples=int(np.count_nonzero(context["y_over_c"] <= 0)),
        derivative_chunk_size=args.derivative_chunk_size,
        limitations=["Independent order-specific photon variance is unavailable.",
                     "Only model-supported stellar and reported-error telluric features vary; missing features remain fixed or absent.",
                     "Strong absorption is included, but physical information vanishes where terrestrial transmission approaches zero.",
                     "The physical initialization uses held-out calibration wavelength blocks for nuisance parameters.",
                     "No planet injection test has yet been performed for this new order and mask."])
    product = OUT/f"{output_stem}.npz"
    temporary = OUT/f"{output_stem}.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    temporary.replace(product)
    result["product_sha256"] = digest(product)
    temporary = OUT/f"{output_stem}.tmp.json"
    temporary.write_text(json.dumps(result, indent=2)+"\n")
    temporary.replace(OUT/f"{output_stem}.json")
    print(f"Completed cube order {args.order} stellar/strength fit", result["seconds"], flush=True)
    print(json.dumps({name: value["evaluation_included"] for name, value in results.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True, help="Zero-based released cube row")
    parser.add_argument("--tag", help="Product prefix; default allorders_oNN")
    parser.add_argument("--physical-prefix", help="Physical product prefix, ending in _e")
    parser.add_argument("--derivative-chunk-size", type=int, default=32)
    parser.add_argument("--clone-retries", type=int, default=0)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--null-refits", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--lsmr-maxiter", type=int, default=200)
    parser.add_argument("--lsmr-tolerance", type=float, default=1e-8)
    parser.add_argument("--strength-solver", choices=["trf", "lbfgsb"], default="lbfgsb")
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        tag = args.tag or f"allorders_o{args.order:02d}"
        if Path(tag).name == tag:
            OUT.mkdir(parents=True, exist_ok=True)
            failure = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                           cube_index=args.order, status="failed", adopted=False,
                           exception_type=type(error).__name__, message=str(error),
                           checkpoint=f"{tag}_stellar_checkpoint.npz",
                           checkpoint_exists=(OUT/f"{tag}_stellar_checkpoint.npz").exists())
            (OUT/f"{tag}_stellar_failure.json").write_text(json.dumps(failure, indent=2)+"\n")
        raise
