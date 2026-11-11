"""Fit Order 7 with the existing stellar and weak-line correction scheme."""

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

import jax.numpy as jnp
import numpy as np

from fit_blase_stellar import Problem
from fit_line_strength import build_context, make_forward, metrics

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
    OUT.mkdir(parents=True, exist_ok=True)
    frozen_paths = [OUT / name for name in ("adopted_telluric_model.json",
                    "adopted_blase_telluric_fit.npz", "adopted_telluric_fit.npz")]
    frozen_paths = [path for path in frozen_paths if path.exists()]
    frozen_hashes = {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    context = build_context("order7_physical_e10", "order7_physical_e", SPECIES)
    np.testing.assert_array_equal(context["orders"], [5])
    np.testing.assert_array_equal(context["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    assert context["allowed"].shape == (1, 1848) and context["allowed"].all()
    assert np.isfinite(context["y_over_c"]).all()
    problem = Problem(context)
    problem.telluric_checkpoint_path = OUT/"order7_stellar_checkpoint.npz"
    problem.telluric_solver = args.strength_solver
    problem.telluric_solver_options = dict(atol=args.lsmr_tolerance, btol=args.lsmr_tolerance,
                                          maxiter=args.lsmr_maxiter)
    derivative_checks = []
    for block in problem.blocks:
        q0 = problem.initial_stellar[block["stellar_slice"]]
        z0 = problem.adopted_z[block["strength_slice"]]
        y = jnp.asarray(context["y_over_c"][block["index"], :5])
        residual, q_jacobian, z_jacobian = block["original_functions"]
        rng = np.random.default_rng(20261111)
        for name, vector, jacobian in [("stellar", q0, q_jacobian), ("telluric", z0, z_jacobian)]:
            direction = rng.normal(size=vector.size)
            direction /= np.linalg.norm(direction)
            step = 1e-4
            if name == "stellar":
                difference = (np.asarray(residual(q0+step*direction, z0, y))
                              -np.asarray(residual(q0-step*direction, z0, y)))/(2*step)
            else:
                difference = (np.asarray(residual(q0, z0+step*direction, y))
                              -np.asarray(residual(q0, z0-step*direction, y)))/(2*step)
            analytic = np.asarray(jacobian(q0, z0, y))@direction
            error = float(np.linalg.norm(difference-analytic)/np.linalg.norm(analytic))
            assert error < 1e-5, (name, error)
            derivative_checks.append(dict(parameter=name, relative_l2=error))
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
        q, stellar_fit = problem.fit_stellar(initial=initial_q, max_nfev=args.max_nfev)
    np.savez_compressed(OUT/"order7_stellar_checkpoint.npz", q=q,
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
        np.savez_compressed(OUT/"order7_stellar_checkpoint.npz", q=q, z=z, component=component)
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
    arrays = dict(orders=context["orders"], display_order=np.array([7]),
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
        assert np.isfinite(value).all() and value.min() >= 0
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
    assert np.min(1+bounds*z) >= -1e-10
    assert frozen_hashes == {str(p.relative_to(ROOT)): digest(p) for p in frozen_paths}
    sources = list(dict.fromkeys(context["source_files"] + [Path(__file__), ROOT/"fit_blase_stellar.py",
                ROOT/"fit_line_strength.py", ROOT/"blase_stellar.py", ROOT/"line_strength_support.py"]))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
        display_order=7, cube_index=5, wavelength_um=[float(context["wave"].min()), float(context["wave"].max())],
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
        solver_attempts=json.loads((OUT/"order7_solver_attempts.json").read_text())
                        if (OUT/"order7_solver_attempts.json").exists() else [],
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
                   q5_independent_max_error=max(q5_errors), strength_minimum_multiplier=float(np.min(1+bounds*z)),
                   legacy_adopted_sources_unchanged=True),
        source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources},
        legacy_adopted_sha256=frozen_hashes, seconds=time.monotonic()-started,
        limitations=["Independent order-specific photon variance is unavailable.",
                     "Only model-supported stellar and reported-error telluric features vary; missing features remain fixed or absent.",
                     "Strong absorption is included, but physical information vanishes where terrestrial transmission approaches zero.",
                     "The physical initialization uses held-out calibration wavelength blocks for nuisance parameters.",
                     "No planet injection test has yet been performed for this new order and mask."])
    np.savez_compressed(OUT/"order7_stellar_fit.npz", **arrays)
    result["product_sha256"] = digest(OUT/"order7_stellar_fit.npz")
    (OUT/"order7_stellar_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Completed Order 7 stellar/strength fit", result["seconds"], flush=True)
    print(json.dumps({name: value["evaluation_included"] for name, value in results.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=200)
    parser.add_argument("--null-refits", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--polish", action="store_true")
    parser.add_argument("--lsmr-maxiter", type=int, default=200)
    parser.add_argument("--lsmr-tolerance", type=float, default=1e-8)
    parser.add_argument("--strength-solver", choices=["trf", "lbfgsb"], default="trf")
    main(parser.parse_args())
