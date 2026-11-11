"""Compare regularized line widths and centers with the adopted fixed IP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from atmosphere_profile import load_profile
from fit_line_strength import build_context as original_context, metrics, WORKING_SIGMA
from line_strength_support import FINITE_UPPER
from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity.lpf.lpf import voigt
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import Patm, c

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CENTER_UPPER = {3: .01, 4: .001, 5: .0001, 6: .00001, 7: .000001,
                8: .0000001, 9: .00000001}
CASES = [dict(name="strength_control", mode="strength", prior_multiplier=1.)]
CASES += [dict(name=f"{mode}_prior{k}", mode=mode, prior_multiplier=float(k))
          for mode in ["width", "center", "joint"] for k in [1, 3]]


def line_functions(block, selected, profile):
    """Evaluate independent layered profiles and their two local derivatives."""
    records = [block["records"][i] for i in selected]
    raw_indices = np.array([r["raw_index"] for r in records])
    with np.load(ROOT / "cache/H2O.npz") as raw:
        a = {key: np.asarray(raw[key][raw_indices]) for key in
             ["nu", "sw", "a", "gamma_air", "gamma_self", "n_air", "elower",
              "delta_air", "line_mass", "iso"]}
        ts = raw["temperatures"]
        isotope = np.searchsorted(raw["partition_isotopes"], a["iso"])
        partition = raw["partition_sums"][isotope]
    gamma, sigma, shift, weights = [], [], [], []
    for t, p, column in zip(profile["temperature_K"], profile["pressure_bar"],
                            profile["number_column_cm2"]*profile["vmr"]["H2O"]):
        qr = np.array([np.interp(t, ts, q)/np.interp(296., ts, q) for q in partition])
        weights.append(column*np.asarray(line_strength(t, jnp.log(a["sw"]),
                            a["nu"], a["elower"], qr, 296.)))
        gamma.append(np.asarray(gamma_hitran(p, t, 0., a["n_air"], a["gamma_air"], a["gamma_self"])))
        sigma.append(np.asarray(doppler_sigma(a["nu"], t, a["line_mass"])))
        shift.append(a["delta_air"]*p/Patm)
    constants = [jnp.asarray(a["nu"]), jnp.asarray(np.array(shift).T),
                 jnp.asarray(np.array(sigma).T), jnp.asarray(np.array(gamma).T),
                 gamma_natural(jnp.asarray(a["a"])), jnp.asarray(np.array(weights).T)]
    grid = block["grid"]

    def one(q, nu, shifts, sigmas, collisional, natural, weighted_strength):
        offset = grid[None, :]-nu-shifts[:, None]-nu*q[1]/c
        widths = collisional*jnp.exp(q[0])+natural
        values = jax.vmap(voigt)(offset, sigmas, widths)
        return jnp.sum(weighted_strength[:, None]*values, axis=0)

    values = jax.jit(lambda q: jax.vmap(one)(q, *constants))
    derivative = jax.jit(lambda q: jax.vmap(jax.jacfwd(one, argnums=0))(q, *constants))
    return values, derivative


def prepare_block(block, selected, profile):
    """Keep opacity changes before exponentiation; use the fixed linear IP."""
    n, m = len(block["records"]), len(selected)
    original_tau = jnp.asarray(block["line_tau"])
    upper, scales = jnp.asarray(block["upper"]), jnp.asarray(block["line_factors"])
    airmass = jnp.asarray(block["airmass"])
    grid, velocity, target = block["grid"], block["velocity"], block["target"]
    fraction = block["wing_fraction"]
    if m:
        values, derivative = line_functions(block, selected, profile)
        check = np.asarray(values(jnp.zeros((m, 2))))
        np.testing.assert_allclose(check, original_tau[selected], atol=2e-12, rtol=1e-10)
        block["line_tau_reconstruction_max_error"] = float(np.max(abs(check-original_tau[selected])))
    else:
        values = lambda q: jnp.zeros((0, original_tau.shape[1]))
        derivative = lambda q: jnp.zeros((0, original_tau.shape[1], 2))
        block["line_tau_reconstruction_max_error"] = 0.

    def response(row, width, offset):
        core = sampling(target, grid, ipgauss(row, velocity, width), offset)
        wing = sampling(target, grid, ipgauss(row, velocity, 2*width), offset)
        return (1-fraction)*core+fraction*wing

    response_rows = jax.vmap(response)

    @jax.jit
    def forward_parts(theta, changed_tau):
        z = theta[:n]
        tau_lines = original_tau.at[selected].set(changed_tau)
        tau = jnp.asarray(block["baseline_tau"])+(scales*(upper*z)) @ tau_lines
        tau += scales[:, selected] @ (changed_tau-original_tau[selected])
        intrinsic = jnp.asarray(block["shifted_star"])*jnp.exp(-airmass[:, None]*tau)
        prediction = response_rows(intrinsic, block["widths"], block["offsets"])
        return prediction, intrinsic, tau_lines

    @jax.jit
    def full_derivative(theta, intrinsic, tau_lines, opacity_jac):
        dz = scales[:, None, :]*upper[None, None, :]*tau_lines.T[None, :, :]
        dq = jnp.einsum("em,mgk->egmk", scales[:, selected]*(1+upper[selected]*theta[:n][selected]), opacity_jac)
        dtau = jnp.concatenate([dz, dq.reshape(dz.shape[0], dz.shape[1], 2*m)], axis=-1)
        dflux = -airmass[:, None, None]*intrinsic[:, :, None]*dtau
        return jax.vmap(response_rows, in_axes=(2, None, None), out_axes=2)(
            dflux, block["widths"], block["offsets"])

    cache = {}

    def evaluate(theta, jacobian=False):
        theta = np.asarray(theta)
        if "theta" not in cache or not np.array_equal(theta, cache["theta"]):
            q = jnp.asarray(theta[n:].reshape(m, 2))
            changed = values(q)
            pred, intrinsic, tau_lines = forward_parts(jnp.asarray(theta), changed)
            cache.clear()
            cache.update(theta=theta.copy(), q=q, prediction=np.asarray(pred),
                         intrinsic=intrinsic, tau_lines=tau_lines)
        if jacobian and "jacobian" not in cache:
            cache["jacobian"] = np.asarray(full_derivative(jnp.asarray(theta), cache["intrinsic"],
                                                          cache["tau_lines"], derivative(cache["q"])))
        return cache["jacobian"] if jacobian else cache["prediction"]

    return evaluate


def build_context():
    context = original_context()
    meta = json.loads((OUT / "motion_clean_control.json").read_text())
    assert not meta["reference_self_broadening"] and meta["pressure_shift"]
    profile = load_profile(meta["layers"])
    with np.load(OUT / "adopted_telluric_fit.npz") as saved:
        contrast = saved["line_contrast"]
        adopted_z = saved["parameters"][saved["line_parameter_indices"]]
        adopted_prediction = saved["component"]
    cursor, initial, records, line_indices, shape_indices = 0, [], [], [], []
    for b, block in enumerate(context["blocks"]):
        old = block["parameter_slice"]
        n = len(block["records"])
        selected = np.array([i for i, r in enumerate(block["records"])
                             if r["species"] == "H2O" and contrast[old][i] > .02], dtype=int)
        block["selected"] = selected
        block["strength_prior"] = 1+9*contrast[old]**2/(contrast[old]**2+.005**2)
        shape_priors = []
        for i in selected:
            r = block["records"][i]
            width = FINITE_UPPER[int(r["ierr"][2])]
            center = CENTER_UPPER[int(r["ierr"][0])]*c/r["nu"]
            shape_priors.extend([np.log1p(width), center])
            records.append(dict(r, original_contrast=float(contrast[old][i]),
                                log_width_prior=float(np.log1p(width)), center_prior_kms=float(center),
                                width_parameter_index=cursor+n+len(shape_priors)-2,
                                center_parameter_index=cursor+n+len(shape_priors)-1))
        block["shape_priors"] = np.array(shape_priors)
        block["shape_slice"] = slice(cursor, cursor+n+2*len(selected))
        block["evaluate_shape"] = prepare_block(block, selected, profile)
        log_wave = np.log(context["wave"][b])
        x = 2*(log_wave-log_wave.min())/np.ptp(log_wave)-1
        block["basis"] = np.polynomial.chebyshev.chebvander(x, 5)
        block["inverse"] = np.linalg.pinv(block["basis"][context["allowed"][b]])
        initial.extend(adopted_z[old]); initial.extend(np.zeros(2*len(selected)))
        line_indices.extend(range(cursor, cursor+n))
        shape_indices.extend(range(cursor+n, cursor+n+2*len(selected)))
        cursor += n+2*len(selected)
        print(f"Prepared order {block['order']}: {len(selected)} H2O profiles", flush=True)
    context.update(initial=np.array(initial), n_parameters=cursor, shape_records=records,
                   line_parameter_indices=np.array(line_indices), shape_parameter_indices=np.array(shape_indices))
    context["adopted_component"] = adopted_prediction
    pred = predict_components(context, context["initial"])
    context["adopted_reconstruction_max_error"] = float(np.max(abs(pred-adopted_prediction)))
    np.testing.assert_allclose(pred, adopted_prediction, atol=1e-10, rtol=0)
    return context


def predict_components(context, theta):
    return np.array([b["evaluate_shape"](theta[b["shape_slice"]]) for b in context["blocks"]])


def filtered_block(block, values, mask):
    if values.ndim == 2:
        return values-(values[:, mask] @ block["inverse"].T) @ block["basis"].T
    coefficients = np.einsum("cp,epk->eck", block["inverse"], values[:, mask])
    return values-np.einsum("pc,eck->epk", block["basis"], coefficients)


def fit_shapes(context, config, target=None, initial=None, train_indices=None):
    target = context["y_over_c"] if target is None else np.asarray(target)
    theta = context["initial"].copy() if initial is None else np.array(initial, copy=True)
    train = np.arange(5) if train_indices is None else np.asarray(train_indices, dtype=int)
    if not len(train) or len(np.unique(train)) != len(train) or np.any((train < 0) | (train >= 5)):
        raise ValueError("Use unique training indices in [0,5)")
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        n, m = len(block["records"]), len(block["selected"])
        free_shapes = np.tile([config["mode"] in ["width", "joint"],
                              config["mode"] in ["center", "joint"]], m)
        free = np.r_[np.ones(n, dtype=bool), free_shapes]
        prior = np.r_[block["strength_prior"], block["shape_priors"]*config["prior_multiplier"]]
        lower = np.r_[-1/block["upper"], np.tile([-np.log(2.), -1.], m)]
        upper = np.r_[np.full(n, np.inf), np.tile([np.log(2.), 1.], m)]
        start = theta[block["shape_slice"]].copy()
        start[~free] = 0.
        start = np.clip(start, lower+1e-10, upper-1e-10)[free]
        mask = context["allowed"][b]

        def expand(x):
            full = np.zeros(n+2*m)
            full[free] = x
            return full

        def data(x):
            residual = target[b, :5]-block["evaluate_shape"](expand(x))[:5]
            return filtered_block(block, residual, mask)[train][:, mask].ravel()/WORKING_SIGMA

        def fun(x):
            return np.r_[data(x), x/prior[free]]

        def jac(x):
            derivative = filtered_block(block, block["evaluate_shape"](expand(x), True)[:5], mask)
            return np.vstack([-derivative[train][:, mask].reshape(-1, n+2*m)[:, free]/WORKING_SIGMA,
                              np.diag(1/prior[free])])

        fit = least_squares(fun, start, jac=jac, bounds=(lower[free], upper[free]),
                            x_scale="jac", max_nfev=150, ftol=1e-8, xtol=1e-9, gtol=1e-7)
        full = expand(fit.x)
        theta[block["shape_slice"]] = full
        diagnostics.append(dict(order=int(block["order"]), success=bool(fit.success),
            message=fit.message, nfev=int(fit.nfev), optimality=float(fit.optimality),
            training_indices=train.tolist(), n_free=int(free.sum()),
            data_objective=float(np.sum(data(fit.x)**2)), prior_objective=float(np.sum((fit.x/prior[free])**2)),
            shape_at_bound_count=int(np.sum(np.minimum(full[n:]-lower[n:], upper[n:]-full[n:]) < 1e-5))))
    if not all(d["success"] for d in diagnostics):
        raise RuntimeError(f"Nonconverged fit: {diagnostics}")
    return theta, diagnostics


def validate(context):
    results = []
    for b, block in enumerate(context["blocks"]):
        theta = context["initial"][block["shape_slice"]].copy()
        n = len(block["records"])
        rng = np.random.default_rng(473+b)
        if len(theta) > n:
            theta[n:] = np.tile([.03, .02], len(block["selected"]))
        direction = rng.normal(size=len(theta))
        direction[n:] *= .05
        direction /= np.linalg.norm(direction)
        step = 1e-4
        before = time.monotonic()
        analytic = block["evaluate_shape"](theta, True) @ direction
        finite = (block["evaluate_shape"](theta+step*direction)-block["evaluate_shape"](theta-step*direction))/(2*step)
        relative = float(np.linalg.norm(analytic-finite)/np.linalg.norm(finite))
        if relative > 2e-5:
            raise ValueError(f"Directional derivative mismatch: {relative}")
        results.append(dict(order=int(block["order"]), derivative_relative_error=relative,
                            max_absolute_error=float(np.max(abs(analytic-finite))),
                            seconds=time.monotonic()-before))
        print("Validation", results[-1], flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    started = time.monotonic()
    context = build_context()
    validation = validate(context)
    if args.benchmark:
        print("Build and validation seconds", time.monotonic()-started, flush=True)
        return
    plan = dict(cases=CASES, selected_species="H2O", selection="Original predicted contrast > 2%, frozen before fitting",
        line_count=len(context["shape_records"]), shape_parameterization="gamma_air *= exp(q_width); nu += nu*q_center/c",
        prior="Width log1p(relative uncertainty upper edge); center absolute uncertainty upper edge converted to km/s; multiplier 1 or 3",
        uncertainty_caveat="HITRAN error classes are chosen regularization scales, not independent calibrated Gaussian sigmas",
        bounds="Width factor 0.5 to 2; center displacement -1 to +1 km/s",
        fixed="Original IP, TP, columns, pressure shift coefficients, n_air, wavelength solution, stellar model and Q5 masks",
        strength_prior="Adopted contrast10 with nonnegative strengths, all 269 original strength parameters refitted",
        selection_rule="Minimum pooled leave-one-training-exposure-out SSE; evaluation data never select configuration",
        scope="Conditional on earlier fitted nuisance parameters, not end-to-end cross-validation")
    (OUT/"line_shape_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    cv, cv_arrays, fitted = {}, {}, {}
    # First fit each predetermined case on all training spectra; do not inspect evaluation metrics.
    for config in CASES:
        theta, diag = fit_shapes(context, config)
        fitted[config["name"]] = (theta, diag)
        print("Full training fit", config["name"], [d["nfev"] for d in diag], flush=True)
        folds = []
        for held in range(5):
            fitted_theta, fit = fit_shapes(context, config, initial=theta,
                                           train_indices=[i for i in range(5) if i != held])
            pred = predict_components(context, fitted_theta)
            residual = context["project"](context["y_over_c"][:, :5]-pred[:, :5])[:, held]
            sse = float(np.sum(residual[context["allowed"]]**2))
            folds.append(dict(heldout_exposure=int(context["exposures"][held]), sse=sse, fit=fit))
            cv_arrays[f"theta_{config['name']}_fold{held}"] = fitted_theta
        total = sum(f["sse"] for f in folds)
        cv[config["name"]] = dict(config=config, folds=folds, sse=total,
                                  rms_percent=float(100*np.sqrt(total/(5*context["allowed"].sum()))))
        (OUT/"line_shape_cv_checkpoint.json").write_text(json.dumps(cv, indent=2)+"\n")
        print("CV", config["name"], cv[config["name"]]["rms_percent"], flush=True)
    ranking = sorted(cv, key=lambda name: cv[name]["sse"])
    selection = dict(selected_case=ranking[0], ranking=ranking, cross_validation=cv)
    (OUT/"line_shape_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    np.savez_compressed(OUT/"line_shape_cv.npz", **cv_arrays)
    cases, arrays = {}, {}
    for config in CASES:
        name = config["name"]
        theta, diag = fitted[name]
        component = predict_components(context, theta)
        statistic, residual = metrics(context, component)
        cases[name] = dict(config=config, fit=diag, metrics=statistic)
        arrays.update({f"theta_{name}": theta, f"component_{name}": component, f"residual_{name}": residual})
        print("Evaluation", name, statistic["evaluation_score"]["rms_percent"], flush=True)
    sources = context["source_files"]+[Path(__file__), ROOT/"fit_line_strength.py", ROOT/"line_strength_support.py",
        ROOT/"atmosphere_profile.py", ROOT/"cache/H2O.par", ROOT/"cache/H2O.npz",
        OUT/"adopted_telluric_model.json", OUT/"adopted_telluric_fit.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=plan, selection=selection,
        cases=cases, shape_records=context["shape_records"], validation=validation,
        adopted_reconstruction_max_error=context["adopted_reconstruction_max_error"],
        seconds=time.monotonic()-started,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"line_shape_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"line_shape_fit.npz", orders=context["orders"], exposures=context["exposures"],
        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
        y_over_c=context["y_over_c"], continuum=context["continuum"],
        line_parameter_indices=context["line_parameter_indices"], shape_parameter_indices=context["shape_parameter_indices"],
        theta_adopted=context["initial"], component_adopted=context["adopted_component"], **arrays)


if __name__ == "__main__":
    main()
