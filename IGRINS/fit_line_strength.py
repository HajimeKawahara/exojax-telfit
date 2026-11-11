"""Fit shared HITRAN intensity corrections with explicit uncertainty regularization.

This is a conditional fit: atmosphere, instrument, star and original continuum
are fixed. The adopted degree-five polynomial is profiled in continuum units.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from line_strength_support import native_grid, prepare_line_tau, select_lines
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TRAIN = [0, 19, 39, 59, 78]
EVALUATION = [10, 30, 50, 70]
WORKING_SIGMA = .0045
PLAN = dict(
    training_exposures=TRAIN, evaluation_exposures=EVALUATION,
    correction="S_i = S_i,HITRAN * (1 + u_i*z_i); shared across all exposures",
    uncertainty_upper={4: .20, 5: .10, 6: .05, 7: .02, 8: .01, 9: .001},
    excluded_codes=[0, 1, 2, 3], reference_strength_min=1e-25,
    nearest_included_velocity_limit_kms=10., z_bounds=[-1., 1.],
    primary_prior_scale=1., sensitivity_prior_scales=[.5, 2.],
    penalty="sum((z_i/prior_scale)^2); bin upper edges are assumed scales, not HITRAN Gaussian sigmas",
    working_residual_scale=WORKING_SIGMA,
    objective="sum((Q*(observed/original_C - exact_component)/working_sigma)^2) + penalty",
    filter="Q removes degree-five polynomials on all included pixels per exposure/order",
    fixed="TP, columns/abundances, wavelength calibration, IP, stellar parameters, original continuum",
    interpretation="Conditional regularized fit, not a calibrated posterior or a revision of HITRAN",
)


def load_fit(tag):
    meta = json.loads((OUT / f"{tag}.json").read_text())
    with np.load(OUT / f"{tag}.npz") as data:
        return meta, {key: data[key] for key in data.files}


def make_forward(block):
    """Keep opacity changes inside exponentiation and before instrumental response."""
    grid, velocity = block["grid"], block["velocity"]
    line_tau, upper = jnp.asarray(block["line_tau"]), jnp.asarray(block["upper"])
    baseline_tau = jnp.asarray(block["baseline_tau"])
    line_factors = jnp.asarray(block["line_factors"])
    shifted_star = jnp.asarray(block["shifted_star"])
    airmass = jnp.asarray(block["airmass"])
    widths, offsets = jnp.asarray(block["widths"]), jnp.asarray(block["offsets"])
    target, fraction = block["target"], block["wing_fraction"]

    def component(z, n_exposures):
        delta_tau = (line_factors[:n_exposures]*(upper*z)[None, :]) @ line_tau
        intrinsic = shifted_star[:n_exposures]*jnp.exp(
            -airmass[:n_exposures, None]*(baseline_tau[:n_exposures]+delta_tau))

        def response(row, width, rv):
            core = sampling(target, grid, ipgauss(row, velocity, width), rv)
            wing = sampling(target, grid, ipgauss(row, velocity, 2*width), rv)
            return (1-fraction)*core+fraction*wing

        return jax.vmap(response)(intrinsic, widths[:n_exposures], offsets[:n_exposures])

    return jax.jit(lambda z: component(z, len(TRAIN))), jax.jit(lambda z: component(z, len(TRAIN)+len(EVALUATION)))


def build_context(source_tag="motion_clean_control", evaluation_prefix="nonstellar_control_e",
                  species=("H2O", "CO2")):
    """Build exact model callbacks and fixed masks without fitting any corrections."""
    meta, source = load_fit(source_tag)
    species = tuple(species)
    if tuple(meta["species"]) != species:
        raise ValueError("Requested species must match the physical source fit.")
    assert source["exposures"][:-1].tolist() == TRAIN
    orders, wave, allowed = [source[key] for key in ["orders", "wavelength_um", "included_mask"]]
    score = source["score_mask"][:, -1]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    profile = load_profile(meta["layers"])
    selections, selection_metadata = [], []
    for b, order in enumerate(orders):
        per_order, descriptions = [], []
        for molecule in species:
            records, description = select_lines(molecule, wave[b], allowed[b],
                                                PLAN["reference_strength_min"],
                                                PLAN["nearest_included_velocity_limit_kms"])
            per_order.extend([dict(record, order=int(order)) for record in records])
            descriptions.append(description)
        selections.append(per_order)
        selection_metadata.append(dict(order=int(order), species=descriptions))

    # Line selection above depends only on metadata and the frozen mask.
    observed = [source["observed"][:, :-1]]
    continua = [source["continuum"][:, :-1]]
    reference = [source["component"][:, :-1]]
    parameters = [meta["parameters"] for _ in TRAIN]
    source_files = [OUT / f"{source_tag}.json", OUT / f"{source_tag}.npz"]
    for exposure in EVALUATION:
        other_meta, other = load_fit(f"{evaluation_prefix}{exposure}")
        np.testing.assert_array_equal(other["included_mask"], allowed)
        np.testing.assert_array_equal(other["score_mask"][:, -1], score)
        np.testing.assert_array_equal(other["component"][:, :-1], source["component"][:, :-1])
        observed.append(other["observed"][:, -1:])
        continua.append(other["continuum"][:, -1:])
        reference.append(other["component"][:, -1:])
        parameters.append(other_meta["parameters"])
        source_files += [OUT / f"{evaluation_prefix}{exposure}.json", OUT / f"{evaluation_prefix}{exposure}.npz"]
    observed = np.concatenate(observed, axis=1)
    continua = np.concatenate(continua, axis=1)
    reference = np.concatenate(reference, axis=1)
    exposures = np.asarray(TRAIN+EVALUATION)
    airmass = info.iloc[exposures]["Air Mass"].to_numpy()
    relative_rv = info.iloc[exposures]["RV [km/s]"].to_numpy()-info.iloc[0]["RV [km/s]"]
    p = meta["parameters"]
    blocks, all_records, bases, inverses = [], [], [], []
    for b, order in enumerate(orders):
        grid = native_grid(wave[b], meta["dv_kms"])
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, p["stellar_extra_sigma_kms"])
        shifted_star = np.array([sampling(grid, grid, star, p["stellar_rv_reference_kms"]+rv)
                                 for rv in relative_rv])
        taus, individual, tau_metadata = {}, [], []
        for molecule in species:
            db = load_mdb(molecule, grid[0]-meta["line_margin_cm1"], grid[-1]+meta["line_margin_cm1"], meta["line_cutoff"])
            taus[molecule] = profile_tau(OpaDirect(db, grid), grid, profile, molecule,
                                        meta["pressure_shift"], meta["reference_self_broadening"])[0]
            records = [record for record in selections[b] if record["species"] == molecule]
            values, tau_meta = prepare_line_tau(molecule, records, grid, profile,
                                                meta["pressure_shift"], meta["reference_self_broadening"])
            if np.max(values.sum(axis=0)-taus[molecule]) > 1e-12:
                raise ValueError("Selected lines exceed the baseline optical depth")
            individual.append(values); tau_metadata.append(tau_meta)
        scales = {
            molecule: np.array([10**par[f"log10_water_e{exposure}" if molecule == "H2O"
                                       else f"log10_scale_{molecule}"]
                                for par, exposure in zip(parameters, exposures)])
            for molecule in species
        }
        widths = np.array([par[f"ip_sigma_o{order}_kms"]*np.exp(par.get(f"log_ip_width_e{e}", 0.))
                           for par, e in zip(parameters, exposures)])
        offsets = np.array([par[f"calibration_rv_o{order}_kms"]+par.get(f"calibration_rv_e{e}_kms", 0.)
                            for par, e in zip(parameters, exposures)])
        records = selections[b]
        line_factors = np.array([scales[record["species"]] for record in records]).T
        target = (1e4/wave[b])*(1+p[f"stretch_o{order}_kms"]*np.linspace(-1, 1, len(wave[b]))/c)
        sl = slice(len(all_records), len(all_records)+len(records))
        all_records.extend(records)
        block = dict(order=int(order), grid=jnp.asarray(grid), velocity=velocity,
                     line_tau=np.concatenate(individual), upper=np.array([r["uncertainty_upper"] for r in records]),
                     baseline_tau=sum(scales[molecule][:, None]*taus[molecule] for molecule in species),
                     line_factors=line_factors, shifted_star=shifted_star, airmass=airmass,
                     widths=widths, offsets=offsets, target=jnp.asarray(target),
                     wing_fraction=p["ip_wing_fraction"], records=records, parameter_slice=sl,
                     tau_metadata=tau_metadata)
        block["train_forward"], block["forward"] = make_forward(block)
        log_v = c*np.log(wave[b]/np.exp(np.mean(np.log(wave[b]))))
        x = 2*(log_v-log_v.min())/np.ptp(log_v)-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        inverse = np.linalg.pinv(basis[allowed[b]])
        bases.append(basis); inverses.append(inverse)
        blocks.append(block)
        print(f"Order {order}: {len(records)} shared line corrections", flush=True)

    def project(array):
        array = np.asarray(array)
        return np.array([row-(row[..., mask] @ inv.T) @ basis.T
                         for row, mask, inv, basis in zip(array, allowed, inverses, bases)])

    context = dict(wave=wave, allowed=allowed, score=score, orders=orders, exposures=exposures,
                   y_over_c=observed/continua, reference_component=reference, continuum=continua,
                   project=project, blocks=blocks, records=all_records, n_parameters=len(all_records),
                   selection_metadata=selection_metadata, source_files=source_files,
                   physical_parameters=p, species=species, physical_metadata=meta)
    for b, block in enumerate(blocks):
        prepare_fit_callbacks(block, allowed[b], bases[b], inverses[b])
    reconstruction = predict_components(context, np.zeros(len(all_records)))
    error = float(np.max(np.abs(reconstruction-reference)))
    if error > 1e-11:
        raise ValueError(f"Baseline model reconstruction failed: {error}")
    context["baseline_reconstruction_error"] = error
    return context


def prepare_fit_callbacks(block, mask, basis, inverse):
    forward = block["train_forward"]
    basis, inverse = jnp.asarray(basis), jnp.asarray(inverse)

    def residual(z, target):
        rows = target-forward(z)
        filtered = rows-(rows[:, mask] @ inverse.T) @ basis.T
        return filtered[:, mask].ravel()/WORKING_SIGMA

    block["data_residual"] = jax.jit(residual)
    block["data_jacobian"] = jax.jit(jax.jacfwd(residual, argnums=0))


def predict_components(context, theta):
    return np.array([np.asarray(block["forward"](theta[block["parameter_slice"]]))
                     for block in context["blocks"]])


def fit_strengths(context, target=None, prior_scale=1., initial=None):
    """Fit only the five training exposures; target may contain injected signals."""
    target = context["y_over_c"] if target is None else np.asarray(target)
    initial = np.zeros(context["n_parameters"]) if initial is None else np.asarray(initial)
    theta = initial.copy()
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        sl = block["parameter_slice"]
        training_target = jnp.asarray(target[b, :len(TRAIN)])
        z0 = np.clip(initial[sl], -1+1e-10, 1-1e-10)
        n = len(z0)

        def fun(z):
            return np.concatenate([np.asarray(block["data_residual"](z, training_target)), z/prior_scale])

        def jac(z):
            return np.vstack([np.asarray(block["data_jacobian"](z, training_target)), np.eye(n)/prior_scale])

        fit = least_squares(fun, z0, jac=jac, bounds=(-np.ones(n), np.ones(n)),
                            method="trf", x_scale=1., max_nfev=100,
                            ftol=1e-9, xtol=1e-9, gtol=1e-7)
        theta[sl] = fit.x
        data_residual = np.asarray(block["data_residual"](fit.x, training_target))
        diagnostics.append(dict(order=block["order"], success=bool(fit.success),
                                nfev=int(fit.nfev), optimality=float(fit.optimality),
                                message=fit.message, data_objective=float(data_residual@data_residual),
                                prior_objective=float(np.sum((fit.x/prior_scale)**2)),
                                at_cap_count=int(np.sum(np.abs(fit.x)>.999)),
                                free_parameter_count=n))
        print(f"Order {block['order']} prior {prior_scale:g}: {fit.nfev} iterations, caps {diagnostics[-1]['at_cap_count']}/{n}", flush=True)
    if not all(record["success"] for record in diagnostics):
        raise ValueError("A line-strength optimization did not converge")
    return theta, diagnostics


def metrics(context, component):
    residual = context["project"](context["y_over_c"]-component)
    results = {}
    for scope, es, mask in [("training", slice(0, 5), context["allowed"]),
                            ("evaluation_score", slice(5, None), context["score"]),
                            ("evaluation_included", slice(5, None), context["allowed"])]:
        rows = residual[:, es]
        keep = np.broadcast_to(mask[:, None], rows.shape)
        fixed = rows*context["continuum"][:, es]
        results[scope] = dict(n_samples=int(keep.sum()),
                             rms_percent=float(100*np.sqrt(np.mean(rows[keep]**2))),
                             fixed_rms_percent=float(100*np.sqrt(np.mean(fixed[keep]**2))),
                             per_order={str(order): float(100*np.sqrt(np.mean(rows[b][keep[b]]**2)))
                                        for b, order in enumerate(context["orders"])})
    return results, residual


def main():
    started = time.monotonic()
    (OUT / "line_strength_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    context = build_context()
    zero = np.zeros(context["n_parameters"])
    baseline, baseline_residual = metrics(context, context["reference_component"])
    np.testing.assert_allclose(baseline["evaluation_score"]["rms_percent"], .9265650432361973, atol=1e-10)
    derivative_checks = []
    for b, block in enumerate(context["blocks"]):
        z = zero[block["parameter_slice"]]
        y = jnp.asarray(context["y_over_c"][b, :5])
        direction = np.random.default_rng(137+b).normal(size=len(z))
        direction /= np.linalg.norm(direction)
        step = 1e-5
        finite = (np.asarray(block["data_residual"](z+step*direction, y))
                  - np.asarray(block["data_residual"](z-step*direction, y)))/(2*step)
        exact = np.asarray(block["data_jacobian"](z, y))@direction
        error = float(np.linalg.norm(exact-finite)/max(np.linalg.norm(exact), 1e-30))
        if error > 1e-5:
            raise ValueError(f"Derivative check failed: {error}")
        derivative_checks.append(dict(order=block["order"], relative_error=error))
    cases, arrays, csv_rows = {}, {}, []
    for tag, prior in [("primary", 1.), ("half_prior_width", .5), ("double_prior_width", 2.)]:
        theta, diagnostics = fit_strengths(context, prior_scale=prior)
        components = predict_components(context, theta)
        case_metrics, residual = metrics(context, components)
        species_statistics = {}
        for species in ["H2O", "CO2"]:
            selected = np.array([record["species"] == species for record in context["records"]])
            upper = np.array([record["uncertainty_upper"] for record in context["records"]])[selected]
            delta = upper*theta[selected]
            weights = np.array([record["sw"] for record in context["records"]])[selected]
            species_statistics[species] = dict(n_lines=int(selected.sum()),
                mean_fractional_correction=float(np.mean(delta)),
                strength_weighted_mean_fractional_correction=float(np.average(delta, weights=weights)),
                rms_fractional_correction=float(np.sqrt(np.mean(delta**2))))
        cases[tag] = dict(prior_scale=prior, fit=diagnostics, metrics=case_metrics,
                          species_statistics=species_statistics,
                          at_cap_count=int(np.sum(np.abs(theta)>.999)))
        arrays[f"theta_{tag}"] = theta
        arrays[f"component_{tag}"] = components
        arrays[f"residual_{tag}"] = residual
        for index, (record, z) in enumerate(zip(context["records"], theta)):
            csv_rows.append(dict(case=tag, index=index, order=record["order"], species=record["species"],
                                 raw_index=record["raw_index"], nu_cm1=record["nu"], iso=record["iso"],
                                 ierr=record["ierr"], reference_strength=record["sw"],
                                 strength_code=record["strength_code"], upper_fraction=record["uncertainty_upper"],
                                 z=float(z), fractional_correction=float(z*record["uncertainty_upper"]),
                                 at_cap=bool(abs(z)>.999)))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN,
                  n_parameters=context["n_parameters"], line_records=context["records"],
                  selection_metadata=context["selection_metadata"], baseline=baseline, cases=cases,
                  baseline_reconstruction_error=context["baseline_reconstruction_error"],
                  derivative_checks=derivative_checks,
                  tau_metadata=[block["tau_metadata"] for block in context["blocks"]],
                  seconds=time.monotonic()-started,
                  limitations=["HITRAN finite bin edges define assumed caps and shrinkage scales, not calibrated confidence intervals.",
                               "Nonfinite uncertainty codes and weaker/distant lines stay fixed.",
                               "Per-species common strength changes are confounded with frozen molecular columns.",
                               "This conditional fit cannot establish that recovered shifts are true line-list errors.",
                               "Working diagonal residual weight ignores measured pixel correlations; no formal significance or posterior error is reported.",
                               "All-included polynomial filtering uses evaluation score pixels; only line strengths are trained exclusively on the five training exposures.",
                               "The spectra and mask were explored previously; evaluation is not a pristine new model-selection experiment."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in context["source_files"]+[Path(__file__), ROOT/"line_strength_support.py",
                                     ROOT/"cache/H2O.par", ROOT/"cache/CO2.par"]})
    (OUT / "line_strength_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    with (OUT / "line_strength_corrections.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
    np.savez_compressed(OUT / "line_strength_fit.npz", orders=context["orders"], exposures=context["exposures"],
                        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
                        y_over_c=context["y_over_c"], continuum=context["continuum"],
                        component_baseline=context["reference_component"], residual_baseline=baseline_residual, **arrays)
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for b, ax in enumerate(axes):
        mask = context["score"][b]
        for key, label, color in [("baseline", "Fixed HITRAN strengths", "0.6"),
                                   ("primary", "Uncertainty-constrained strengths", "C0")]:
            data = baseline_residual if key == "baseline" else arrays[f"residual_{key}"]
            mean = np.where(mask, np.mean(data[b, 5:], axis=0), np.nan)
            ax.plot(context["wave"][b], 100*mean, lw=.9, label=label, color=color)
        ax.axhline(0, color="0.3", lw=.5)
        ax.set(xlabel="Observed wavelength (micron)", ylabel="Filtered mean residual (%)", title=f"Order {context['orders'][b]}")
        ax.legend(fontsize=8)
    fig.savefig(OUT / "line_strength_fit.png", dpi=160); plt.close(fig)
    print(json.dumps(dict(baseline=baseline, cases=cases, seconds=result["seconds"]), indent=2))


if __name__ == "__main__":
    main()
