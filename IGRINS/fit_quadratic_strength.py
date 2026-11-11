"""Fit quadratic instrumental widths and regularized effective line strengths.

Weak-line prior widths depend on predicted, masked spectral contrast, never
on the observed residual. Existing experiments remain unchanged.
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

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_line_strength import build_context as original_context, metrics, WORKING_SIGMA
from fit_relaxed_strength import correction_summary as original_correction_summary
from quadratic_ip_response import prepare_block

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CONTRAST_SCALE = .005
IP_PRIOR = np.full(3, .15)
IP_BOUND = np.array([np.log(1.5), .3, .3])
FAMILIES = ["fixed", "quadratic_gaussian", "quadratic_mixture"]
CASES = [dict(name=f"{family}_contrast{k}", ip_family=family,
              strength_prior="contrast", prior_scale=float(k))
         for family in FAMILIES for k in [3, 10]]
CV_NAMES = [case["name"] for case in CASES]
CASES += [dict(name=f"{family}_{prior}", ip_family=family,
               strength_prior=prior, prior_scale=10.)
          for family in FAMILIES[1:] for prior in ["fixed", "uniform"]]
CASES += [dict(name="fixed_uniform", ip_family="fixed",
               strength_prior="uniform", prior_scale=10.)]
PLAN = dict(
    cases=CASES, cross_validation_cases=CV_NAMES,
    strength_model="S_i=S_i,HITRAN*(1+u_i*z_i), with S_i>=0 and no upper cap",
    contrast="Maximum absolute original-model dF/d(delta S/S) before poly5, over five training exposures and included pixels",
    contrast_scale=CONTRAST_SCALE,
    strength_prior="z_i/[1+(k-1)*d_i^2/(d_i^2+d0^2)]; weak lines approach the nominal bin scale",
    uniform_control="z_i/k", uncertainty_note="HITRAN bin upper edges are chosen regularization scales, not calibrated Gaussian sigmas",
    ip_model="Quadratic FWHM in pixel coordinate, proportional to exp(q0)*(1+q1*x+q2*x^2); x in [-1,1]",
    ip_bounds=IP_BOUND.tolist(), ip_prior_widths=IP_PRIOR.tolist(),
    ip_reference="https://arxiv.org/abs/2104.02082 section 3.6; functional form only, no imported calibration coefficients",
    ip_families="Single Gaussian and previous fixed-fraction two-Gaussian mixture; source-normalized transport on padded log-wavenumber grid",
    parameter_sharing="All new line and IP coefficients are shared across exposures; old epoch width factors remain fixed",
    fixed="TP, molecular columns, stellar model, wavelength solution, original continuum, masks and degree-five projection",
    selection="Minimum pooled leave-one-training-exposure-out error among six contrast-prior cases; final evaluation data do not select the case",
    selection_limitation="Conditional on previously fitted nuisance parameters, including epoch IP factors; not end-to-end independent CV",
    training_exposures=[0, 19, 39, 59, 78], evaluation_exposures=[10, 30, 50, 70],
)


def build_context():
    """Reuse opacity and masks; freeze weak-line weights before fitting."""
    context = original_context()
    line_indices, ip_indices, contrasts, cursor = [], [], [], 0
    for b, block in enumerate(context["blocks"]):
        n = len(block["records"])
        derivative = np.asarray(jax.jit(jax.jacfwd(block["train_forward"]))(jnp.zeros(n)))
        contrast = np.max(np.abs(derivative[:, context["allowed"][b]]), axis=(0, 1))/block["upper"]
        block["line_contrast"] = contrast
        contrasts.extend(contrast)
        block["original_parameter_slice"] = block["parameter_slice"]
        block["parameter_slice"] = slice(cursor, cursor+n+3)
        line_indices.extend(range(cursor, cursor+n))
        ip_indices.extend(range(cursor+n, cursor+n+3))
        cursor += n+3
        block["models"] = {}
        log_wave = np.log(context["wave"][b])
        x = 2*(log_wave-log_wave.min())/np.ptp(log_wave)-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        inverse = np.linalg.pinv(basis[context["allowed"][b]])
        for family in FAMILIES:
            mode = "constant_velocity" if family == "fixed" else "quadratic_pixel"
            fraction = 0. if family == "quadratic_gaussian" else block["wing_fraction"]
            model = prepare_block(block, context["wave"][b], mode=mode, wing_fraction=fraction)
            if family == "fixed":
                # Reuse the equivalent FFT response for the unchanged-IP control.
                old_train, old_full = block["train_forward"], block["forward"]
                model["train_forward"] = jax.jit(lambda p, fn=old_train, n=n: fn(p[:n]))
                model["forward"] = jax.jit(lambda p, fn=old_full, n=n: fn(p[:n]))
                model["train_jacobian"] = jax.jit(jax.jacfwd(model["train_forward"]))
            forward = model["train_forward"]

            def residual(theta, target, forward=forward, mask=context["allowed"][b],
                         basis=jnp.asarray(basis), inverse=jnp.asarray(inverse)):
                rows = target-forward(theta)
                filtered = rows-(rows[:, mask] @ inverse.T) @ basis.T
                return filtered[:, mask].ravel()/WORKING_SIGMA

            model["data_residual"] = jax.jit(residual)

            def derivative(theta, target, raw=model["train_jacobian"], mask=context["allowed"][b],
                           basis=jnp.asarray(basis), inverse=jnp.asarray(inverse)):
                values = raw(theta)
                coefficients = jnp.einsum("cp,epk->eck", inverse, values[:, mask])
                filtered = values-jnp.einsum("pc,eck->epk", basis, coefficients)
                return -filtered[:, mask].reshape(-1, values.shape[-1])/WORKING_SIGMA

            model["data_jacobian"] = jax.jit(derivative)
            block["models"][family] = model
    context["n_parameters"] = cursor
    context["line_parameter_indices"] = np.array(line_indices, dtype=int)
    context["ip_parameter_indices"] = np.array(ip_indices, dtype=int)
    context["line_contrast"] = np.array(contrasts)
    context["contrast_weight"] = context["line_contrast"]**2/(context["line_contrast"]**2+CONTRAST_SCALE**2)
    fixed = predict_components(context, np.zeros(cursor), CASES[0])
    error = float(np.max(np.abs(fixed-context["reference_component"])))
    if error > 1e-10:
        raise ValueError(f"Fixed-IP reconstruction failed: {error}")
    context["fixed_ip_reconstruction_error"] = error
    return context


def prior_widths(block, config):
    """Return widths in z coordinates, with no residual-dependent tuning."""
    k = config["prior_scale"]
    if config["strength_prior"] == "contrast":
        d = block["line_contrast"]
        return 1+(k-1)*d*d/(d*d+CONTRAST_SCALE**2)
    return np.full(len(block["records"]), k)


def predict_components(context, theta, config):
    return np.array([np.asarray(block["models"][config["ip_family"]]["forward"](
        theta[block["parameter_slice"]])) for block in context["blocks"]])


def fit_joint_strength(context, config, target=None, initial=None, train_indices=None):
    """Fit shared intensity and IP coefficients on the chosen training rows."""
    target = context["y_over_c"] if target is None else np.asarray(target)
    theta = np.zeros(context["n_parameters"]) if initial is None else np.asarray(initial).copy()
    indices = np.arange(5) if train_indices is None else np.asarray(train_indices, dtype=int)
    if len(indices) == 0 or len(np.unique(indices)) != len(indices) or np.any((indices < 0) | (indices >= 5)):
        raise ValueError("Use unique training indices in [0,5)")
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        n = len(block["records"])
        free = np.array(([True]*n if config["strength_prior"] != "fixed" else [False]*n)
                        + ([True]*3 if config["ip_family"] != "fixed" else [False]*3))
        model = block["models"][config["ip_family"]]
        lower = np.r_[-1/block["upper"], -IP_BOUND]
        upper = np.r_[np.full(n, np.inf), IP_BOUND]
        scales = np.r_[prior_widths(block, config), IP_PRIOR]
        start = np.clip(theta[block["parameter_slice"]], lower+1e-10, upper-1e-10)[free]
        npixels = int(context["allowed"][b].sum())
        train_target = jnp.asarray(target[b, :5])

        def expand(values):
            full = np.zeros(n+3)
            full[free] = values
            return full

        def data(values):
            return np.asarray(model["data_residual"](expand(values), train_target)).reshape(5, npixels)[indices].ravel()

        def fun(values):
            return np.r_[data(values), values/scales[free]]

        def jac(values):
            derivative = np.asarray(model["data_jacobian"](expand(values), train_target)).reshape(5, npixels, n+3)
            return np.vstack([derivative[indices].reshape(-1, n+3)[:, free], np.diag(1/scales[free])])

        fit = least_squares(fun, start, jac=jac, bounds=(lower[free], upper[free]),
                            method="trf", x_scale="jac", max_nfev=200,
                            ftol=1e-9, xtol=1e-9, gtol=1e-7)
        full = expand(fit.x)
        theta[block["parameter_slice"]] = full
        diagnostics.append(dict(order=block["order"], success=bool(fit.success), message=fit.message,
                                nfev=int(fit.nfev), optimality=float(fit.optimality),
                                data_objective=float(np.sum(data(fit.x)**2)),
                                prior_objective=float(np.sum((fit.x/scales[free])**2)),
                                training_indices=indices.tolist(), n_free=int(free.sum()),
                                ip_parameters=full[-3:].tolist(),
                                ip_bound_count=int(np.sum(np.abs(full[-3:]) > IP_BOUND-1e-5)),
                                zero_intensity_count=int(np.sum(1+block["upper"]*full[:n] < 1e-5))))
    if not all(row["success"] for row in diagnostics):
        raise ValueError(f"Nonconverged fit: {diagnostics}")
    return theta, diagnostics


def correction_summary(context, theta):
    z = theta[context["line_parameter_indices"]]
    result = original_correction_summary(context, z)
    delta = z*np.array([record["uncertainty_upper"] for record in context["records"]])
    result["contrast_bins"] = {}
    for name, low, high in [("below_0.1_percent", 0, .001), ("0.1_to_0.5_percent", .001, .005),
                             ("0.5_to_2_percent", .005, .02), ("above_2_percent", .02, np.inf)]:
        mask = (context["line_contrast"] >= low) & (context["line_contrast"] < high)
        result["contrast_bins"][name] = dict(n_lines=int(mask.sum()),
            median_absolute_fraction=float(np.median(abs(delta[mask]))) if mask.any() else None,
            over_50_percent_count=int(np.sum(abs(delta[mask]) > .5)),
            zero_intensity_count=int(np.sum(1+delta[mask] < 1e-5)))
    return result


def main():
    started = time.monotonic()
    (OUT/"quadratic_strength_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    context = build_context()
    cross_validation, cv_arrays = {}, {}
    for config in CASES[:len(CV_NAMES)]:
        folds, total = [], 0.
        for heldout in range(5):
            indices = [i for i in range(5) if i != heldout]
            theta, fit = fit_joint_strength(context, config, train_indices=indices)
            component = predict_components(context, theta, config)
            residual = context["project"](context["y_over_c"][:, :5]-component[:, :5])[:, heldout]
            sse = float(np.sum(residual[context["allowed"]]**2))
            folds.append(dict(heldout_exposure=int(context["exposures"][heldout]), sse=sse,
                              rms_percent=float(100*np.sqrt(sse/context["allowed"].sum())), fit=fit))
            total += sse
            cv_arrays[f"theta_{config['name']}_fold{heldout}"] = theta
            print(f"CV {config['name']} fold {heldout}: {folds[-1]['rms_percent']:.6f}%", flush=True)
        cross_validation[config["name"]] = dict(config=config, folds=folds, sse=total,
            rms_percent=float(100*np.sqrt(total/(5*context["allowed"].sum()))))
    ranking = sorted(cross_validation, key=lambda name: cross_validation[name]["sse"])
    joint_ranking = [name for name in ranking if not name.startswith("fixed_")]
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=ranking[0],
                     selected_joint_case=joint_ranking[0], ranking=ranking, cross_validation=cross_validation,
                     criterion=PLAN["selection"], limitation=PLAN["selection_limitation"])
    # Commit the choice before computing the four-exposure evaluation metrics.
    (OUT/"quadratic_strength_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    np.savez_compressed(OUT/"quadratic_strength_cv.npz", **cv_arrays)
    baseline, baseline_residual = metrics(context, context["reference_component"])
    cases, arrays, rows = {}, {}, []
    for config in CASES:
        name = config["name"]
        theta, fit = fit_joint_strength(context, config)
        component = predict_components(context, theta, config)
        metric, residual = metrics(context, component)
        cases[name] = dict(config=config, fit=fit, metrics=metric, corrections=correction_summary(context, theta))
        arrays.update({f"theta_{name}": theta, f"component_{name}": component, f"residual_{name}": residual})
        widths = np.concatenate([prior_widths(block, config) for block in context["blocks"]])
        for i, (record, z) in enumerate(zip(context["records"], theta[context["line_parameter_indices"]])):
            rows.append(dict(case=name, order=record["order"], species=record["species"], raw_index=record["raw_index"],
                             nu_cm1=record["nu"], reference_strength=record["sw"], uncertainty_upper=record["uncertainty_upper"],
                             baseline_contrast=context["line_contrast"][i], prior_multiplier=widths[i],
                             z=float(z), fractional_correction=float(z*record["uncertainty_upper"])))
        print(f"{name}: train {metric['training']['rms_percent']:.6f}%, eval {metric['evaluation_score']['rms_percent']:.6f}%, "
              f"zeros {cases[name]['corrections']['zero_intensity_count']}", flush=True)
    with np.load(OUT/"relaxed_strength_fit.npz") as previous:
        np.testing.assert_allclose(arrays["component_fixed_uniform"], previous["component_soft_prior10"], atol=2e-7, rtol=0)
    paths = context["source_files"]+[Path(__file__), ROOT/"quadratic_ip_response.py", ROOT/"fit_line_strength.py",
                                    ROOT/"fit_relaxed_strength.py", ROOT/"line_strength_support.py"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN, selection=selection,
                  selected_case=ranking[0], selected_joint_case=joint_ranking[0], baseline=baseline, cases=cases,
                  line_records=context["records"], fixed_ip_reconstruction_error=context["fixed_ip_reconstruction_error"],
                  seconds=time.monotonic()-started,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/"quadratic_strength_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"quadratic_strength_fit.npz", orders=context["orders"], exposures=context["exposures"],
                        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
                        y_over_c=context["y_over_c"], continuum=context["continuum"], component_baseline=context["reference_component"],
                        residual_baseline=baseline_residual, line_parameter_indices=context["line_parameter_indices"],
                        ip_parameter_indices=context["ip_parameter_indices"], line_contrast=context["line_contrast"], **arrays)
    with (OUT/"quadratic_strength_corrections.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for family in FAMILIES:
        names = [f"{family}_contrast{k}" for k in [3, 10]]
        axes[0].plot([3, 10], [cases[n]["metrics"]["evaluation_score"]["rms_percent"] for n in names], "o-", label=family)
    axes[0].axhline(cases["fixed_uniform"]["metrics"]["evaluation_score"]["rms_percent"], ls="--", color="0.5", label="Previous uniform10")
    axes[0].set(xlabel="Maximum contrast-weighted prior multiplier", ylabel="Evaluation RMS (%)", xticks=[3, 10])
    axes[0].legend(fontsize=8)
    for name in ["fixed_uniform", ranking[0]]:
        z = arrays[f"theta_{name}"][context["line_parameter_indices"]]
        upper = np.array([r["uncertainty_upper"] for r in context["records"]])
        axes[1].scatter(100*context["line_contrast"], 100*upper*z, s=12, alpha=.6, label=name)
    axes[1].set(xscale="log", xlabel="Baseline line contrast (%)", ylabel="Line-strength correction (%)")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/"quadratic_strength_fit.png", dpi=160); plt.close(fig)
    print("CV selected:", ranking[0], "best new IP:", joint_ranking[0], "seconds:", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
