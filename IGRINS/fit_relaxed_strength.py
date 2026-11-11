"""Test wider line-intensity uncertainty and replace arbitrary caps by priors.

The original constrained fit and data remain intact. Soft cases require only
nonnegative intensities; a training-only conditional CV selects their width.
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

import jax.numpy as jnp
import matplotlib
import numpy as np
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fit_line_strength import build_context, predict_components, metrics, WORKING_SIGMA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
PRIOR_SCALES = [1., 2., 3., 5., 10.]
CASES = ([dict(name=f"cap{cap}_prior1", prior_scale=1., cap=float(cap)) for cap in [1, 2, 3]]
         + [dict(name=f"soft_prior{scale:g}", prior_scale=scale, cap=None) for scale in PRIOR_SCALES])
PLAN = dict(
    cases=CASES, parameterization="S_i=S_i,HITRAN*(1+u_i*z_i)",
    soft_bounds="z_i >= -1/u_i (nonnegative intensity), no upper bound",
    capped_bounds="intersection of [-cap,cap] and z_i>=-1/u_i",
    prior="sum((z_i/prior_scale)^2); u_i is the finite HITRAN bin upper edge, not a known Gaussian sigma",
    primary_selection="Minimum pooled leave-one-training-exposure-out error across five training exposures, soft cases only",
    selection_scope="Conditional on original fitted atmosphere/instrument/star/continuum; these nuisance parameters are not refitted in folds",
    training_exposures=[0, 19, 39, 59, 78], evaluation_exposures=[10, 30, 50, 70],
    fixed="Same 269 selected lines, opacity basis, physical nuisance parameters and degree-five polynomial operator",
    previous_test_note="The preceding half/double-prior test kept z in [-1,1]; it did not allow larger intensity corrections",
    caveat="A larger inferred correction can absorb other model deficiencies; it is not a measured HITRAN database error",
)


def fit_relaxed(context, target=None, prior_scale=1., cap=None, initial=None, train_indices=None):
    """Fit selected training exposures with soft shrinkage and physical positivity."""
    target = context["y_over_c"] if target is None else np.asarray(target)
    initial = np.zeros(context["n_parameters"]) if initial is None else np.asarray(initial)
    selected = np.arange(5) if train_indices is None else np.asarray(train_indices, dtype=int)
    if not len(selected) or len(np.unique(selected)) != len(selected) or np.any((selected < 0) | (selected >= 5)):
        raise ValueError("Use unique indices within the five training exposures")
    if prior_scale <= 0 or (cap is not None and cap <= 0):
        raise ValueError("Prior scale and any explicit cap must be positive")
    theta, diagnostics = initial.copy(), []
    for b, block in enumerate(context["blocks"]):
        sl = block["parameter_slice"]
        upper_fraction = block["upper"]
        lower = -1./upper_fraction
        upper = np.full_like(lower, np.inf)
        if cap is not None:
            lower = np.maximum(lower, -cap)
            upper = np.minimum(upper, cap)
        z0 = np.clip(initial[sl], lower+1e-10, upper-1e-10)
        n, npixels = len(z0), int(context["allowed"][b].sum())
        train_target = jnp.asarray(target[b, :5])

        def data(z):
            return np.asarray(block["data_residual"](z, train_target)).reshape(5, npixels)[selected].ravel()

        def fun(z):
            return np.concatenate([data(z), z/prior_scale])

        def jac(z):
            derivative = np.asarray(block["data_jacobian"](z, train_target)).reshape(5, npixels, n)
            return np.vstack([derivative[selected].reshape(-1, n), np.eye(n)/prior_scale])

        fit = least_squares(fun, z0, jac=jac, bounds=(lower, upper), method="trf", x_scale=1.,
                            max_nfev=200, ftol=1e-9, xtol=1e-9, gtol=1e-7)
        theta[sl] = fit.x
        factors = 1+upper_fraction*fit.x
        if np.min(factors) < -1e-12:
            raise ValueError("Negative fitted line intensity")
        diagnostics.append(dict(order=block["order"], success=bool(fit.success), message=fit.message,
                                nfev=int(fit.nfev), optimality=float(fit.optimality),
                                data_objective=float(np.sum(data(fit.x)**2)),
                                prior_objective=float(np.sum((fit.x/prior_scale)**2)),
                                training_indices=selected.tolist(),
                                outside_original_cap_count=int(np.sum(np.abs(fit.x)>1+1e-6)),
                                explicit_cap_count=int(np.sum(np.abs(fit.x)>cap-1e-3)) if cap is not None else 0,
                                zero_intensity_count=int(np.sum(factors<1e-5)),
                                minimum_intensity_factor=float(factors.min())))
    if not all(item["success"] for item in diagnostics):
        raise ValueError(f"Optimization did not converge: {diagnostics}")
    return theta, diagnostics


def correction_summary(context, theta):
    upper = np.array([r["uncertainty_upper"] for r in context["records"]])
    delta = upper*theta
    result = dict(outside_original_count=int(np.sum(np.abs(theta)>1+1e-6)),
                  zero_intensity_count=int(np.sum(1+delta<1e-5)),
                  median_absolute_fraction=float(np.median(abs(delta))),
                  p95_absolute_fraction=float(np.percentile(abs(delta), 95)),
                  minimum_factor=float(np.min(1+delta)), maximum_factor=float(np.max(1+delta)),
                  maximum_absolute_z=float(np.max(np.abs(theta))), species={})
    for species in ["H2O", "CO2"]:
        keep = np.array([r["species"] == species for r in context["records"]])
        strength = np.array([r["sw"] for r in context["records"]])[keep]
        result["species"][species] = dict(n_lines=int(keep.sum()),
            mean_fraction=float(np.mean(delta[keep])),
            strength_weighted_mean_fraction=float(np.average(delta[keep], weights=strength)),
            median_absolute_fraction=float(np.median(abs(delta[keep]))),
            p95_absolute_fraction=float(np.percentile(abs(delta[keep]), 95)))
    return result


def main():
    started = time.monotonic()
    (OUT / "relaxed_strength_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    context = build_context()
    cross_validation, cv_arrays = {}, {}
    mask = context["allowed"]
    # Select a prior width using the five training spectra only.
    for scale in PRIOR_SCALES:
        name = f"soft_prior{scale:g}"
        folds, total_sse, count = [], 0., 0
        for heldout in range(5):
            selected = [index for index in range(5) if index != heldout]
            theta, fit = fit_relaxed(context, prior_scale=scale, train_indices=selected)
            component = predict_components(context, theta)
            residual = context["project"](context["y_over_c"][:, :5]-component[:, :5])[:, heldout]
            sse = float(np.sum(residual[mask]**2))
            total_sse += sse; count += int(mask.sum())
            folds.append(dict(heldout_exposure=int(context["exposures"][heldout]),
                              rms_percent=float(100*np.sqrt(sse/mask.sum())), sse=sse, fit=fit))
            cv_arrays[f"theta_{name}_fold{heldout}"] = theta
        cross_validation[name] = dict(prior_scale=scale, folds=folds, sse=total_sse,
                                     n_samples=count, rms_percent=float(100*np.sqrt(total_sse/count)))
        print(f"CV {name}: {cross_validation[name]['rms_percent']:.6f}%", flush=True)
    ranking = sorted(cross_validation, key=lambda key: cross_validation[key]["sse"])
    selected_name = ranking[0]
    selection = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=selected_name,
                     ranking=ranking, cross_validation=cross_validation,
                     criterion=PLAN["primary_selection"], limitation=PLAN["selection_scope"],
                     at_largest_tested_prior_width=selected_name == f"soft_prior{max(PRIOR_SCALES):g}")
    # Freeze this choice before calculating or printing evaluation metrics.
    (OUT / "relaxed_strength_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    np.savez_compressed(OUT / "relaxed_strength_cv.npz", **cv_arrays)
    baseline, baseline_residual = metrics(context, context["reference_component"])
    cases, arrays, rows = {}, {}, []
    for config in CASES:
        name = config["name"]
        theta, fit = fit_relaxed(context, prior_scale=config["prior_scale"], cap=config["cap"])
        component = predict_components(context, theta)
        case_metrics, residual = metrics(context, component)
        cases[name] = dict(config=config, fit=fit, metrics=case_metrics,
                           corrections=correction_summary(context, theta))
        arrays[f"theta_{name}"] = theta
        arrays[f"component_{name}"] = component
        arrays[f"residual_{name}"] = residual
        for index, (record, z) in enumerate(zip(context["records"], theta)):
            rows.append(dict(case=name, index=index, order=record["order"], species=record["species"],
                             raw_index=record["raw_index"], nu_cm1=record["nu"],
                             uncertainty_upper=record["uncertainty_upper"], reference_strength=record["sw"],
                             z=float(z), fractional_correction=float(record["uncertainty_upper"]*z)))
        print(f"{name}: train {case_metrics['training']['rms_percent']:.6f}%, "
              f"evaluation {case_metrics['evaluation_score']['rms_percent']:.6f}%, "
              f"outside original {cases[name]['corrections']['outside_original_count']}", flush=True)
    with np.load(OUT / "line_strength_fit.npz") as old:
        np.testing.assert_allclose(arrays["component_cap1_prior1"], old["component_primary"], atol=1e-9, rtol=0)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN,
                  selected_case=selected_name, selection=selection, baseline=baseline, cases=cases,
                  line_records=context["records"], baseline_reconstruction_error=context["baseline_reconstruction_error"],
                  seconds=time.monotonic()-started,
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in context["source_files"]+[Path(__file__), ROOT/"fit_line_strength.py",
                                     ROOT/"line_strength_support.py", ROOT/"cache/H2O.par", ROOT/"cache/CO2.par"]})
    (OUT / "relaxed_strength_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "relaxed_strength_fit.npz", orders=context["orders"], exposures=context["exposures"],
                        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
                        y_over_c=context["y_over_c"], continuum=context["continuum"],
                        component_baseline=context["reference_component"], residual_baseline=baseline_residual, **arrays)
    with (OUT / "relaxed_strength_corrections.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for scope, label in [("training", "Training"), ("evaluation_score", "Evaluation score")]:
        axes[0].plot(PRIOR_SCALES, [cases[f"soft_prior{s:g}"]["metrics"][scope]["rms_percent"] for s in PRIOR_SCALES], "o-", label=label)
    axes[0].plot(PRIOR_SCALES, [cross_validation[f"soft_prior{s:g}"]["rms_percent"] for s in PRIOR_SCALES], "o-", label="Conditional training CV")
    axes[0].axhline(cases["cap1_prior1"]["metrics"]["evaluation_score"]["rms_percent"], ls="--", color="0.5", label="Previous capped evaluation")
    axes[0].set(xscale="log", xlabel="Prior width / HITRAN bin upper edge", ylabel="Continuum-relative RMS (%)")
    axes[0].legend(fontsize=8)
    upper = np.array([r["uncertainty_upper"] for r in context["records"]])
    maximum_correction = np.max(100*upper*arrays[f"theta_{selected_name}"])
    histogram_bins = np.linspace(-100, max(150, 50*np.ceil(maximum_correction/50)), 51)
    for name, label in [("cap1_prior1", "Previous cap"), (selected_name, "Selected soft prior")]:
        axes[1].hist(100*upper*arrays[f"theta_{name}"], bins=histogram_bins, alpha=.5, label=label)
    axes[1].set(xlabel="Fitted line-intensity correction (%)", ylabel="Number of lines")
    axes[1].legend(fontsize=8)
    fig.savefig(OUT / "relaxed_strength_fit.png", dpi=160); plt.close(fig)
    print("Training-CV selected:", selected_name, "elapsed seconds:", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
