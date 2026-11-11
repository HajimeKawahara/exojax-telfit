"""Test shared background and wavelength corrections with the adopted IP fixed."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import lsq_linear

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_relaxed_strength import residual_statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
CASES = {
    "baseline": [],
    "background_counts": ["counts_offset"],
    "background_leakage0": ["leakage_c0"],
    "background_leakage1": ["leakage_c0", "leakage_c1"],
    "wavelength0": ["wave_c0"],
    "wavelength1": ["wave_c0", "wave_c1"],
    "wavelength2": ["wave_c0", "wave_c1", "wave_c2"],
    "background_leakage0_wavelength2": ["leakage_c0", "wave_c0", "wave_c1", "wave_c2"],
    "background_leakage1_wavelength2": ["leakage_c0", "leakage_c1", "wave_c0", "wave_c1", "wave_c2"],
    "background_counts_wavelength2": ["counts_offset", "wave_c0", "wave_c1", "wave_c2"],
}


def project(values, wave, allowed):
    """Apply the existing Q5 projection along the last axis."""
    result = np.empty_like(values)
    for b, wavelength in enumerate(wave):
        x = 2*(np.log(wavelength)-np.log(wavelength).min())/np.ptp(np.log(wavelength))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, r = np.linalg.qr(basis[allowed[b]], mode="reduced")
        coefficients = np.linalg.solve(r, (values[b][..., allowed[b]] @ q).T).T
        result[b] = values[b]-coefficients @ basis.T
    return result


def build_context():
    """Use only saved predictions and fixed masks; do not refit physical parameters."""
    source = OUT / "adopted_telluric_fit.npz"
    with np.load(source) as saved:
        data = {key: saved[key] for key in saved.files}
    wave, model, continuum = [data[key] for key in ["wavelength_um", "component", "continuum"]]
    records, columns, scales = [], [], []
    for b, order in enumerate(data["orders"]):
        velocity = C_KMS*np.log(wave[b])
        x = 2*(velocity-velocity.min())/np.ptp(velocity)-1
        basis = np.polynomial.chebyshev.chebvander(x, 2)
        sort = np.argsort(velocity)
        derivative = CubicSpline(velocity[sort], model[b][:, sort], axis=1)(velocity, 1)
        # An offset constant in the input-flux units has the same absolute
        # normalization for all epochs of an order. The leakage model includes
        # renormalization: (F+b)/(1+b) = F+f*(1-F), f=b/(1+b).
        scale = float(np.median(continuum[b, :5, data["included_mask"][b]]))
        definitions = [("counts_offset", scale/continuum[b], .03, "fraction of training median continuum"),
                       ("leakage_c0", 1-model[b], .03, "normalized leakage fraction"),
                       ("leakage_c1", basis[:, 1]*(1-model[b]), .03, "normalized leakage slope"),
                       ("wave_c0", derivative*basis[:, 0], .2, "km/s"),
                       ("wave_c1", derivative*basis[:, 1], .2, "km/s"),
                       ("wave_c2", derivative*basis[:, 2], .2, "km/s")]
        for name, local, bound, unit in definitions:
            column = np.zeros_like(model)
            column[b] = local
            columns.append(column)
            scales.append(bound)
            records.append(dict(index=len(records), order=int(order), order_index=b,
                                name=name, bound=bound, unit=unit))
    data["design_raw"] = np.stack(columns, axis=-1)
    data["design_filtered"] = np.stack([project(column, wave, data["included_mask"])
                                         for column in columns], axis=-1)
    data["parameter_records"], data["bounds"] = records, np.asarray(scales)
    np.testing.assert_allclose(project(data["residual_unfiltered"], wave, data["included_mask"]),
                               data["residual_filtered"], atol=1e-11, rtol=0)
    return data


def fit_case(context, case, target=None, training_indices=None):
    """Fit a bounded shared correction to five training epochs or their subset."""
    if target is None:
        target = context["residual_filtered"]
    indices = np.arange(5) if training_indices is None else np.asarray(training_indices, dtype=int)
    if len(indices) != len(np.unique(indices)) or np.any((indices < 0) | (indices >= 5)):
        raise ValueError("Only unique training indices in [0,5) are permitted")
    active = np.array([record["name"] in CASES[case] for record in context["parameter_records"]])
    theta = np.zeros(len(active))
    if not active.any():
        return theta, dict(success=True, n_free=0, active_bounds=0, optimality=0.)
    keep = np.broadcast_to(context["included_mask"][:, None], target[:, indices].shape)
    design = context["design_filtered"][:, indices][keep][:, active]
    limits = context["bounds"][active]
    fit = lsq_linear(design*limits, target[:, indices][keep], bounds=(-1., 1.),
                     method="bvls", tol=1e-12, max_iter=1000)
    theta[active] = fit.x*limits
    return theta, dict(success=bool(fit.success), message=fit.message,
                       n_free=int(active.sum()), active_bounds=int(np.sum(fit.active_mask != 0)),
                       optimality=float(fit.optimality), condition_number=float(np.linalg.cond(design*limits)))


def predict_correction(context, theta, filtered=True):
    """Return the additive correction in normalized flux units."""
    return context["design_filtered" if filtered else "design_raw"] @ theta


def interpolation_check(context, theta):
    """Check the first derivative against shifting the saved sampled model exactly."""
    linear, shifted = np.zeros_like(context["component"]), np.zeros_like(context["component"])
    ranges = []
    for b, order in enumerate(context["orders"]):
        velocity = C_KMS*np.log(context["wavelength_um"][b])
        x = 2*(velocity-velocity.min())/np.ptp(velocity)-1
        basis = np.polynomial.chebyshev.chebvander(x, 2)
        indices = [r["index"] for r in context["parameter_records"]
                   if r["order_index"] == b and r["name"].startswith("wave_")]
        delta = basis @ theta[indices]
        sort = np.argsort(velocity)
        shifted[b] = CubicSpline(velocity[sort], context["component"][b][:, sort], axis=1)(velocity+delta)-context["component"][b]
        linear[b] = context["design_raw"][b][..., indices] @ theta[indices]
        ranges.append(dict(order=int(order), minimum_shift_kms=float(delta.min()), maximum_shift_kms=float(delta.max()),
                           maximum_shift_pixels=float(np.max(abs(delta)/abs(np.gradient(velocity))))))
    difference = project(shifted-linear, context["wavelength_um"], context["included_mask"])
    keep = np.broadcast_to(context["included_mask"][:, None], difference.shape)
    return dict(ranges=ranges, maximum_filtered_flux_difference_percent=float(100*abs(difference[keep]).max()),
                evaluation_score=residual_statistics(difference[:, 5:], context["score_mask"])), difference


def main():
    context = build_context()
    source_paths = [Path(__file__), ROOT/"diagnose_relaxed_strength.py", OUT/"adopted_telluric_fit.npz",
                    OUT/"adopted_telluric_model.json", OUT/"poly5_spectral_budget.json", OUT/"nonstellar_spectral_selection.npz"]
    hashes = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), baseline="fixed_contrast10",
                cases=CASES, parameter_records=context["parameter_records"],
                training_exposures=context["exposures"][:5].tolist(), evaluation_exposures=context["exposures"][5:].tolist(),
                selection="Minimum pooled leave-one-training-exposure-out SSE on fixed included pixels; evaluation is not used for selection",
                bounds_note="Exploratory per-coefficient bounds, not measured prior uncertainties; signed background admits over-subtraction",
                frozen="Stellar model, TP, molecular columns, line strengths, original IP, continuum, Q5 and masks",
                source_sha256=hashes)
    (OUT/"background_wavelength_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    arrays = {key: context[key] for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask",
              "y_over_c", "component", "continuum", "residual_filtered", "design_raw", "design_filtered", "bounds"]}
    cases, cv = {}, {}
    for name in CASES:
        folds, sse = [], 0.
        for held in range(5):
            theta, fit = fit_case(context, name, training_indices=[i for i in range(5) if i != held])
            residual = context["residual_filtered"]-predict_correction(context, theta)
            metric = residual_statistics(residual[:, [held]], context["included_mask"])
            sse += metric["sse"]
            folds.append(dict(exposure=int(context["exposures"][held]), metrics=metric, fit=fit))
            arrays[f"cv_theta_{name}_{held}"] = theta
        cv[name] = dict(sse=sse, rms_percent=float(100*np.sqrt(sse/(5*context["included_mask"].sum()))), folds=folds)
    ranking = sorted(cv, key=lambda name: cv[name]["sse"])
    # Freeze the training-only selection before accessing evaluation summaries.
    selection = dict(selected_case=ranking[0], ranking=ranking, cross_validation=cv)
    (OUT/"background_wavelength_selection.json").write_text(json.dumps(selection, indent=2)+"\n")
    with np.load(OUT/"nonstellar_spectral_selection.npz") as saved:
        category = saved["category"]
    regions = json.loads((OUT/"poly5_spectral_budget.json").read_text())["region_sets"]["post_filter_training_selection"]
    for name in CASES:
        theta, fit = fit_case(context, name)
        correction = predict_correction(context, theta)
        residual = context["residual_filtered"]-correction
        arrays[f"theta_{name}"] = theta
        arrays[f"correction_raw_{name}"] = predict_correction(context, theta, filtered=False)
        arrays[f"correction_filtered_{name}"] = correction
        arrays[f"residual_{name}"] = residual
        check, interpolation_error = interpolation_check(context, theta)
        arrays[f"interpolation_error_{name}"] = interpolation_error
        metrics = dict(training=residual_statistics(residual[:, :5], context["included_mask"]),
                       evaluation_score=residual_statistics(residual[:, 5:], context["score_mask"]),
                       evaluation_included=residual_statistics(residual[:, 5:], context["included_mask"]))
        metrics["evaluation_per_order"] = {
            str(int(order)): residual_statistics(residual[b:b+1, 5:], context["score_mask"][b:b+1])
            for b, order in enumerate(context["orders"])}
        metrics["categories"] = {
            label: residual_statistics(residual[:, 5:], context["score_mask"]&(category == code))
            for code, label in enumerate(["H2O-dominant", "CO2-dominant", "Mixed", "Weak absorption"])}
        local = []
        for region in regions:
            b, (start, stop) = region["order_index"], region["array_interval"]
            mask = np.zeros_like(context["score_mask"])
            mask[b, start:stop] = context["score_mask"][b, start:stop]
            local.append(dict(order=region["order"], peak_wavelength_um=region["peak_wavelength_um"],
                              metrics=residual_statistics(residual[:, 5:], mask)))
        cases[name] = dict(fit=fit, theta=theta.tolist(), metrics=metrics, regions=local,
                           interpolation_check=check,
                           exact_saved_model_shift_score=residual_statistics(residual[:, 5:]-interpolation_error[:, 5:], context["score_mask"]))
        print(name, "CV", cv[name]["rms_percent"], "evaluation", metrics["evaluation_score"]["rms_percent"], flush=True)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=plan, selection=selection, cases=cases,
        caveats=["Conditional on nuisance fits and the five-epoch adopted line strengths; not end-to-end independent cross-validation.",
                 "No evaluation exposure determines the new shared coefficients or case selection; Q5 is the existing per-exposure linear projection.",
                 "Background leakage can mimic column/strength changes; a fitted coefficient is not a unique background measurement.",
                 "A constant offset refers to the public cube's flux units, not original detector counts, which are unavailable.",
                 "Wavelength derivatives and finite-shift checks use the saved observed-resolution model, not unsampled detector spectra.",
                 "No detector position map, interpolation phase/kernel, sky spectra, pixel uncertainties or individual nods are available.",
                 "Smooth wavelength correction tests calibration/resampling displacement only; they do not test all extraction artifacts.",
                 "Bounds are per coefficient; the total linear leakage slope can reach 6% and quadratic shift 0.6 km/s.",
                 "No arbitrary narrow residual template is fitted; the adopted model has not changed."],
        injection_interface="Import build_context, fit_case and predict_correction; add the identically Q5-filtered injected spectrum to context['residual_filtered'] as target; only training indices are used. Saved design_filtered and bounds are sufficient to reproduce fits.",
        source_sha256=hashes)
    np.savez_compressed(OUT/"background_wavelength_fit.npz", **arrays)
    (OUT/"background_wavelength_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    selected = ranking[0]
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    for b, ax in enumerate(axes):
        for name, color in [("baseline", ".5"), (selected, "C0")]:
            mean = np.where(context["score_mask"][b], arrays[f"residual_{name}"][b, 5:].mean(axis=0), np.nan)
            ax.plot(context["wavelength_um"][b], 100*mean, lw=.8, label=name, color=color)
        ax.set(xlabel="Wavelength (micron)", ylabel="Mean Q5 residual (%)", title=f"Order {context['orders'][b]}")
        ax.legend(fontsize=8)
    fig.savefig(OUT/"background_wavelength_fit.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
