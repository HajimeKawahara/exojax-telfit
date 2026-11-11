"""Profile adopted line strengths jointly with the frozen background/wavelength case."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from fit_background_wavelength import CASES, build_context as background_context
from fit_line_strength import WORKING_SIGMA, build_context, metrics, predict_components

ROOT = Path(__file__).resolve().parent
OUT = ROOT/"results"


def main():
    background = background_context()
    background_meta = json.loads((OUT/"background_wavelength_fit.json").read_text())
    selected = background_meta["selection"]["selected_case"]
    active = np.array([r["name"] in CASES[selected] for r in background["parameter_records"]])
    context = build_context()
    z = background["parameters"][background["line_parameter_indices"]]
    component = predict_components(context, z)
    np.testing.assert_allclose(component, background["component"], atol=1e-10, rtol=0)
    np.testing.assert_array_equal(context["allowed"], background["included_mask"])
    baseline_metrics, residual = metrics(context, component)
    mask = context["allowed"]
    keep = np.broadcast_to(mask[:, None], residual[:, :5].shape)
    target = residual[:, :5][keep]
    strength_matrix = np.zeros((len(target), len(z)))
    start = 0
    for b, block in enumerate(context["blocks"]):
        n = 5*int(mask[b].sum())
        derivative = -WORKING_SIGMA*np.asarray(block["data_jacobian"](
            z[block["parameter_slice"]], jnp.asarray(context["y_over_c"][b, :5])))
        strength_matrix[start:start+n, block["parameter_slice"]] = derivative
        start += n
    extra_matrix = background["design_filtered"][:, :5][keep][:, active]
    contrast = background["line_contrast"]
    widths = 1+9*contrast**2/(contrast**2+.005**2)
    uncertainty = np.array([record["uncertainty_upper"] for record in context["records"]])
    lower_strength = -1/uncertainty-z
    cases, arrays = {}, dict(wavelength_um=context["wave"], included_mask=mask, score_mask=context["score"],
                             exposures=context["exposures"], orders=context["orders"],
                             component_baseline=component, residual_baseline=residual,
                             strength_z_baseline=z, strength_training_design=strength_matrix,
                             background_training_design=extra_matrix, strength_prior_widths=widths,
                             active_background_parameter_indices=np.flatnonzero(active),
                             target_training_filtered=target, upper_uncertainty_fraction=uncertainty)
    for name, use_background in [("local_strength_only_control", False), ("local_joint_background_wavelength_strength", True)]:
        n_extra = int(active.sum()) if use_background else 0
        matrix = np.column_stack([strength_matrix, extra_matrix]) if use_background else strength_matrix
        penalties = np.zeros((len(z), len(z)+n_extra))
        penalties[:, :len(z)] = np.diag(1/widths)
        design = np.vstack([matrix/WORKING_SIGMA, penalties])
        data = np.r_[target/WORKING_SIGMA, -z/widths]
        lower = np.r_[lower_strength, -background["bounds"][active]] if use_background else lower_strength
        upper = np.r_[np.full(len(z), np.inf), background["bounds"][active]] if use_background else np.full(len(z), np.inf)
        fitted = lsq_linear(design, data, bounds=(lower, upper), method="bvls", tol=1e-11, max_iter=1000)
        if not fitted.success:
            raise RuntimeError(fitted.message)
        final_z = z+fitted.x[:len(z)]
        theta = np.zeros(len(active))
        if use_background:
            theta[active] = fitted.x[len(z):]
        strength_component = predict_components(context, final_z)
        raw_correction = background["design_raw"] @ theta
        final_component = strength_component+raw_correction
        statistic, final_residual = metrics(context, final_component)
        exact_training = context["project"](final_component-component)[:, :5][keep]
        approximation_error = exact_training-matrix @ fitted.x
        gradient = design.T @ (design @ fitted.x-data)
        lower_active = fitted.x <= lower+1e-8
        upper_active = fitted.x >= upper-1e-8
        interior = ~lower_active&~upper_active
        kkt_violation = max(float(np.max(abs(gradient[interior]), initial=0)),
                            float(np.max(-gradient[lower_active], initial=0)),
                            float(np.max(gradient[upper_active], initial=0)))
        cases[name] = dict(metrics=statistic, fit_success=bool(fitted.success), message=fitted.message,
                           optimality=float(fitted.optimality), kkt_violation=kkt_violation,
                           iterations=int(fitted.nit), strength_parameter_count=len(z), additional_parameter_count=n_extra,
                           background_theta=theta.tolist(), strength_z_change_max=float(np.max(abs(final_z-z))),
                           strength_fractional_change_max=float(np.max(abs(uncertainty*(final_z-z)))),
                           minimum_final_intensity_factor=float(np.min(1+uncertainty*final_z)),
                           additional_bound_count=int(np.sum(abs(theta[active]) >= background["bounds"][active]-1e-8)) if use_background else 0,
                           linearization_training_rms_error_percent=float(100*np.sqrt(np.mean(approximation_error**2))),
                           linearization_training_max_error_percent=float(100*np.max(abs(approximation_error))))
        arrays[f"strength_z_{name}"] = final_z
        arrays[f"background_theta_{name}"] = theta
        arrays[f"component_{name}"] = final_component
        arrays[f"residual_{name}"] = final_residual
        arrays[f"raw_background_correction_{name}"] = raw_correction
        arrays[f"strength_component_{name}"] = strength_component
        print(name, json.dumps(cases[name]), flush=True)
    sources = [Path(__file__), ROOT/"fit_background_wavelength.py", ROOT/"fit_line_strength.py",
               ROOT/"line_strength_support.py", OUT/"background_wavelength_fit.json", OUT/"background_wavelength_fit.npz",
               OUT/"adopted_telluric_fit.npz", OUT/"adopted_telluric_model.json"]+context["source_files"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_background_case=selected,
                  selection="Previously frozen background-only training-CV choice; no reselection or evaluation tuning",
                  baseline_metrics=baseline_metrics, cases=cases,
                  model="One local linearized joint fit; evaluate final line strengths through exact existing opacity/response forward model and add the frozen background/wavelength design",
                  penalty="Same adopted contrast10 strength prior and nonnegative intensities; additional coefficients retain the previous bounds and no new penalty",
                  limitations=["One local profiling step is not a converged nonlinear joint fit.",
                               "Background/wavelength derivatives remain frozen at the adopted model.",
                               "No model choice uses evaluation exposures; original nuisance fits remain conditional.",
                               "IP, TP, columns, star, original wavelength solution and masks are fixed.",
                               "Effective background and line-strength changes are not unique physical error measurements.",
                               "The preceding background planet test does not cover this joint strength profiling case."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    np.savez_compressed(OUT/"background_wavelength_strength_profile.npz", **arrays)
    (OUT/"background_wavelength_strength_profile.json").write_text(json.dumps(result, indent=2)+"\n")
    report = ["Local joint strength and background/wavelength profile", "=====================================================", "",
              "Frozen added case: "+selected, "",
              f"Adopted baseline evaluation RMS: {baseline_metrics['evaluation_score']['rms_percent']:.9f}%",
              "All coefficients use five training exposures; four evaluation exposures are predictions.", ""]
    for name, case in cases.items():
        report += [name, f"  exact evaluation RMS: {case['metrics']['evaluation_score']['rms_percent']:.9f}%",
                   f"  exact training RMS: {case['metrics']['training']['rms_percent']:.9f}%",
                   f"  local-model training RMS error: {case['linearization_training_rms_error_percent']:.9f}%"]
    report += ["", result["model"], result["penalty"], ""]+result["limitations"]+[""]
    (OUT/"BACKGROUND_WAVELENGTH_STRENGTH_PROFILE_REPORT.txt").write_text("\n".join(report))


if __name__ == "__main__":
    main()
