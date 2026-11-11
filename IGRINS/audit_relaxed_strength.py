"""Independently audit the relaxed, conditional HITRAN strength experiment."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from audit_line_strength import independent_filter, numpy_component

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def independent_project(context, values):
    """Project each spectrum with QR, independently of the fitted pseudoinverse."""
    return np.array([
        independent_filter(row, wavelength, mask)
        for row, wavelength, mask in zip(
            values, context["wave"], context["allowed"])
    ])


def independent_components(context, theta):
    """Use direct NumPy convolutions to check the nonlinear opacity calculation."""
    return np.array([
        numpy_component(block, theta[block["parameter_slice"]])
        for block in context["blocks"]
    ])


def independent_metrics(context, residual):
    result = {}
    scopes = [
        ("training", slice(0, 5), context["allowed"]),
        ("evaluation_score", slice(5, None), context["score"]),
        ("evaluation_included", slice(5, None), context["allowed"]),
    ]
    for name, exposures, mask in scopes:
        rows = residual[:, exposures]
        keep = np.broadcast_to(mask[:, None], rows.shape)
        fixed = rows * context["continuum"][:, exposures]
        common = np.broadcast_to(rows.mean(axis=1)[:, None], rows.shape)
        varying = rows - common
        rms = float(100 * np.sqrt(np.mean(rows[keep] ** 2)))
        common_rms = float(100 * np.sqrt(np.mean(common[keep] ** 2)))
        varying_rms = float(100 * np.sqrt(np.mean(varying[keep] ** 2)))
        np.testing.assert_allclose(common_rms ** 2 + varying_rms ** 2,
                                   rms ** 2, rtol=1e-12)
        result[name] = dict(
            n_samples=int(keep.sum()), rms_percent=rms,
            fixed_rms_percent=float(100 * np.sqrt(np.mean(fixed[keep] ** 2))),
            common_rms_percent=common_rms,
            varying_rms_percent=varying_rms,
            per_order={str(order): float(100 * np.sqrt(
                np.mean(rows[b][keep[b]] ** 2)))
                for b, order in enumerate(context["orders"])},
        )
    return result


def compare_metrics(actual, expected):
    for scope, values in actual.items():
        assert values["n_samples"] == expected[scope]["n_samples"]
        for key in ["rms_percent", "fixed_rms_percent"]:
            np.testing.assert_allclose(values[key], expected[scope][key],
                                       rtol=0, atol=1e-9)
        for order, rms in values["per_order"].items():
            np.testing.assert_allclose(rms, expected[scope]["per_order"][order],
                                       rtol=0, atol=1e-9)


def validate_parameters(context, theta, cap):
    upper = np.array([row["uncertainty_upper"]
                      for row in context["records"]])
    strength_ratio = 1 + upper * theta
    assert np.all(np.isfinite(theta))
    assert np.min(strength_ratio) >= -1e-12
    if cap is not None:
        assert np.max(np.abs(theta)) <= cap + 1e-10
    minimum_tau = []
    for block in context["blocks"]:
        z = theta[block["parameter_slice"]]
        delta = (block["line_factors"] * (block["upper"] * z)[None])
        total_tau = block["baseline_tau"] + delta @ block["line_tau"]
        minimum_tau.append(float(np.min(total_tau)))
    assert min(minimum_tau) >= -1e-12
    return dict(
        minimum_strength_ratio=float(strength_ratio.min()),
        maximum_strength_ratio=float(strength_ratio.max()),
        maximum_absolute_z=float(np.max(np.abs(theta))),
        count_beyond_nominal_bin_edge=int(np.sum(np.abs(theta) > 1 + 1e-6)),
        minimum_optical_depth=min(minimum_tau),
    )


def verify_old_products():
    """Check that existing fit provenance still matches its recorded audit."""
    previous = json.loads((OUT / "line_strength_independent_audit.json").read_text())
    checked = {}
    for name, expected in previous["source_sha256"].items():
        actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
        assert actual == expected, name
        checked[name] = actual
    return checked


def check_objective(context, residual, theta, scale, fit, indices):
    for b, block in enumerate(context["blocks"]):
        rows = residual[b, indices][:, context["allowed"][b]]
        data_objective = float(np.sum((rows / .0045) ** 2))
        prior_objective = float(np.sum(
            (theta[block["parameter_slice"]] / scale) ** 2))
        assert fit[b]["training_indices"] == list(indices)
        assert fit[b]["success"]
        np.testing.assert_allclose(data_objective, fit[b]["data_objective"],
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(prior_objective, fit[b]["prior_objective"],
                                   rtol=1e-12, atol=1e-12)


def audit_cross_validation(context, selection, saved):
    reports, errors = {}, []
    for name, case in selection["cross_validation"].items():
        pooled_sse = 0.
        folds = []
        assert len(case["folds"]) == 5
        for heldout, fold in enumerate(case["folds"]):
            assert fold["heldout_exposure"] == context["exposures"][heldout]
            theta = saved[f"theta_{name}_fold{heldout}"]
            physical = validate_parameters(context, theta, None)
            component = independent_components(context, theta)
            residual = independent_project(context, context["y_over_c"] - component)
            sse = float(np.sum(residual[:, heldout][context["allowed"]] ** 2))
            errors.append(abs(sse - fold["sse"]))
            np.testing.assert_allclose(sse, fold["sse"], rtol=1e-9, atol=1e-12)
            rms = float(100 * np.sqrt(sse / context["allowed"].sum()))
            np.testing.assert_allclose(rms, fold["rms_percent"], rtol=0, atol=1e-9)
            indices = [index for index in range(5) if index != heldout]
            check_objective(context, residual, theta, case["prior_scale"],
                            fold["fit"], indices)
            pooled_sse += sse
            folds.append(dict(heldout_exposure=fold["heldout_exposure"], sse=sse,
                              rms_percent=rms, physical=physical))
        np.testing.assert_allclose(pooled_sse, case["sse"], rtol=1e-9, atol=1e-12)
        assert case["n_samples"] == 5 * context["allowed"].sum()
        reports[name] = dict(sse=pooled_sse, folds=folds)
    ranking = sorted(reports, key=lambda name: reports[name]["sse"])
    assert ranking == selection["ranking"]
    assert ranking[0] == selection["selected_case"]
    return dict(cases=reports, ranking=ranking, selected_case=ranking[0],
                maximum_fold_sse_difference=max(errors))


def check_refits(context, metadata, saved, cv_arrays):
    from fit_relaxed_strength import fit_relaxed

    name = metadata["selected_case"]
    case = metadata["cases"][name]
    scale = case["config"]["prior_scale"]
    theta = saved[f"theta_{name}"]
    rng = np.random.default_rng(853)
    corrupted = context["y_over_c"].copy()
    corrupted[:, 5:] += 100 * rng.normal(size=corrupted[:, 5:].shape)
    repeated, _ = fit_relaxed(context, target=corrupted, prior_scale=scale)
    eval_difference = float(np.max(np.abs(repeated - theta)))
    np.testing.assert_allclose(repeated, theta, rtol=0, atol=1e-7)

    # The omitted training row must not enter that fold's fit either.
    corrupted[:, 0] += 100 * rng.normal(size=corrupted[:, 0].shape)
    repeated_fold, _ = fit_relaxed(context, target=corrupted, prior_scale=scale,
                                   train_indices=[1, 2, 3, 4])
    cv_difference = float(np.max(np.abs(repeated_fold - cv_arrays[f"theta_{name}_fold0"])))
    np.testing.assert_allclose(repeated_fold, cv_arrays[f"theta_{name}_fold0"],
                               rtol=0, atol=1e-7)

    upper = np.array([record["uncertainty_upper"]
                      for record in context["records"]])
    initial = (np.random.default_rng(369).uniform(-.8, .8, len(upper))
               * np.minimum(scale, 1 / upper))
    restarted, diagnostics = fit_relaxed(context, prior_scale=scale, initial=initial)
    objective = sum(item["data_objective"] + item["prior_objective"]
                    for item in case["fit"])
    restarted_objective = sum(item["data_objective"] + item["prior_objective"]
                              for item in diagnostics)
    np.testing.assert_allclose(restarted_objective, objective, rtol=1e-7, atol=1e-8)
    component = independent_components(context, restarted)
    residual = independent_project(context, context["y_over_c"] - component)
    restarted_metrics = independent_metrics(context, residual)
    rms = restarted_metrics["evaluation_score"]["rms_percent"]
    np.testing.assert_allclose(rms, case["metrics"]["evaluation_score"]["rms_percent"],
                               rtol=0, atol=1e-5)
    for invalid in [[5], [0, 0], []]:
        try:
            fit_relaxed(context, train_indices=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid training indices accepted: {invalid}")
    return dict(
        heldout_target_corruption_parameter_difference=eval_difference,
        omitted_training_row_corruption_parameter_difference=cv_difference,
        random_restart=dict(seed=369,
            initial_rule="uniform(-0.8,0.8)*min(prior_scale,1/u), per line",
            original_objective=objective, restarted_objective=restarted_objective,
            maximum_parameter_difference=float(np.max(np.abs(restarted - theta))),
            evaluation_score_rms_percent=rms, physical=validate_parameters(context, restarted, None)),
    )


def main():
    from fit_line_strength import build_context

    old_products = verify_old_products()
    metadata = json.loads((OUT / "relaxed_strength_fit.json").read_text())
    selection = json.loads((OUT / "relaxed_strength_selection.json").read_text())
    assert selection == metadata["selection"]
    assert selection["selected_case"] == metadata["selected_case"]
    saved = np.load(OUT / "relaxed_strength_fit.npz")
    cv_arrays = np.load(OUT / "relaxed_strength_cv.npz")
    old = np.load(OUT / "line_strength_fit.npz")
    context = build_context()
    assert metadata["line_records"] == context["records"]
    assert len(context["records"]) == 269
    for name, key in [("y_over_c", "y_over_c"), ("continuum", "continuum"),
                      ("included_mask", "allowed"), ("score_mask", "score"),
                      ("wavelength_um", "wave"), ("orders", "orders"),
                      ("exposures", "exposures"),
                      ("component_baseline", "reference_component")]:
        np.testing.assert_array_equal(saved[name], context[key])
        np.testing.assert_array_equal(saved[name], old[name])
    np.testing.assert_allclose(saved["component_cap1_prior1"], old["component_primary"],
                               rtol=0, atol=1e-9)
    cases = {}
    for name in ["baseline"] + list(metadata["cases"]):
        case = None if name == "baseline" else metadata["cases"][name]
        theta = np.zeros(269) if case is None else saved[f"theta_{name}"]
        cap = None if case is None else case["config"]["cap"]
        physical = validate_parameters(context, theta, cap)
        component = independent_components(context, theta)
        component_error = float(np.max(np.abs(component - saved[f"component_{name}"])))
        np.testing.assert_allclose(component, saved[f"component_{name}"], rtol=0, atol=1e-9)
        residual = independent_project(context, context["y_over_c"] - component)
        residual_error = float(np.max(np.abs(residual - saved[f"residual_{name}"])))
        np.testing.assert_allclose(residual, saved[f"residual_{name}"], rtol=0, atol=1e-9)
        scores = independent_metrics(context, residual)
        compare_metrics(scores, metadata["baseline"] if case is None else case["metrics"])
        if case is not None:
            check_objective(context, residual, theta, case["config"]["prior_scale"],
                            case["fit"], list(range(5)))
            assert physical["count_beyond_nominal_bin_edge"] == case["corrections"]["outside_original_count"]
            np.testing.assert_allclose(physical["minimum_strength_ratio"],
                                       case["corrections"]["minimum_factor"], rtol=0, atol=1e-12)
            np.testing.assert_allclose(physical["maximum_strength_ratio"],
                                       case["corrections"]["maximum_factor"], rtol=0, atol=1e-12)
        cases[name] = dict(physical=physical, metrics=scores,
                          maximum_component_difference=component_error,
                          maximum_residual_difference=residual_error)
    print("Independent fit product checks passed", flush=True)
    cv = audit_cross_validation(context, selection, cv_arrays)
    print("Independent conditional CV checks passed", flush=True)
    refits = check_refits(context, metadata, saved, cv_arrays)
    print("Target isolation and random restart checks passed", flush=True)
    paths = [Path(__file__), ROOT / "audit_line_strength.py", ROOT / "fit_relaxed_strength.py",
             OUT / "relaxed_strength_fit.json", OUT / "relaxed_strength_fit.npz",
             OUT / "relaxed_strength_selection.json", OUT / "relaxed_strength_cv.npz"]
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
        old_products_unchanged=old_products, cases=cases, cross_validation=cv, refits=refits,
        limitations=[
            "HITRAN uncertainty categories are neither guaranteed hard bounds nor calibrated Gaussian sigmas.",
            "CV isolates line correction fitting only; original nuisance parameters were fitted using all five training exposures.",
            "Evaluation spectra have their own frozen nuisance fits and fitted polynomial projection; this is conditional evaluation.",
            "Larger effective intensity corrections do not identify the physical source of residuals.",
            "One random restart checks local numerical reproducibility, not global uniqueness or line identifiability.",
        ],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths},
    )
    (OUT / "relaxed_strength_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
