"""Independently audit variable-IP fits and weak-line regularization."""
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

from audit_relaxed_strength import independent_metrics, independent_project

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def source_broaden(values, widths, velocity, wing_fraction):
    """Transport each input sample with a normalized local Gaussian kernel."""
    values = np.asarray(values)
    widths = np.broadcast_to(widths, values.shape)
    velocity = np.asarray(velocity)
    assert np.all(widths > 0)
    offsets = np.arange(len(velocity)) - len(velocity) // 2
    profiles = []
    for factor in [1., 2.]:
        profile = np.exp(-.5 * (velocity[:, None] / (factor * widths[None])) ** 2)
        profile /= profile.sum(axis=0)[None]
        profiles.append(profile)
    kernels = (1 - wing_fraction) * profiles[0] + wing_fraction * profiles[1]
    np.testing.assert_allclose(kernels.sum(axis=0), 1, rtol=0, atol=2e-15)
    result = np.zeros_like(values)
    # A source outside the native grid is zero, matching the original padding.
    for offset, kernel in zip(offsets, kernels):
        if offset < 0:
            result[:offset] += values[-offset:] * kernel[-offset:]
        elif offset > 0:
            result[offset:] += values[:-offset] * kernel[:-offset]
        else:
            result += values * kernel
    return result


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_geometry(block):
    """Rebuild detector geometry using explicit finite differences."""
    target = np.asarray(block["target"])
    grid = np.asarray(block["grid"])
    count = len(target)
    coordinate = np.linspace(-1, 1, count)
    logarithmic_velocity = C_KMS * np.log(target)
    pixel_velocity = np.empty(count)
    pixel_velocity[1:-1] = (logarithmic_velocity[2:] - logarithmic_velocity[:-2]) / 2
    pixel_velocity[0] = (-3 * logarithmic_velocity[0]
                         + 4 * logarithmic_velocity[1] - logarithmic_velocity[2]) / 2
    pixel_velocity[-1] = (3 * logarithmic_velocity[-1]
                          - 4 * logarithmic_velocity[-2] + logarithmic_velocity[-3]) / 2
    midpoint_velocity = float(np.interp((count - 1) / 2, np.arange(count), pixel_velocity))
    assert np.min(pixel_velocity) > 0
    x, ratio = [], []

    def interpolate_extrapolate(abscissa, ordinate):
        result = np.interp(grid, abscissa, ordinate)
        left, right = grid < abscissa[0], grid > abscissa[-1]
        result[left] = (ordinate[0] + (grid[left] - abscissa[0])
                        * (ordinate[1] - ordinate[0]) / (abscissa[1] - abscissa[0]))
        result[right] = (ordinate[-1] + (grid[right] - abscissa[-1])
                         * (ordinate[-1] - ordinate[-2]) / (abscissa[-1] - abscissa[-2]))
        return result

    for rv in block["offsets"]:
        shifted_target = target * (1 + rv / C_KMS)
        x.append(interpolate_extrapolate(shifted_target, coordinate))
        ratio.append(interpolate_extrapolate(shifted_target, pixel_velocity / midpoint_velocity))
    return np.array(x), np.array(ratio), pixel_velocity, midpoint_velocity


def independent_widths(block, q, family):
    """Compute the physical positive Gaussian widths at each source location."""
    q = np.asarray(q)
    if family == "fixed":
        np.testing.assert_array_equal(q, 0)
        return np.broadcast_to(np.asarray(block["widths"])[:, None],
                               np.asarray(block["baseline_tau"]).shape)
    assert np.all(np.abs(q) <= np.array([np.log(1.5), .3, .3]) + 1e-10)
    x, ratio, _, _ = independent_geometry(block)
    widths = (np.asarray(block["widths"])[:, None] * np.exp(q[0])
              * (1 + q[1] * x + q[2] * x ** 2) * ratio)
    assert np.min(widths) > 0
    return widths


def numpy_components(context, theta, config, count=9):
    """Evaluate fitted opacity and nonstationary response without JAX."""
    result = []
    family = config["ip_family"]
    for block in context["blocks"]:
        parameters = theta[block["parameter_slice"]]
        nline = len(block["upper"])
        z, q = parameters[:nline], parameters[nline:]
        assert len(q) == 3
        delta_tau = (np.asarray(block["line_factors"])
                     * (block["upper"] * z)[None]) @ block["line_tau"]
        total_tau = np.asarray(block["baseline_tau"]) + delta_tau
        assert np.min(total_tau) >= -1e-12
        intrinsic = (np.asarray(block["shifted_star"])
                     * np.exp(-block["airmass"][:, None] * total_tau))
        fraction = 0. if family == "quadratic_gaussian" else block["wing_fraction"]
        widths = independent_widths(block, q, family)
        spectra = []
        for row, sigma, rv in zip(intrinsic[:count], widths[:count], block["offsets"][:count]):
            broadened = source_broaden(row, sigma, block["velocity"], fraction)
            spectra.append(np.interp(np.asarray(block["target"]) * (1 + rv / C_KMS),
                                     np.asarray(block["grid"]), broadened))
        result.append(spectra)
    return np.array(result)


def independent_line_contrast(context):
    """Predict each line's pre-filter fractional-strength derivative contrast."""
    contrasts = []
    for b, block in enumerate(context["blocks"]):
        intrinsic = np.asarray(block["shifted_star"][:5]) * np.exp(
            -block["airmass"][:5, None] * block["baseline_tau"][:5])
        order_contrasts = np.zeros(len(block["upper"]))
        for e in range(5):
            kernels = []
            for multiplier in [1., 2.]:
                kernel = np.exp(-.5 * (np.asarray(block["velocity"])
                                      / (multiplier * block["widths"][e])) ** 2)
                kernel /= kernel.sum()
                kernels.append(kernel)
            kernel = ((1 - block["wing_fraction"]) * kernels[0]
                      + block["wing_fraction"] * kernels[1])
            for i, line_tau in enumerate(block["line_tau"]):
                derivative = (-block["airmass"][e] * intrinsic[e]
                              * block["line_factors"][e, i] * line_tau)
                derivative = np.convolve(derivative, kernel, mode="same")
                observed_derivative = np.interp(
                    np.asarray(block["target"]) * (1 + block["offsets"][e] / C_KMS),
                    np.asarray(block["grid"]), derivative)
                order_contrasts[i] = max(order_contrasts[i],
                    np.max(np.abs(observed_derivative[context["allowed"][b]])))
        contrasts.extend(order_contrasts)
    return np.array(contrasts)


def validate_parameters(context, theta, config):
    """Check every physical bound, including parameters held fixed in controls."""
    factors, minima, maxima, tau_minima, ip_boundary = [], [], [], [], 0
    per_order = []
    for block in context["blocks"]:
        parameters = theta[block["parameter_slice"]]
        n = len(block["upper"])
        z, q = parameters[:n], parameters[n:]
        assert np.all(np.isfinite(parameters))
        if config["strength_prior"] == "fixed":
            np.testing.assert_array_equal(z, 0)
        strengths = 1 + block["upper"] * z
        assert strengths.min() >= -1e-12
        factors.extend(strengths)
        tau = (block["baseline_tau"]
               + (block["line_factors"] * (block["upper"] * z)[None]) @ block["line_tau"])
        assert tau.min() >= -1e-12
        tau_minima.append(float(tau.min()))
        widths = independent_widths(block, q, config["ip_family"])
        minima.append(float(widths.min()))
        maxima.append(float(widths.max()))
        ip_boundary += int(np.sum(np.abs(q) >= np.array([np.log(1.5), .3, .3]) - 1e-5))
        x, ratio, pixel_velocity, central_velocity = independent_geometry(block)
        curve_x = np.linspace(-1, 1, 11)
        curve_pixel_velocity = np.interp(curve_x, np.linspace(-1, 1, len(pixel_velocity)), pixel_velocity)
        if config["ip_family"] == "fixed":
            curve_sigma = np.full(11, block["widths"][0])
        else:
            curve_sigma = (block["widths"][0] * np.exp(q[0])
                           * (1 + q[1] * curve_x + q[2] * curve_x ** 2)
                           * curve_pixel_velocity / central_velocity)
        model_widths = np.asarray(block["models"][config["ip_family"]]["widths"](q))
        # Equivalent finite-difference formulas round differently before the
        # endpoint slopes are extrapolated across the padding.
        np.testing.assert_allclose(widths, model_widths, rtol=5e-8, atol=2e-9)
        per_order.append(dict(order=int(block["order"]), parameters=q.tolist(),
            at_bounds=(np.abs(q) >= np.array([np.log(1.5), .3, .3]) - 1e-5).tolist(),
            minimum_native_sigma_kms=float(widths.min()), maximum_native_sigma_kms=float(widths.max()),
            independent_geometry_width_max_difference_kms=float(np.max(np.abs(widths-model_widths))),
            curve_reference_exposure=int(context["exposures"][0]), curve_coordinate=curve_x.tolist(),
            curve_core_fwhm_kms=(2 * np.sqrt(2 * np.log(2)) * curve_sigma).tolist(),
            curve_core_fwhm_pixels=(2 * np.sqrt(2 * np.log(2)) * curve_sigma / curve_pixel_velocity).tolist(),
            note="Core Gaussian FWHM; mixture total FWHM is different. Pixel coordinate is the aligned cube proxy."))
    factors = np.array(factors)
    return dict(minimum_intensity_factor=float(factors.min()),
                maximum_intensity_factor=float(factors.max()),
                zero_intensity_count=int(np.sum(factors < 1e-5)),
                minimum_optical_depth=min(tau_minima),
                minimum_native_sigma_kms=min(minima),
                maximum_native_sigma_kms=max(maxima),
                ip_boundary_count=ip_boundary, per_order=per_order)


def check_objective(context, theta, config, residual, diagnostics, indices):
    """Recalculate data and all active prior terms using independent residuals."""
    for b, (block, fit) in enumerate(zip(context["blocks"], diagnostics)):
        parameters = theta[block["parameter_slice"]]
        n = len(block["upper"])
        k, d = config["prior_scale"], block["line_contrast"]
        scales = (1 + (k - 1) * d ** 2 / (d ** 2 + .005 ** 2)
                  if config["strength_prior"] == "contrast" else np.full(n, k))
        prior = np.sum((parameters[:n] / scales) ** 2) + np.sum((parameters[n:] / .15) ** 2)
        data = np.sum((residual[b, indices][:, context["allowed"][b]] / .0045) ** 2)
        assert fit["success"]
        assert fit["training_indices"] == list(indices)
        expected_free = (n if config["strength_prior"] != "fixed" else 0)
        expected_free += 3 if config["ip_family"] != "fixed" else 0
        assert fit["n_free"] == expected_free
        np.testing.assert_allclose(prior, fit["prior_objective"], rtol=1e-10, atol=1e-8)
        np.testing.assert_allclose(data, fit["data_objective"], rtol=1e-9, atol=1e-7)


def check_metrics(actual, expected):
    for name, scope in actual.items():
        assert scope["n_samples"] == expected[name]["n_samples"]
        for key in ["rms_percent", "fixed_rms_percent"]:
            np.testing.assert_allclose(scope[key], expected[name][key], rtol=0, atol=1e-8)
        for order, rms in scope["per_order"].items():
            np.testing.assert_allclose(rms, expected[name]["per_order"][order], rtol=0, atol=1e-8)


def audit_cv(context, selection, saved):
    errors, results = [], {}
    for name, case in selection["cross_validation"].items():
        total, folds = 0., []
        for heldout, fold in enumerate(case["folds"]):
            assert fold["heldout_exposure"] == context["exposures"][heldout]
            theta = saved[f"theta_{name}_fold{heldout}"]
            physical = validate_parameters(context, theta, case["config"])
            component = numpy_components(context, theta, case["config"], count=5)
            residual = independent_project(context, context["y_over_c"][:, :5] - component)
            sse = float(np.sum(residual[:, heldout][context["allowed"]] ** 2))
            errors.append(abs(sse - fold["sse"]))
            np.testing.assert_allclose(sse, fold["sse"], rtol=1e-9, atol=1e-10)
            rms = float(100 * np.sqrt(sse / context["allowed"].sum()))
            np.testing.assert_allclose(rms, fold["rms_percent"], rtol=0, atol=1e-8)
            indices = [i for i in range(5) if i != heldout]
            check_objective(context, theta, case["config"], residual, fold["fit"], indices)
            total += sse
            folds.append(dict(heldout_exposure=fold["heldout_exposure"], sse=sse, physical=physical))
        np.testing.assert_allclose(total, case["sse"], rtol=1e-9, atol=1e-10)
        results[name] = dict(sse=total, folds=folds)
        print("Audited CV:", name, flush=True)
    ranking = sorted(results, key=lambda name: results[name]["sse"])
    assert ranking == selection["ranking"]
    assert ranking[0] == selection["selected_case"]
    joint = [name for name in ranking if not name.startswith("fixed_")]
    assert joint[0] == selection["selected_joint_case"]
    return dict(cases=results, ranking=ranking, maximum_fold_sse_difference=max(errors))


def check_derivatives(context, metadata, saved):
    """Verify the fitted residual Jacobian with independent finite differences."""
    records = []
    for name in dict.fromkeys([metadata["selected_case"], metadata["selected_joint_case"]]):
        config = metadata["cases"][name]["config"]
        theta = saved[f"theta_{name}"]
        for b, block in enumerate(context["blocks"]):
            n = len(block["upper"])
            point = theta[block["parameter_slice"]].copy()
            # Move away from positivity and explicit boundaries for centered differences.
            point[:n] = np.maximum(point[:n], -1 / block["upper"] + .1)
            point[n:] = np.clip(point[n:], [-.3, -.25, -.25], [.3, .25, .25])
            rng = np.random.default_rng(135 + b)
            direction = np.r_[rng.normal(size=n), rng.normal(size=3) * .03]
            if config["ip_family"] == "fixed":
                direction[-3:] = 0
            direction /= np.linalg.norm(direction)
            step = 1e-4
            plus, minus = theta.copy(), theta.copy()
            plus[block["parameter_slice"]] = point + step * direction
            minus[block["parameter_slice"]] = point - step * direction
            positive = independent_project(context, context["y_over_c"][:, :5]
                                           - numpy_components(context, plus, config, count=5))[b]
            negative = independent_project(context, context["y_over_c"][:, :5]
                                           - numpy_components(context, minus, config, count=5))[b]
            finite = ((positive - negative)[:, context["allowed"][b]].ravel()
                      / (.0045 * 2 * step))
            exact = np.asarray(block["models"][config["ip_family"]]["data_jacobian"](
                point, context["y_over_c"][b, :5])) @ direction
            error = float(np.linalg.norm(exact - finite) / max(np.linalg.norm(finite), 1e-20))
            assert error < 3e-5, (name, b, error)
            records.append(dict(case=name, order=int(block["order"]), relative_error=error))
    return records


def check_refits(context, metadata, saved, cv_saved):
    from fit_quadratic_strength import fit_joint_strength

    # Always probe a new-IP case, even if the fixed-IP control wins CV.
    name = metadata["selected_joint_case"]
    config = metadata["cases"][name]["config"]
    expected = saved[f"theta_{name}"]
    rng = np.random.default_rng(853)
    corrupted = context["y_over_c"].copy()
    corrupted[:, 5:] += 100 * rng.normal(size=corrupted[:, 5:].shape)
    repeated, _ = fit_joint_strength(context, config, target=corrupted)
    evaluation_difference = float(np.max(np.abs(repeated - expected)))
    np.testing.assert_allclose(repeated, expected, rtol=0, atol=1e-6)
    corrupted[:, 0] += 100 * rng.normal(size=corrupted[:, 0].shape)
    omitted, _ = fit_joint_strength(context, config, target=corrupted, train_indices=[1, 2, 3, 4])
    omitted_difference = float(np.max(np.abs(omitted - cv_saved[f"theta_{name}_fold0"])))
    np.testing.assert_allclose(omitted, cv_saved[f"theta_{name}_fold0"], rtol=0, atol=1e-6)

    initial = np.zeros(context["n_parameters"])
    initial[context["line_parameter_indices"]] = np.random.default_rng(369).uniform(-.8, .8, 269)
    initial[context["ip_parameter_indices"]] = np.random.default_rng(370).uniform(-.08, .08, 9)
    restarted, diagnostics = fit_joint_strength(context, config, initial=initial)
    original_objective = sum(row["data_objective"] + row["prior_objective"]
                             for row in metadata["cases"][name]["fit"])
    repeated_objective = sum(row["data_objective"] + row["prior_objective"] for row in diagnostics)
    np.testing.assert_allclose(repeated_objective, original_objective, rtol=2e-6, atol=1e-7)
    component = numpy_components(context, restarted, config)
    residual = independent_project(context, context["y_over_c"] - component)
    scores = independent_metrics(context, residual)
    np.testing.assert_allclose(scores["evaluation_score"]["rms_percent"],
        metadata["cases"][name]["metrics"]["evaluation_score"]["rms_percent"], rtol=0, atol=2e-5)
    return dict(case=name, evaluation_target_corruption_difference=evaluation_difference,
                omitted_training_row_corruption_difference=omitted_difference,
                random_restart=dict(original_objective=original_objective,
                    repeated_objective=repeated_objective,
                    evaluation_score_rms_percent=scores["evaluation_score"]["rms_percent"],
                    maximum_parameter_difference=float(np.max(np.abs(restarted - expected)))))


def save_counterfactuals(context, metadata, saved):
    """Swap jointly fitted components to measure conditional model interaction."""
    name = metadata["selected_joint_case"]
    config = metadata["cases"][name]["config"]
    fixed_name = f"fixed_contrast{config['prior_scale']:g}"
    joint = saved[f"theta_{name}"].copy()
    old_ip = joint.copy()
    old_ip[context["ip_parameter_indices"]] = 0
    component_old_ip = numpy_components(context, old_ip, dict(config, ip_family="fixed"))
    old_strength = saved[f"theta_{fixed_name}"].copy()
    old_strength[context["ip_parameter_indices"]] = joint[context["ip_parameter_indices"]]
    component_old_strength = numpy_components(context, old_strength, config)
    arrays = dict(component_joint_strength_old_ip=component_old_ip,
                  component_fixed_strength_joint_ip=component_old_strength,
                  theta_joint_strength_old_ip=old_ip, theta_fixed_strength_joint_ip=old_strength)
    result = dict(joint_case=name, matched_fixed_case=fixed_name,
                  interpretation="Parameter swaps without refitting; conditional interactions, not causal error attribution")
    for key, component in [("joint_strength_old_ip", component_old_ip),
                            ("fixed_strength_joint_ip", component_old_strength)]:
        residual = independent_project(context, context["y_over_c"] - component)
        arrays[f"residual_{key}"] = residual
        result[key] = independent_metrics(context, residual)
    path = OUT / "quadratic_strength_counterfactual.npz"
    np.savez_compressed(path, **arrays)
    result["path"] = str(path.relative_to(ROOT))
    result["sha256"] = sha256(path)
    return result


def main():
    from fit_quadratic_strength import build_context

    previous = json.loads((OUT / "relaxed_strength_independent_audit.json").read_text())
    old_hashes = previous["source_sha256"]
    for name, expected in old_hashes.items():
        assert sha256(ROOT / name) == expected, name
    metadata = json.loads((OUT / "quadratic_strength_fit.json").read_text())
    for name, expected in metadata["source_sha256"].items():
        assert sha256(ROOT / name) == expected, name
    selection = json.loads((OUT / "quadratic_strength_selection.json").read_text())
    assert selection == metadata["selection"]
    saved = np.load(OUT / "quadratic_strength_fit.npz")
    cv_saved = np.load(OUT / "quadratic_strength_cv.npz")
    context = build_context()
    for saved_key, context_key in [("wavelength_um", "wave"), ("included_mask", "allowed"),
        ("score_mask", "score"), ("y_over_c", "y_over_c"), ("continuum", "continuum"),
        ("component_baseline", "reference_component"), ("orders", "orders"),
        ("exposures", "exposures"), ("line_parameter_indices", "line_parameter_indices"),
        ("ip_parameter_indices", "ip_parameter_indices")]:
        np.testing.assert_array_equal(saved[saved_key], context[context_key])
    assert metadata["line_records"] == context["records"]
    contrast = independent_line_contrast(context)
    np.testing.assert_allclose(contrast, context["line_contrast"], rtol=1e-9, atol=1e-12)
    np.testing.assert_array_equal(saved["line_contrast"], context["line_contrast"])
    contrast_error = float(np.max(np.abs(contrast - context["line_contrast"])))
    print("Independent baseline line contrasts agree:", contrast_error, flush=True)
    cases = {}
    for name, case in metadata["cases"].items():
        theta = saved[f"theta_{name}"]
        physical = validate_parameters(context, theta, case["config"])
        component = numpy_components(context, theta, case["config"])
        component_error = float(np.max(np.abs(component - saved[f"component_{name}"])))
        np.testing.assert_allclose(component, saved[f"component_{name}"], rtol=0, atol=1e-9)
        residual = independent_project(context, context["y_over_c"] - component)
        residual_error = float(np.max(np.abs(residual - saved[f"residual_{name}"])))
        np.testing.assert_allclose(residual, saved[f"residual_{name}"], rtol=0, atol=1e-9)
        scores = independent_metrics(context, residual)
        check_metrics(scores, case["metrics"])
        check_objective(context, theta, case["config"], residual, case["fit"], list(range(5)))
        assert physical["zero_intensity_count"] == case["corrections"]["zero_intensity_count"]
        cases[name] = dict(physical=physical, metrics=scores,
                          maximum_component_difference=component_error,
                          maximum_residual_difference=residual_error)
        print("Audited fit:", name, flush=True)
    counterfactuals = save_counterfactuals(context, metadata, saved)
    print("Saved independent counterfactual components", flush=True)
    cv = audit_cv(context, selection, cv_saved)
    derivatives = check_derivatives(context, metadata, saved)
    print("Independent derivatives agree", flush=True)
    refits = check_refits(context, metadata, saved, cv_saved)
    paths = [Path(__file__), ROOT / "audit_relaxed_strength.py", ROOT / "audit_line_strength.py",
             ROOT / "fit_quadratic_strength.py", ROOT / "quadratic_ip_response.py",
             OUT / "quadratic_strength_fit.json", OUT / "quadratic_strength_fit.npz",
             OUT / "quadratic_strength_selection.json", OUT / "quadratic_strength_cv.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                  previous_products_unchanged=old_hashes, cases=cases, cross_validation=cv,
                  maximum_line_contrast_difference=contrast_error,
                  derivatives=derivatives, refits=refits, counterfactuals=counterfactuals,
                  limitations=[
                      "The contrast-dependent prior is a modeling choice, not a calibrated measurement of HITRAN errors.",
                      "Small predicted line contrast does not measure blend identifiability or actual information independently.",
                      "All new parameters are conditional on previously fitted atmosphere, stellar and wavelength nuisance parameters.",
                      "CV and evaluation test the added stage; prior nuisance fits and per-spectrum polynomial estimates remain.",
                      "One random restart assesses local numerical reproducibility, not global uniqueness.",
                  ], source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in paths})
    (OUT / "quadratic_strength_independent_audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("All independent checks passed", flush=True)


if __name__ == "__main__":
    main()
