"""Independently check selected HITRAN intensity corrections and fit products."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import wofz

from atmosphere_profile import load_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
UPPER = {4: .2, 5: .1, 6: .05, 7: .02, 8: .01, 9: .001}
C_KMS = 299792.458


def independent_filter(values, wavelength, mask):
    """Evaluate the degree-five residual projection using an independent QR."""
    velocity = C_KMS*np.log(wavelength)
    x = 2*(velocity-velocity.min())/np.ptp(velocity)-1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    q, upper = np.linalg.qr(basis[mask], mode="reduced")
    coefficients = np.linalg.solve(upper, q.T @ values[..., mask].T)
    return values-(basis @ coefficients).T


def independent_tau(raw, indices, grid, profile, species):
    """Compute Voigt opacity with SciPy, without the ExoJAX line evaluator."""
    nu = raw["nu"][indices]
    mass = raw["line_mass"][indices]
    isotope_positions = np.searchsorted(raw["partition_isotopes"], raw["iso"][indices])
    result = np.zeros((len(indices), len(grid)))
    for t, pressure, number, vmr in zip(profile["temperature_K"], profile["pressure_bar"],
                                         profile["number_column_cm2"], profile["vmr"][species]):
        partition_ratio = np.array([np.interp(t, raw["temperatures"], row)
                                    / np.interp(296., raw["temperatures"], row)
                                    for row in raw["partition_sums"]])[isotope_positions]
        strength = (raw["sw"][indices] / partition_ratio
                    * np.exp(-1.4387773538277202*raw["elower"][indices]*(1/t-1/296.))
                    * np.expm1(-1.4387773538277202*nu/t)
                    / np.expm1(-1.4387773538277202*nu/296.))
        sigma = 3.0415595e-7*np.sqrt(t/mass)*nu
        gamma = ((296./t)**raw["n_air"][indices]*raw["gamma_air"][indices]*pressure/1.01325
                 + 2.6544188e-12*raw["a"][indices])
        centers = nu+raw["delta_air"][indices]*pressure/1.01325
        z = (grid[None]-centers[:, None]+1j*gamma[:, None])/(np.sqrt(2)*sigma[:, None])
        profile_values = wofz(z).real/(np.sqrt(2*np.pi)*sigma[:, None])
        result += number*vmr*strength[:, None]*profile_values
    return result


def audit_support():
    support_path = OUT / "line_strength_support.json"
    support = json.loads(support_path.read_text())
    source = np.load(OUT / "motion_clean_control.npz")
    profile = load_profile(20)
    reports = []
    for block in support:
        species, order = block["species"], block["order"]
        b = int(np.flatnonzero(source["orders"] == order)[0])
        wave = source["wavelength_um"][b]
        keep = source["included_mask"][b]
        nu = 1e4/wave
        raw_path = ROOT / "cache" / f"{species}.par"
        lines = raw_path.read_text().splitlines()
        all_nu = np.array([float(row[3:15]) for row in lines])
        all_strength = np.array([float(row[15:25]) for row in lines])
        all_code = np.array([int(row[128]) for row in lines])
        candidates = np.flatnonzero((all_strength >= 1e-25)
                                   & (all_nu >= nu.min()*np.exp(-10/C_KMS))
                                   & (all_nu <= nu.max()*np.exp(10/C_KMS)))
        nearest = np.min(np.abs(C_KMS*np.log(nu[keep, None]/all_nu[candidates][None])), axis=0)
        candidates = candidates[nearest <= 10]
        expected = candidates[np.isin(all_code[candidates], list(UPPER))]
        indices = np.array([record["raw_index"] for record in block["records"]])
        np.testing.assert_array_equal(indices, expected)
        for record in block["records"]:
            row = lines[record["raw_index"]]
            assert record["ierr"] == row[127:133]
            assert record["strength_code"] == int(row[128])
            assert record["uncertainty_upper"] == UPPER[int(row[128])]
            assert record["nu"] == float(row[3:15])
            assert record["sw"] == float(row[15:25])
        npz_path = ROOT / block["tau_metadata"]["path"]
        cached = np.load(npz_path)
        raw = np.load(ROOT / "cache" / f"{species}.npz")
        np.testing.assert_array_equal(cached["raw_indices"], indices)
        assert block["tau_metadata"]["raw_sha256"] == hashlib.sha256(raw_path.read_bytes()).hexdigest()
        independent = independent_tau(raw, indices, cached["grid"], profile, species)
        difference = independent-cached["tau"]
        relative = float(np.linalg.norm(difference)/np.linalg.norm(independent))
        maximum = float(np.max(np.abs(difference)))
        assert relative < 3e-5, relative
        reports.append(dict(order=order, species=species, selected_count=len(indices),
                            eligible_count=len(candidates), excluded_count=len(candidates)-len(indices),
                            scipy_vs_exojax_relative_l2=relative,
                            scipy_vs_exojax_maximum_tau_difference=maximum))
    return dict(passed=True, selected_total=sum(item["selected_count"] for item in reports),
                blocks=reports, note="Independent SciPy Voigt checks the new per-line opacity basis; physical constants match the existing model.")


def numpy_component(block, theta):
    """Check the new nonlinear update using NumPy convolution and interpolation."""
    grid, velocity = np.asarray(block["grid"]), np.asarray(block["velocity"])
    delta_tau = (np.asarray(block["line_factors"])*(block["upper"]*theta)[None]) @ block["line_tau"]
    total_tau = np.asarray(block["baseline_tau"])+delta_tau
    assert np.min(total_tau) >= -1e-12
    intrinsic = block["shifted_star"]*np.exp(-block["airmass"][:, None]*total_tau)
    answer = []
    for row, sigma, rv in zip(intrinsic, block["widths"], block["offsets"]):
        parts = []
        for width in [sigma, 2*sigma]:
            kernel = np.exp(-.5*(velocity/width)**2)
            kernel /= kernel.sum()
            parts.append(np.interp(np.asarray(block["target"])*(1+rv/C_KMS), grid,
                                   np.convolve(row, kernel, mode="same")))
        answer.append((1-block["wing_fraction"])*parts[0]+block["wing_fraction"]*parts[1])
    return np.array(answer)


def audit_fit_products():
    import jax.numpy as jnp
    from fit_line_strength import build_context, fit_strengths

    meta = json.loads((OUT / "line_strength_fit.json").read_text())
    saved = np.load(OUT / "line_strength_fit.npz")
    context = build_context()
    source = np.load(OUT / "motion_clean_control.npz")
    np.testing.assert_array_equal(saved["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    np.testing.assert_array_equal(saved["included_mask"], source["included_mask"])
    np.testing.assert_array_equal(saved["score_mask"], source["score_mask"][:, -1])
    independent_y = [source["observed"][:, :-1]/source["continuum"][:, :-1]]
    independent_c = [source["continuum"][:, :-1]]
    independent_component = [source["component"][:, :-1]]
    for exposure in [10, 30, 50, 70]:
        other = np.load(OUT / f"nonstellar_control_e{exposure}.npz")
        independent_y.append(other["observed"][:, -1:]/other["continuum"][:, -1:])
        independent_c.append(other["continuum"][:, -1:])
        independent_component.append(other["component"][:, -1:])
    y = np.concatenate(independent_y, axis=1)
    continuum = np.concatenate(independent_c, axis=1)
    np.testing.assert_array_equal(saved["y_over_c"], y)
    np.testing.assert_array_equal(saved["continuum"], continuum)
    np.testing.assert_array_equal(saved["component_baseline"], np.concatenate(independent_component, axis=1))
    assert meta["line_records"] == context["records"]
    support = json.loads((OUT / "line_strength_support.json").read_text())
    expected_lines = [dict(record, order=block["order"]) for block in support for record in block["records"]]
    assert meta["line_records"] == expected_lines
    assert meta["n_parameters"] == 269
    mask = saved["included_mask"]
    wave = saved["wavelength_um"]

    def project(array):
        return np.array([independent_filter(row, w, keep) for row, w, keep in zip(array, wave, mask)])

    records = []
    maximum_component_error, maximum_residual_error = 0., 0.
    for tag in ["baseline", "primary", "half_prior_width", "double_prior_width"]:
        theta = np.zeros(269) if tag == "baseline" else saved[f"theta_{tag}"]
        assert np.max(np.abs(theta)) <= 1+1e-12
        components = np.array([numpy_component(block, theta[block["parameter_slice"]])
                               for block in context["blocks"]])
        component_error = float(np.max(abs(components-saved[f"component_{tag}"])))
        maximum_component_error = max(maximum_component_error, component_error)
        np.testing.assert_allclose(components, saved[f"component_{tag}"], rtol=0, atol=1e-10)
        residual = project(y-saved[f"component_{tag}"])
        residual_error = float(np.max(abs(residual-saved[f"residual_{tag}"])))
        maximum_residual_error = max(maximum_residual_error, residual_error)
        np.testing.assert_allclose(residual, saved[f"residual_{tag}"], rtol=0, atol=1e-12)
        case = meta["baseline"] if tag == "baseline" else meta["cases"][tag]["metrics"]
        metrics = {}
        for scope, es, keep in [("training", slice(0, 5), mask),
                                ("evaluation_score", slice(5, None), saved["score_mask"]),
                                ("evaluation_included", slice(5, None), mask)]:
            r = residual[:, es]
            selection = np.broadcast_to(keep[:, None], r.shape)
            rms = float(100*np.sqrt(np.mean(r[selection]**2)))
            fixed = r*continuum[:, es]
            fixed_rms = float(100*np.sqrt(np.mean(fixed[selection]**2)))
            np.testing.assert_allclose(rms, case[scope]["rms_percent"], atol=1e-11, rtol=0)
            np.testing.assert_allclose(fixed_rms, case[scope]["fixed_rms_percent"], atol=1e-11, rtol=0)
            for b, order in enumerate(saved["orders"]):
                value = float(100*np.sqrt(np.mean(r[b][selection[b]]**2)))
                np.testing.assert_allclose(value, case[scope]["per_order"][str(order)], atol=1e-11, rtol=0)
            common = np.broadcast_to(r.mean(axis=1)[:, None], r.shape)
            variable = r-common
            common_rms = float(100*np.sqrt(np.mean(common[selection]**2)))
            variable_rms = float(100*np.sqrt(np.mean(variable[selection]**2)))
            np.testing.assert_allclose(common_rms**2+variable_rms**2, rms**2, rtol=1e-12)
            metrics[scope] = dict(rms_percent=rms, fixed_rms_percent=fixed_rms,
                                  common_rms_percent=common_rms, variable_rms_percent=variable_rms)
        if tag != "baseline":
            case_meta = meta["cases"][tag]
            np.testing.assert_equal(np.sum(abs(theta)>.999), case_meta["at_cap_count"])
            prior_scale = case_meta["prior_scale"]
            for b, block in enumerate(context["blocks"]):
                sl = block["parameter_slice"]
                objective = float(np.sum((residual[b, :5, :][:, mask[b]]/.0045)**2))
                prior = float(np.sum((theta[sl]/prior_scale)**2))
                np.testing.assert_allclose(objective, case_meta["fit"][b]["data_objective"], rtol=1e-10)
                np.testing.assert_allclose(prior, case_meta["fit"][b]["prior_objective"], rtol=1e-12)
            upper = np.array([r["uncertainty_upper"] for r in meta["line_records"]])
            for species in ["H2O", "CO2"]:
                keep = np.array([r["species"] == species for r in meta["line_records"]])
                delta = theta[keep]*upper[keep]
                strength = np.array([r["sw"] for r in meta["line_records"]])[keep]
                check = case_meta["species_statistics"][species]
                np.testing.assert_allclose(np.mean(delta), check["mean_fractional_correction"], rtol=1e-12)
                np.testing.assert_allclose(np.average(delta, weights=strength), check["strength_weighted_mean_fractional_correction"], rtol=1e-12)
        records.append(dict(case=tag, metrics=metrics, component_maximum_error=component_error,
                            residual_maximum_error=residual_error))

    # A full repeated fit with corrupted held-out targets tests the actual fitting
    # entry point, rather than only reproducing its training-array slice.
    corrupted = y.copy()
    corrupted[:, 5:] += np.random.default_rng(853).normal(size=corrupted[:, 5:].shape)*100
    repeated, _ = fit_strengths(context, target=corrupted, prior_scale=1.)
    leakage_error = float(np.max(np.abs(repeated-saved["theta_primary"])))
    np.testing.assert_allclose(repeated, saved["theta_primary"], rtol=0, atol=1e-7)
    restarted, restart_diagnostics = fit_strengths(
        context, prior_scale=1., initial=np.random.default_rng(369).uniform(-.8, .8, 269))
    original_objective = sum(item["data_objective"]+item["prior_objective"]
                             for item in meta["cases"]["primary"]["fit"])
    restarted_objective = sum(item["data_objective"]+item["prior_objective"]
                              for item in restart_diagnostics)
    np.testing.assert_allclose(restarted_objective, original_objective, rtol=1e-8, atol=1e-8)
    restarted_components = np.array([numpy_component(block, restarted[block["parameter_slice"]])
                                     for block in context["blocks"]])
    restarted_residual = project(y-restarted_components)[:, 5:]
    scoring = np.broadcast_to(saved["score_mask"][:, None], restarted_residual.shape)
    restarted_rms = float(100*np.sqrt(np.mean(restarted_residual[scoring]**2)))
    np.testing.assert_allclose(restarted_rms,
                               meta["cases"]["primary"]["metrics"]["evaluation_score"]["rms_percent"],
                               rtol=0, atol=1e-6)
    return dict(passed=True, records=records,
                maximum_independent_component_error=maximum_component_error,
                maximum_independent_filter_error=maximum_residual_error,
                heldout_target_corruption_parameter_difference=leakage_error,
                random_restart=dict(seed=369, range=[-.8, .8],
                                    original_objective=original_objective,
                                    restarted_objective=restarted_objective,
                                    evaluation_score_rms_percent=restarted_rms,
                                    maximum_parameter_difference=float(np.max(abs(restarted-saved["theta_primary"])))),
                likelihood_note="Projected least squares with fixed working noise and assumed shrinkage; not a calibrated posterior.",
                interpretation_note="Correlated/blended line corrections and shared abundance modes are not separately identified.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--support-only", action="store_true")
    args = parser.parse_args()
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), support=audit_support(),
                  fit_products_checked=False)
    if not args.support_only:
        result["fit"] = audit_fit_products()
        result["fit_products_checked"] = True
    result["passed"] = result["support"]["passed"] and (args.support_only or result["fit"]["passed"])
    paths = [Path(__file__), OUT / "line_strength_support.json"]
    if not args.support_only:
        paths += [OUT / "line_strength_fit.json", OUT / "line_strength_fit.npz", ROOT / "fit_line_strength.py"]
    result["source_sha256"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                               for path in paths}
    (OUT / "line_strength_independent_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Support passed:", result["support"]["selected_total"], "selected lines")
    print("Fit products checked:", result["fit_products_checked"])


if __name__ == "__main__":
    main()
