"""Describe variable-IP fits and contrast-dependent line regularization."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
from scipy.optimize import brentq

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from diagnose_relaxed_strength import residual_statistics, correction_statistics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
BIN_EDGES = [0., .001, .005, .02, np.inf]
BIN_LABELS = ["below_0.1_percent", "0.1_to_0.5_percent", "0.5_to_2_percent", "above_2_percent"]


def load_arrays(path):
    with np.load(path) as saved:
        return {key: saved[key] for key in saved.files}


def project(array, wave, allowed):
    """Independently eliminate degree-five polynomials with a QR projection."""
    result = []
    for values, wavelength, mask in zip(array, wave, allowed):
        coordinate = 2*(np.log(wavelength)-np.log(wavelength).min())/np.ptp(np.log(wavelength))-1
        basis = np.polynomial.chebyshev.chebvander(coordinate, 5)
        q, r = np.linalg.qr(basis[mask], mode="reduced")
        coefficients = np.linalg.solve(r, (values[..., mask] @ q).T).T
        result.append(values-coefficients @ basis.T)
    return np.array(result)


def mixed_fwhm_factor(fraction):
    """Return the full mixture FWHM divided by its narrow-component sigma."""
    peak = (1-fraction)+fraction/2
    halfwidth = brentq(lambda x: (1-fraction)*np.exp(-x*x/2)
                      + fraction/2*np.exp(-x*x/8)-peak/2, 0, 6)
    return 2*halfwidth


def width_curves(meta, arrays, original_parameters):
    """Recover local input-line FWHM on the aligned cube's pixel-coordinate proxy."""
    curves = {}
    bounds = np.asarray(meta["plan"]["ip_bounds"])
    for name, case in meta["cases"].items():
        family = case["config"]["ip_family"]
        parameters = arrays[f"theta_{name}"][arrays["ip_parameter_indices"]].reshape(3, 3)
        per_order = []
        for b, (order, q) in enumerate(zip(arrays["orders"], parameters)):
            wave = arrays["wavelength_um"][b]
            coordinate = np.linspace(-1, 1, len(wave))
            p = original_parameters[0]
            target = 1e4/wave*(1+p[f"stretch_o{order}_kms"]*coordinate/C_KMS)
            pixel_velocity = np.gradient(C_KMS*np.log(target), edge_order=2)
            center_velocity = np.interp((len(wave)-1)/2, np.arange(len(wave)), pixel_velocity)
            old_width = np.array([par[f"ip_sigma_o{order}_kms"]
                                  * np.exp(par.get(f"log_ip_width_e{e}", 0.))
                                  for par, e in zip(original_parameters, arrays["exposures"])])
            fraction = 0. if family == "quadratic_gaussian" else p["ip_wing_fraction"]
            shape = np.ones(len(wave)) if family == "fixed" else (
                np.exp(q[0])*(1+q[1]*coordinate+q[2]*coordinate**2)*pixel_velocity/center_velocity)
            sigma = old_width[:, None]*shape
            fwhm = mixed_fwhm_factor(fraction)*sigma
            fwhm_pixels = fwhm/pixel_velocity
            assert np.all(sigma > 0)
            hits = np.abs(q) >= bounds-1e-5
            assert int(hits.sum()) == case["fit"][b]["ip_bound_count"]
            sample = np.unique(np.r_[np.linspace(0, len(wave)-1, 31, dtype=int), len(wave)//2])
            per_order.append(dict(order=int(order), parameters=q.tolist(), bound_hits=hits.tolist(),
                wing_fraction=float(fraction), coordinate=coordinate[sample].tolist(),
                wavelength_um=wave[sample].tolist(),
                fwhm_kms=fwhm[:, sample].tolist(), fwhm_pixels=fwhm_pixels[:, sample].tolist(),
                reference_exposure=int(arrays["exposures"][0]),
                reference_min_fwhm_kms=float(fwhm[0].min()), reference_max_fwhm_kms=float(fwhm[0].max()),
                reference_min_fwhm_pixels=float(fwhm_pixels[0].min()), reference_max_fwhm_pixels=float(fwhm_pixels[0].max()),
                note="Total local mixture FWHM, not the core alone; original detector coordinates were not retained in the aligned cube"))
        curves[name] = per_order
    return curves


def main():
    paths = [OUT/"quadratic_strength_fit.json", OUT/"quadratic_strength_fit.npz",
             OUT/"relaxed_strength_fit.json", OUT/"relaxed_strength_fit.npz",
             OUT/"poly5_spectral_budget.json", OUT/"nonstellar_spectral_selection.npz",
             OUT/"quadratic_strength_counterfactual.npz", OUT/"motion_clean_control.json"]
    meta = json.loads(paths[0].read_text())
    arrays = load_arrays(paths[1])
    old_meta, old = json.loads(paths[2].read_text()), load_arrays(paths[3])
    regions = json.loads(paths[4].read_text())["region_sets"]["post_filter_training_selection"]
    category = load_arrays(paths[5])["category"]
    counter = load_arrays(paths[6])
    original = json.loads(paths[7].read_text())["parameters"]
    original_parameters = [original]*5
    for exposure in arrays["exposures"][5:]:
        path = OUT/f"nonstellar_control_e{exposure}.json"
        paths.append(path)
        original_parameters.append(json.loads(path.read_text())["parameters"])
    for key in ["wavelength_um", "included_mask", "score_mask", "y_over_c", "continuum"]:
        np.testing.assert_array_equal(arrays[key], old[key])
    assert meta["line_records"] == old_meta["line_records"]
    np.testing.assert_allclose(arrays["component_fixed_uniform"], old["component_soft_prior10"], atol=2e-7, rtol=0)
    joint = meta["selected_joint_case"]
    scale = meta["cases"][joint]["config"]["prior_scale"]
    fixed = f"fixed_contrast{scale:g}"
    compare = ["baseline", "previous_uniform10", fixed, joint]
    residuals = {"baseline": arrays["residual_baseline"][:, 5:],
                 "previous_uniform10": old["residual_soft_prior10"][:, 5:]}
    residuals.update({name: arrays[f"residual_{name}"][:, 5:] for name in meta["cases"]})
    score, allowed, wave = [arrays[key] for key in ["score_mask", "included_mask", "wavelength_um"]]
    overall = {name: residual_statistics(values, score) for name, values in residuals.items()}
    for name, case in meta["cases"].items():
        np.testing.assert_allclose(overall[name]["rms_percent"], case["metrics"]["evaluation_score"]["rms_percent"], atol=1e-12)
    categories = [dict(category=label, statistics={name: residual_statistics(values, score&(category == code))
                  for name, values in residuals.items()})
                  for code, label in enumerate(["H2O-dominant", "CO2-dominant", "Mixed", "Weak absorption"])]
    for name in residuals:
        np.testing.assert_allclose(sum(row["statistics"][name]["sse"] for row in categories), overall[name]["sse"], rtol=1e-12)
    local = []
    for region in regions:
        b, (start, stop) = region["order_index"], region["array_interval"]
        mask = np.zeros_like(score)
        mask[b, start:stop] = allowed[b, start:stop]
        local.append(dict(order=region["order"], peak_wavelength_um=region["peak_wavelength_um"],
            selection="Frozen post-filter training-selected interval", array_interval=[start, stop],
            score={name: residual_statistics(values, score&mask) for name, values in residuals.items()},
            all_included={name: residual_statistics(values, mask) for name, values in residuals.items()}))
    records, contrast = meta["line_records"], arrays["line_contrast"]
    upper = np.array([record["uncertainty_upper"] for record in records])
    strength = np.array([record["sw"] for record in records])
    masks = {"all": np.ones(len(records), dtype=bool)}
    masks.update({label: (contrast >= low)&(contrast < high)
                  for label, low, high in zip(BIN_LABELS, BIN_EDGES[:-1], BIN_EDGES[1:])})
    masks.update({"S296_below_1e-24": strength < 1e-24,
                  "S296_at_least_1e-23": strength >= 1e-23})
    corrections, delta = {}, {}
    for name in meta["cases"]:
        z = arrays[f"theta_{name}"][arrays["line_parameter_indices"]]
        delta[name] = upper*z
        corrections[name] = {label: correction_statistics(z, upper, strength, mask)
                             for label, mask in masks.items() if mask.any()}
    delta["previous_uniform10"] = upper*old["theta_soft_prior10"]
    corrections["previous_uniform10"] = {label: correction_statistics(old["theta_soft_prior10"], upper, strength, mask)
                                         for label, mask in masks.items() if mask.any()}
    example_index = next(i for i, record in enumerate(records)
                         if record["species"] == "H2O" and abs(record["nu"]-6675.85253) < 1e-5)
    example_record = records[example_index]
    example = dict(index=example_index, species=example_record["species"], nu_cm1=example_record["nu"],
                   wavelength_um=1e4/example_record["nu"], reference_strength_296K=example_record["sw"],
                   baseline_derivative_contrast=float(contrast[example_index]),
                   uncertainty_upper_fraction=float(upper[example_index]),
                   fractional_corrections={name: float(delta[name][example_index]) for name in compare[1:]},
                   weighted_prior_multiplier=float(1+(scale-1)*contrast[example_index]**2
                                                    /(contrast[example_index]**2+meta["plan"]["contrast_scale"]**2)))
    formerly_zero = 1+delta["previous_uniform10"] < 1e-5
    zero_followup = dict(previous_zero_count=int(formerly_zero.sum()),
                         remain_zero={name: int(np.sum((1+delta[name] < 1e-5)&formerly_zero)) for name in compare[2:]},
                         previous_zeros_by_contrast={label: int(np.sum(mask&formerly_zero))
                                                     for label, mask in masks.items() if label in BIN_LABELS})
    old_delta, new_delta = delta[fixed], delta[joint]
    changes = dict(
        matched_fixed_case=fixed, joint_case=joint,
        lines_with_smaller_absolute_correction=int(np.sum(abs(new_delta)<abs(old_delta)-1e-8)),
        lines_with_larger_absolute_correction=int(np.sum(abs(new_delta)>abs(old_delta)+1e-8)),
        median_absolute_correction_change=float(np.median(abs(new_delta-old_delta))),
        fixed_contrast_weighted_rms_fraction=float(np.sqrt(np.average(old_delta**2, weights=contrast**2))),
        joint_contrast_weighted_rms_fraction=float(np.sqrt(np.average(new_delta**2, weights=contrast**2))),
        correction_vector_cosine=float(np.dot(new_delta, old_delta)/(np.linalg.norm(new_delta)*np.linalg.norm(old_delta))),
        contrast_weight_note="Descriptive weights only; blended lines are not statistically independent")
    curves = width_curves(meta, arrays, original_parameters)

    # Two independent NumPy forward evaluations supplied by the numerical audit
    # hold one part of the joint solution at its matched fixed-IP value.
    counterfactual = {}
    for name in ["component_joint_strength_old_ip", "component_fixed_strength_joint_ip"]:
        filtered = project(arrays["y_over_c"]-counter[name], wave, allowed)
        counterfactual[name] = residual_statistics(filtered[:, 5:], score)
    ip_change = project(counter["component_fixed_strength_joint_ip"]-arrays[f"component_{fixed}"], wave, allowed)[:, 5:]
    line_change = project(arrays[f"component_{joint}"]-counter["component_fixed_strength_joint_ip"], wave, allowed)[:, 5:]
    keep = np.broadcast_to(score[:, None], ip_change.shape)
    ip_values, line_values = ip_change[keep], line_change[keep]
    counterfactual["decomposition"] = dict(
        ip_model_change_rms_percent=float(100*np.sqrt(np.mean(ip_values**2))),
        subsequent_strength_model_change_rms_percent=float(100*np.sqrt(np.mean(line_values**2))),
        signed_cosine=float(np.dot(ip_values, line_values)/(np.linalg.norm(ip_values)*np.linalg.norm(line_values))),
        definition="IP change at fixed-case strengths; then strength change with selected joint IP. No parameter refits.",
        interpretation="Negative cosine indicates compensation in model-space, not a uniquely identified physical error")
    np.testing.assert_allclose(residuals[fixed]-ip_change-line_change, residuals[joint], atol=1e-11, rtol=0)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), selected_case=meta["selected_case"],
        selected_joint_case=joint, matched_fixed_case=fixed, comparison_cases=compare,
        best_fixed_case=next(name for name in meta["selection"]["ranking"] if name.startswith("fixed_")),
        evaluation_exposures=arrays["exposures"][5:].tolist(), overall=overall, categories=categories,
        regions=local, corrections=corrections, ip_strength_correction_changes=changes,
        former_extreme_H2O_example=example, previous_zero_intensities_followup=zero_followup,
        ip_width_curves=curves, counterfactuals=counterfactual,
        caveats=["All metrics use the same continuum-normalized, degree-five-filtered residual and fixed score mask.",
                 "The new line and IP fits are conditional on frozen atmospheric, stellar, wavelength and epoch-IP nuisance parameters.",
                 "Weakness means baseline predicted spectral derivative contrast, not solely reference line strength at 296 K.",
                 "The aligned cube pixel index is a proxy for original detector position; published IGRINS RV coefficients were not imported.",
                 "Effective line corrections and IP changes can compensate one another; counterfactuals do not identify a unique cause.",
                 "IP curves show the local input-line total FWHM, including the broad component in mixture cases."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths+[Path(__file__), ROOT/"diagnose_relaxed_strength.py"]})
    (OUT/"quadratic_strength_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")

    labels = {"baseline": "Fixed strengths", "previous_uniform10": "Old uniform10", fixed: f"Fixed IP, weighted{scale:g}", joint: "Joint IP + weighted"}
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("IP curves: exposure 0; shading: range across four evaluation exposures, not uncertainty", fontsize=10)
    curve_names = ["fixed_uniform", f"quadratic_gaussian_contrast{scale:g}", f"quadratic_mixture_contrast{scale:g}"]
    for b, ax in enumerate(axes[0]):
        for name, label in zip(curve_names, ["Previous IP", "Quadratic Gaussian", "Quadratic mixture"]):
            curve = np.asarray(curves[name][b]["fwhm_pixels"])
            curve_wave = curves[name][b]["wavelength_um"]
            ax.plot(curve_wave, curve[0], label=label, lw=1.3)
            ax.fill_between(curve_wave, curve[5:].min(axis=0), curve[5:].max(axis=0), alpha=.1)
        ax.set(xlabel="Observed wavelength (micron)", ylabel="Local IP FWHM (cube pixels)", title=f"Order {arrays['orders'][b]}")
        ax.legend(fontsize=7)
    x = np.arange(len(compare))
    common = np.array([overall[name]["common_rms_percent"]**2 for name in compare])
    varying = np.array([overall[name]["temporal_rms_percent"]**2 for name in compare])
    axes[1, 0].bar(x, common, label="Exposure-common")
    axes[1, 0].bar(x, varying, bottom=common, label="Time-varying")
    axes[1, 0].set(xticks=x, xticklabels=[labels[name] for name in compare], ylabel="Evaluation mean-square residual (%²)")
    axes[1, 0].tick_params(axis="x", labelrotation=20, labelsize=7)
    axes[1, 0].legend(fontsize=7)
    for i, name in enumerate(compare[1:]):
        values = [100*corrections[name][label]["median_absolute_fraction"] if label in corrections[name] else np.nan for label in BIN_LABELS]
        axes[1, 1].bar(np.arange(4)+(i-1)*.25, values, width=.25, label=labels[name])
    axes[1, 1].set(xticks=np.arange(4), xticklabels=["<0.1%", "0.1–0.5%", "0.5–2%", ">2%"],
                   xlabel="Baseline line contrast", ylabel="Median absolute strength change (%)")
    axes[1, 1].legend(fontsize=7)
    scatter = axes[1, 2].scatter(100*old_delta, 100*new_delta, c=np.log10(100*contrast), s=15, alpha=.7)
    lo, hi = min(old_delta.min(), new_delta.min())*100, max(old_delta.max(), new_delta.max())*100
    axes[1, 2].plot([lo, hi], [lo, hi], color="0.5", ls="--", lw=.8)
    axes[1, 2].set(xlabel="Strength correction with fixed IP (%)", ylabel="Strength correction with joint IP (%)")
    fig.colorbar(scatter, ax=axes[1, 2], label="log10 baseline contrast (%)", shrink=.7)
    fig.savefig(OUT/"quadratic_strength_diagnostics.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(comparison={name: overall[name] for name in compare},
                          correction_changes=changes, counterfactuals=counterfactual), indent=2))


if __name__ == "__main__":
    main()
