"""Test explicit intensity priors for otherwise eligible unreported lines."""

from __future__ import annotations

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
import numpy as np
from scipy.optimize import least_squares

from atmosphere_profile import load_profile
from fit_line_strength import (
    build_context as original_context, make_forward, metrics,
    predict_components, prepare_fit_callbacks, WORKING_SIGMA,
)
from line_strength_support import parse_lines, prepare_line_tau

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
PLAN = dict(
    primary="joint30",
    cases={"known_control": [0., False], "new_only30": [.3, True],
           "joint30": [.3, False], "joint10": [.1, False], "joint100": [1., False]},
    selection="Primary 30% scale declared before evaluation; 10% and 100% are sensitivity checks, not evaluation-selected alternatives.",
    eligibility="Same metadata-only thresholds: S296>=1e-25 cm/molecule, within10 km/s of an included pixel and order extent; intensity uncertainty codes0-3.",
    new_prior="Gaussian penalty on fractional intensity change delta, centered at0 with scale s*[0.1+0.9*d^2/(d^2+0.005^2)], truncated by delta>=-1. No upper bound.",
    prior_interpretation="Explicit exploratory shrinkage scales, not HITRAN error bars or calibrated posterior uncertainties. The 100% case is a broad stress test.",
    contrast="Frozen original-model maximum absolute dF/d(delta) on training included pixels before Q5; selection/weights do not use residuals.",
    known_prior="Preserve saved fixed_contrast10 prior widths for all existing269 lines.",
    fixed="PHOENIX, atmosphere, columns, old IP, wavelength solution, continua, original masks and degree-five projection.",
    training_frames=[0, 19, 39, 59, 78], evaluation_frames=[10, 30, 50, 70],
    adoption=False,
)


def build_context():
    """Append correction opacity without double counting existing opacity."""
    context = original_context()
    meta = json.loads((OUT/"motion_clean_control.json").read_text())
    profile = load_profile(meta["layers"])
    with np.load(OUT/"adopted_telluric_fit.npz") as saved:
        adopted = {key: saved[key] for key in saved.files}
    old_z = adopted["parameters"][adopted["line_parameter_indices"]]
    np.testing.assert_allclose(predict_components(context, old_z), adopted["component"], atol=1e-10, rtol=0)
    records, initial, is_new, contrast, old_prior, cursor = [], [], [], [], [], 0
    for b, block in enumerate(context["blocks"]):
        old_slice = block["parameter_slice"]
        original_records = block["records"]
        extra = []
        nu = 1e4/context["wave"][b]
        allowed_nu = nu[context["allowed"][b]]
        for species in ["H2O", "CO2"]:
            for row in parse_lines(species):
                if row["strength_code"] > 3 or row["sw"] < 1e-25:
                    continue
                if not nu.min()*np.exp(-10/C_KMS) <= row["nu"] <= nu.max()*np.exp(10/C_KMS):
                    continue
                separation = float(np.min(np.abs(C_KMS*np.log(allowed_nu/row["nu"]))))
                if separation <= 10:
                    extra.append(dict(row, order=int(block["order"]), nearest_included_velocity_kms=separation))
        nold, nnew = len(original_records), len(extra)
        if nnew:
            tau, factors, tau_metadata = [], [], []
            for species in ["H2O", "CO2"]:
                selected = [row for row in extra if row["species"] == species]
                if not selected:
                    continue
                values, detail = prepare_line_tau(species, selected, block["grid"], profile,
                                                  meta["pressure_shift"], meta["reference_self_broadening"])
                tau.extend(values)
                tau_metadata.append(detail)
                index = next(i for i, row in enumerate(original_records) if row["species"] == species)
                factors.extend([block["line_factors"][:, index]]*len(selected))
            block["line_tau"] = np.vstack([block["line_tau"], tau])
            block["line_factors"] = np.column_stack([block["line_factors"], np.asarray(factors).T])
            block["upper"] = np.r_[block["upper"], np.ones(nnew)]
            block["extra_tau_metadata"] = tau_metadata
        block["records"] = original_records + extra
        block["n_old"] = nold
        n = nold+nnew
        block["parameter_slice"] = slice(cursor, cursor+n)
        cursor += n
        block["train_forward"], block["forward"] = make_forward(block)
        wave = context["wave"][b]
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        inverse = np.linalg.pinv(basis[context["allowed"][b]])
        prepare_fit_callbacks(block, context["allowed"][b], basis, inverse)
        old_d = adopted["line_contrast"][old_slice]
        new_d = np.array([])
        if nnew:
            derivative = np.asarray(jax.jit(jax.jacfwd(block["train_forward"]))(jnp.zeros(n)))
            new_d = np.max(np.abs(derivative[:, context["allowed"][b], nold:]), axis=(0, 1))
        initial.extend(np.r_[old_z[old_slice], np.zeros(nnew)])
        contrast.extend(np.r_[old_d, new_d])
        old_prior.extend(np.r_[1+9*old_d**2/(old_d**2+.005**2), np.ones(nnew)])
        is_new.extend([False]*nold+[True]*nnew)
        records.extend(block["records"])
    context.update(n_parameters=cursor, records=records, initial=np.array(initial),
                   is_new=np.array(is_new), contrast=np.array(contrast), known_prior=np.array(old_prior),
                   adopted_component=adopted["component"], adopted_residual=adopted["residual_filtered"])
    reconstructed = predict_components(context, context["initial"])
    context["adopted_reconstruction_error"] = float(np.max(np.abs(reconstructed-adopted["component"])))
    np.testing.assert_allclose(reconstructed, adopted["component"], atol=1e-10, rtol=0)
    return context


def prior_widths(context, scale):
    widths = context["known_prior"].copy()
    d = context["contrast"][context["is_new"]]
    widths[context["is_new"]] = max(scale, 1e-30)*(.1+.9*d*d/(d*d+.005**2))
    return widths


def fit_case(context, scale=.3, new_only=False, target=None, initial=None):
    """Refit intensities using only five training spectra and fixed priors."""
    target = context["y_over_c"] if target is None else np.asarray(target)
    theta = context["initial"].copy() if initial is None else np.asarray(initial).copy()
    widths = prior_widths(context, scale)
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        sl = block["parameter_slice"]
        new = context["is_new"][sl]
        free = new.copy() if new_only else np.ones(len(new), dtype=bool)
        if scale == 0:
            free &= ~new
            theta[sl][new] = 0.
        if not free.any():
            diagnostics.append(dict(order=block["order"], success=True, n_free=0, nfev=0))
            continue
        template = theta[sl].copy()
        local_widths = widths[sl]
        lower = -1/block["upper"]
        y = jnp.asarray(target[b, :5])

        def expand(values):
            full = template.copy()
            full[free] = values
            return full

        def fun(values):
            full = expand(values)
            return np.r_[np.asarray(block["data_residual"](full, y)), full[free]/local_widths[free]]

        def jac(values):
            return np.vstack([np.asarray(block["data_jacobian"](expand(values), y))[:, free],
                              np.diag(1/local_widths[free])])

        fit = least_squares(fun, np.maximum(template[free], lower[free]+1e-10), jac=jac,
                            bounds=(lower[free], np.inf), method="trf", x_scale="jac",
                            max_nfev=160, ftol=1e-9, xtol=1e-9, gtol=1e-7)
        theta[sl] = expand(fit.x)
        diagnostics.append(dict(order=block["order"], success=bool(fit.success), n_free=int(free.sum()),
                                nfev=fit.nfev, message=fit.message, optimality=float(fit.optimality),
                                objective=float(2*fit.cost),
                                zero_strength_count=int(np.sum(1+block["upper"]*theta[sl] < 1e-6))))
        print("Fit", scale, new_only, diagnostics[-1], flush=True)
    if not all(row["success"] for row in diagnostics):
        raise RuntimeError(f"Unconverged fit: {diagnostics}")
    return theta, diagnostics


def local_metrics(context, residual):
    nu = 1e4/context["wave"][0]
    result = {}
    for name, low, high in [("co2_4819_42", 4819.25, 4819.55),
                             ("stellar_4819_92", 4819.74, 4820.10),
                             ("co2_4818_02", 4817.85, 4818.20),
                             ("co2_4820_43", 4820.25, 4820.60)]:
        result[name] = {}
        for mask_name in ["allowed", "score"]:
            mask = (nu >= low) & (nu <= high) & context[mask_name][0]
            values = residual[0, 5:][:, mask]
            result[name][mask_name] = dict(n_samples=values.size,
                rms_percent=float(100*np.sqrt(np.mean(values**2))) if values.size else None)
    return result


def plot_result(context, residuals):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nu = 1e4/context["wave"][0]
    fig, axes = plt.subplots(3, 1, figsize=(12, 9))
    windows = [(nu.min(), nu.max()), (4817.5, 4818.6), (4819., 4820.7)]
    for ax, (low, high) in zip(axes, windows):
        for name, label, color in [("adopted", "Adopted PHOENIX + telluric", "tab:blue"),
                                   ("joint30", "Unknown intensities free: 30% prior", "tab:orange")]:
            mean = np.mean(residuals[name][0, 5:], axis=0)
            ax.plot(nu, 100*np.where(context["allowed"][0], mean, np.nan),
                    color=color, lw=1, label=label)
        for i in np.flatnonzero(context["is_new"]):
            row = context["records"][i]
            if low <= row["nu"] <= high:
                ax.axvline(row["nu"], color="0.4", ls=":", lw=.8)
        ax.axhline(0, color="0.5", lw=.6)
        ax.set(xlim=(low, high), ylabel="Mean evaluation residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)")
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.suptitle("Explicit intensity priors for previously fixed CO2 lines\nOriginal included pixels shown; global scoring subset unchanged. PHOENIX and IP fixed.")
    fig.tight_layout()
    fig.savefig(OUT/"unknown_strength_fit.png", dpi=170)
    plt.close(fig)


def main():
    started = time.monotonic()
    (OUT/"unknown_strength_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    context = build_context()
    new_indices = np.flatnonzero(context["is_new"])
    print("Additional lines", [(context["records"][i]["nu"], context["records"][i]["strength_code"]) for i in new_indices], flush=True)
    checks = []
    for b, block in enumerate(context["blocks"]):
        sl = block["parameter_slice"]
        if not context["is_new"][sl].any():
            continue
        theta = context["initial"][sl]
        direction = np.random.default_rng(4820).normal(size=len(theta))
        direction[~context["is_new"][sl]] *= .1
        y = jnp.asarray(context["y_over_c"][b, :5])
        h = 1e-5
        fd = (np.asarray(block["data_residual"](theta+h*direction, y))-
              np.asarray(block["data_residual"](theta-h*direction, y)))/(2*h)
        analytic = np.asarray(block["data_jacobian"](theta, y))@direction
        error = float(np.linalg.norm(fd-analytic)/np.linalg.norm(analytic))
        assert error < 1e-5, error
        checks.append(dict(order=block["order"], relative_l2=error))
    base_metrics, base_residual = metrics(context, context["adopted_component"])
    cases = {"adopted": dict(metrics=base_metrics, local=local_metrics(context, base_residual))}
    arrays = dict(theta_initial=context["initial"], is_new=context["is_new"], contrast=context["contrast"],
                  known_prior=context["known_prior"], component_adopted=context["adopted_component"], residual_adopted=base_residual)
    residuals = dict(adopted=base_residual)
    for name, (scale, new_only) in PLAN["cases"].items():
        theta, fit = fit_case(context, scale, new_only)
        component = predict_components(context, theta)
        metric, residual = metrics(context, component)
        widths = prior_widths(context, scale)
        corrections = [dict(context["records"][i], contrast=float(context["contrast"][i]),
                            assumed_fractional_prior_scale=float(widths[i]),
                            fractional_correction=float(theta[i]), strength_ratio=float(1+theta[i]))
                       for i in new_indices]
        cases[name] = dict(scale=scale, new_only=new_only, fit=fit, metrics=metric,
                           local=local_metrics(context, residual), new_line_corrections=corrections)
        arrays.update({f"theta_{name}": theta, f"component_{name}": component,
                       f"residual_{name}": residual, f"prior_{name}": widths})
        residuals[name] = residual
        print(name, "evaluation", metric["evaluation_score"]["rms_percent"], "local", cases[name]["local"], flush=True)
    source_paths = context["source_files"]+[Path(__file__), ROOT/"fit_line_strength.py",
        ROOT/"line_strength_support.py", ROOT/"cache/CO2.par", ROOT/"cache/H2O.par",
        OUT/"adopted_telluric_fit.npz", OUT/"unknown_strength_plan.json"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), plan=PLAN, adopted=False,
                  n_new=int(context["is_new"].sum()), n_existing=int((~context["is_new"]).sum()),
                  records=context["records"], cases=cases, derivative_checks=checks,
                  adopted_reconstruction_error=context["adopted_reconstruction_error"],
                  seconds=time.monotonic()-started,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (OUT/"unknown_strength_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"unknown_strength_fit.npz", orders=context["orders"], exposures=context["exposures"],
                        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
                        y_over_c=context["y_over_c"], **arrays)
    plot_result(context, residuals)
    print("Completed", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
