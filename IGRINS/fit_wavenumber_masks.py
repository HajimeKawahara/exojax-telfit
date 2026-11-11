"""Compare pure masking, masked Q5, and exact masked strength refitting."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from fit_line_strength import build_context, predict_components, prepare_fit_callbacks, WORKING_SIGMA

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def make_masked_context(base, included, score, contrast):
    """Keep lines and priors fixed while changing fit and evaluation masks."""
    context = dict(base, allowed=np.array(included, copy=True), score=np.array(score, copy=True))
    blocks, bases, inverses = [], [], []
    for b, original in enumerate(base["blocks"]):
        block = dict(original)
        w = base["wave"][b]
        x = 2*(np.log(w)-np.log(w).min())/np.ptp(np.log(w))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        inverse = np.linalg.pinv(basis[included[b]])
        prepare_fit_callbacks(block, included[b], basis, inverse)
        d = contrast[block["parameter_slice"]]
        block["prior_widths"] = 1+9*d*d/(d*d+.005**2)
        blocks.append(block); bases.append(basis); inverses.append(inverse)

    def project(values):
        return np.array([row-(row[..., mask] @ inverse.T) @ basis.T
                         for row, mask, inverse, basis in zip(values, included, inverses, bases)])

    context.update(blocks=blocks, project=project)
    return context


def fit_masked(context, target=None, initial=None):
    """Fit only five training exposures with the adopted positive strength prior."""
    target = context["y_over_c"] if target is None else np.asarray(target)
    theta = np.zeros(context["n_parameters"]) if initial is None else np.array(initial, copy=True)
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        sl, prior = block["parameter_slice"], block["prior_widths"]
        lower = -1/block["upper"]
        training = jnp.asarray(target[b, :5])

        def data(z):
            return np.asarray(block["data_residual"](z, training))

        def fun(z):
            return np.r_[data(z), z/prior]

        def jac(z):
            return np.vstack([np.asarray(block["data_jacobian"](z, training)), np.diag(1/prior)])

        fit = least_squares(fun, np.maximum(theta[sl], lower+1e-10), jac=jac,
                            bounds=(lower, np.inf), x_scale="jac", max_nfev=300,
                            ftol=1e-12, xtol=1e-12, gtol=1e-8)
        theta[sl] = fit.x
        diagnostics.append(dict(order=int(block["order"]), success=bool(fit.success),
            nfev=int(fit.nfev), optimality=float(fit.optimality), message=fit.message,
            data_objective=float(np.sum(data(fit.x)**2)), prior_objective=float(np.sum((fit.x/prior)**2)),
            n_fit_pixels=int(context["allowed"][b].sum()),
            zero_strengths=int(np.sum(1+block["upper"]*fit.x < 1e-6))))
    if not all(row["success"] for row in diagnostics):
        raise RuntimeError(diagnostics)
    return theta, diagnostics


def statistics(residual, mask, exposures):
    rows = residual[:, exposures]
    keep = np.broadcast_to(mask[:, None], rows.shape)
    mean = np.broadcast_to(rows.mean(axis=1)[:, None], rows.shape)
    return dict(n_samples=int(keep.sum()), rms_percent=float(100*np.sqrt(np.mean(rows[keep]**2))),
                sse=float(np.sum(rows[keep]**2)),
                common_rms_percent=float(100*np.sqrt(np.mean(mean[keep]**2))),
                temporal_rms_percent=float(100*np.sqrt(np.mean((rows-mean)[keep]**2))))


def all_metrics(context, residual):
    return dict(training=statistics(residual, context["allowed"], slice(0, 5)),
                evaluation_score=statistics(residual, context["score"], slice(5, None)),
                evaluation_included=statistics(residual, context["allowed"], slice(5, None)))


def main():
    base = build_context()
    mask_path = OUT/"wavenumber_mask_candidates.npz"
    with np.load(mask_path) as data:
        masks = {key: data[key] for key in data.files}
    with np.load(OUT/"adopted_telluric_fit.npz") as data:
        z0 = data["parameters"][data["line_parameter_indices"]]
        contrast = data["line_contrast"]
        component = data["component"]
        original_residual = data["residual_filtered"]
    np.testing.assert_allclose(predict_components(base, z0), component, atol=1e-10, rtol=0)
    names = masks["case_names"].tolist()
    arrays = dict(orders=base["orders"], exposures=base["exposures"], wavelength_um=base["wave"],
                  original_included_mask=base["allowed"], original_score_mask=base["score"],
                  y_over_c=base["y_over_c"], continuum=base["continuum"],
                  component_adopted=component, residual_adopted=original_residual,
                  strength_z_adopted=z0, line_contrast=contrast, case_names=np.array(names),
                  included_masks=masks["included_masks"], score_masks=masks["score_masks"],
                  exclusion_masks=masks["exclusion_masks"])
    cases = {}
    for i, name in enumerate(names):
        context = make_masked_context(base, masks["included_masks"][i], masks["score_masks"][i], contrast)
        projected = context["project"](context["y_over_c"]-component)
        theta, diagnostics = fit_masked(context, initial=z0)
        prediction = predict_components(context, theta)
        residual = context["project"](context["y_over_c"]-prediction)
        cases[name] = dict(fit=diagnostics,
            pure_mask=all_metrics(context, original_residual),
            reproject_only=all_metrics(context, projected),
            strength_refit=all_metrics(context, residual),
            remaining_change_from_reprojection=all_metrics(context, projected-original_residual),
            remaining_change_from_refit=all_metrics(context, residual-projected))
        arrays.update({f"theta_{name}":theta, f"component_{name}":prediction,
                       f"residual_reproject_{name}":projected, f"residual_refit_{name}":residual})
        print(name, "pure", cases[name]["pure_mask"]["evaluation_score"]["rms_percent"],
              "Q5", cases[name]["reproject_only"]["evaluation_score"]["rms_percent"],
              "refit", cases[name]["strength_refit"]["evaluation_score"]["rms_percent"], flush=True)
    paths = base["source_files"]+[Path(__file__), ROOT/"fit_line_strength.py", ROOT/"line_strength_support.py",
        mask_path, OUT/"wavenumber_mask_candidates.json", OUT/"adopted_telluric_fit.npz", OUT/"adopted_telluric_model.json"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), cases=cases,
        interpretation="Pure masking leaves all retained adopted residual values unchanged. Q5 and strength-refit experiments additionally change retained residuals.",
        fixed="Original IP, atmosphere/columns, star, wavelength, 269 lines and adopted contrast-dependent prior weights",
        mask_source="Training-only OOF residual ranking; same mask for every exposure and frozen during injections",
        solver="Bounded nonlinear least squares with analytic Jacobian, ftol=xtol=1e-12, gtol=1e-8",
        limitations=["Scoring different pixel subsets lowers RMS without improving the discarded observations.",
                     "The new masks do not retroactively remove bad points from the older physical nuisance fits.",
                     "Evaluation spectra determine neither masks nor fitted strength coefficients.",
                     "Earlier nuisance fits and previously inspected evaluation exposures limit independence.",
                     "No candidate mask or refitted model is automatically adopted."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    (OUT/"wavenumber_mask_refit.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT/"wavenumber_mask_refit.npz", **arrays)


if __name__ == "__main__":
    main()
