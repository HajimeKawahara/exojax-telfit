"""Adopt the degree-five residual filter and describe the remaining error.

The time-common pattern is a diagnostic; it is not subtracted from the
adopted data product. Planetary templates must receive the same spatial filter.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smooth_residuals import smoother_matrix

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EXPOSURES = [10, 30, 50, 70]
FILTER = dict(kind="polynomial", degree=5)


def load_fit(tag):
    with np.load(OUT / f"{tag}.npz") as data:
        return {key: data[key] for key in data.files}


def statistics(values):
    values = np.asarray(values)
    return dict(n_samples=int(values.size), rms_percent=float(100*np.sqrt(np.mean(values**2))),
                mean_percent=float(100*np.mean(values)),
                mad_percent=float(100*1.4826*np.median(np.abs(values-np.median(values)))))


def budget(residual, mask):
    keep = np.broadcast_to(mask[:, None], residual.shape)
    common = np.broadcast_to(residual.mean(axis=1)[:, None], residual.shape)
    temporal = residual-common
    total = float(np.mean(residual[keep]**2))
    powers = [float(np.mean(value[keep]**2)) for value in [common, temporal]]
    cross = float(2*np.mean(common[keep]*temporal[keep]))
    np.testing.assert_allclose(sum(powers), total, atol=1e-14, rtol=1e-12)
    return dict(total=statistics(residual[keep]),
                common=dict(**statistics(common[keep]), squared_error_fraction=powers[0]/total),
                temporal=dict(**statistics(temporal[keep]), squared_error_fraction=powers[1]/total),
                twice_cross_product=cross, closure_error=sum(powers)-total)


def transfer(residual, prediction, mask):
    prediction = np.broadcast_to(prediction, residual.shape)
    keep = np.broadcast_to(mask[:, None], residual.shape)
    remainder = residual-prediction
    total = float(np.mean(residual[keep]**2))
    static = float(np.mean(prediction[keep]**2))
    rest = float(np.mean(remainder[keep]**2))
    cross = float(2*np.mean(prediction[keep]*remainder[keep]))
    np.testing.assert_allclose(static+rest+cross, total, rtol=1e-12, atol=1e-14)
    return dict(before=statistics(residual[keep]), after=statistics(remainder[keep]),
                prediction_mse=static, remainder_mse=rest, twice_cross_product=cross,
                total_mse=total, squared_error_reduction_fraction=1-rest/total)


def main():
    source = load_fit("motion_clean_control")
    included, wave, orders = [source[key] for key in ["included_mask", "wavelength_um", "orders"]]
    score = source["score_mask"][:, -1]
    operators = np.array([smoother_matrix(w, m, FILTER) for w, m in zip(wave, included)])
    residuals, continua, physical, observed = [], [], [], []
    paths = [OUT / "motion_clean_control.npz", ROOT / "smooth_residuals.py", Path(__file__)]
    for exposure in EXPOSURES:
        data = load_fit(f"nonstellar_control_e{exposure}")
        np.testing.assert_array_equal(data["included_mask"], included)
        np.testing.assert_array_equal(data["score_mask"][:, -1], score)
        np.testing.assert_allclose(data["model"][:, :-1], source["model"][:, :-1], rtol=0, atol=1e-12)
        continuum = data["continuum"][:, -1]
        residuals.append((data["observed"][:, -1]-data["model"][:, -1])/continuum)
        continua.append(continuum)
        physical.append(data["model"][:, -1])
        observed.append(data["observed"][:, -1])
        paths.append(OUT / f"nonstellar_control_e{exposure}.npz")
    before = np.stack(residuals, axis=1)
    continuum = np.stack(continua, axis=1)
    physical = np.stack(physical, axis=1)
    observed = np.stack(observed, axis=1)
    removed = np.einsum("bij,bej->bei", operators, before)
    residual = before-removed
    model = physical+continuum*removed
    np.testing.assert_allclose((observed-model)/continuum, residual, atol=1e-14, rtol=0)
    with np.load(OUT / "smooth_results.npz") as previous:
        np.testing.assert_allclose(removed, previous["correction_poly5_all_included"], atol=1e-13, rtol=0)
    for operator in operators:
        np.testing.assert_allclose(operator @ operator, operator, atol=1e-12, rtol=0)
    remaining_projection = np.einsum("bij,bej->bei", operators, residual)
    np.testing.assert_allclose(remaining_projection, 0, atol=1e-12, rtol=0)
    training = (source["observed"][:, :-1]-source["model"][:, :-1])/source["continuum"][:, :-1]
    training_filtered = training-np.einsum("bij,bej->bei", operators, training)
    template = training_filtered.mean(axis=1)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  filter=FILTER, exposure_ids=EXPOSURES, orders=orders.tolist(),
                  definition="Residual after spatial filter: (I-H)*(observed-physical_model)/original_continuum",
                  primary_scope="score subset for comparison; these pixels participate in the polynomial fit",
                  included_wavelength_count=int(included.sum()), scoring_wavelength_count=int(score.sum()),
                  budgets={}, template_transfer={}, per_order={},
                  validation=dict(polynomial_idempotence="passed", removed_matches_previous_product="passed",
                                  residual_has_zero_full_mask_polynomial_projection="passed"),
                  limitations=["Statistics after all-included filtering, not independent predictive accuracy.",
                               "Four-exposure mean includes finite-sample noise; common is not equivalent to physical model error.",
                               "Statistical and wavelength-category fractions are not causal source fractions.",
                               "The training residual template is diagnostic only and is not part of the adopted correction.",
                               "Original pixel mask retained; unknown stellar structure is not proven absent.",
                               "No per-pixel error arrays; filtered residuals are correlated.",
                               "The initial telluric parameters remain fixed; planetary templates require the same spatial operator."],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in paths})
    for scope, mask in [("score", score), ("included", included)]:
        result["budgets"][scope] = dict(continuum=budget(residual, mask),
                                      fixed_flux=budget(residual*continuum, mask))
        result["template_transfer"][scope] = dict(
            continuum=transfer(residual, template[:, None], mask),
            fixed_flux=transfer(residual*continuum, continuum*template[:, None], mask))
        result["per_order"][scope] = {str(order): dict(
            continuum=budget(residual[b:b+1], mask[b:b+1]),
            fixed_flux=budget((residual*continuum)[b:b+1], mask[b:b+1]))
            for b, order in enumerate(orders)}
    (OUT / "poly5_residual_budget.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "poly5_residual_budget.npz", orders=orders, exposures=EXPOSURES,
                        wavelength_um=wave, included_mask=included, score_mask=score,
                        observed=observed, physical_model=physical, model=model, continuum=continuum,
                        residual_before=before, smooth_removed=removed, residual_continuum=residual,
                        residual_fixed_flux=residual*continuum, common=residual.mean(axis=1),
                        temporal=residual-residual.mean(axis=1)[:, None],
                        training_template=template, diagnostic_template_remainder=residual-template[:, None])
    adopted = dict(status="provisionally adopted by user", created_utc=result["created_utc"],
                   physical_model="motion_clean_control", filter=FILTER,
                   fit_mask="all included pixels in telluric_mask_motion_clean.npz",
                   coordinate="normalized log wavelength velocity", per_order=True, per_exposure=True,
                   operation=result["definition"], continuum_units="original fitted continuum unchanged",
                   template_operation="Apply the same (I-H) separately to each shifted planetary template",
                   temporal_template_subtraction=False,
                   product="poly5_residual_budget.npz", report="poly5_residual_budget.json",
                   product_sha256=hashlib.sha256((OUT / "poly5_residual_budget.npz").read_bytes()).hexdigest())
    (OUT / "adopted_residual_filter.json").write_text(json.dumps(adopted, indent=2)+"\n")
    labels = [str(order) for order in orders]+["All"]
    parts = [result["per_order"]["score"][str(order)]["continuum"] for order in orders]
    parts.append(result["budgets"]["score"]["continuum"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    x = np.arange(len(labels))
    axes[0].bar(x, [part["total"]["rms_percent"] for part in parts], label="Residual after polynomial 5")
    axes[0].set(ylabel="Original-continuum RMS (%)", xlabel="Order", xticks=x, xticklabels=labels)
    axes[0].legend(fontsize=8)
    common = np.array([100*part["common"]["squared_error_fraction"] for part in parts])
    axes[1].bar(x, common, label="Time-common pattern")
    axes[1].bar(x, 100-common, bottom=common, label="Time-variable residual")
    axes[1].set(ylabel="Share of remaining squared error (%)", xlabel="Order", xticks=x,
                xticklabels=labels, ylim=(0, 120))
    axes[1].legend(fontsize=8, loc="upper center")
    fig.suptitle("After adopted polynomial 5: statistical decomposition, not causal attribution")
    fig.savefig(OUT / "poly5_residual_budget.png", dpi=160)
    plt.close(fig)
    print(json.dumps(dict(budgets=result["budgets"], template_transfer=result["template_transfer"]), indent=2))


if __name__ == "__main__":
    main()
