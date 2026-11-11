"""Compare cached TransformerPayne and PHOENIX stars with fixed telluric masks."""

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

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from fit_line_strength import (
    WORKING_SIGMA, build_context, make_forward, predict_components,
    prepare_fit_callbacks,
)
from plot_premask_report import shade_excluded

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CACHE = ROOT / "cache/stellar_transformerpayne.npz"


def metric(residual, mask, rows, order_indices):
    """Pool equal-weight samples in original continuum units."""
    values = np.concatenate([residual[b, rows][:, mask[b]].ravel()
                             for b in order_indices])
    return dict(n_samples=int(values.size), rms_percent=float(100*np.sqrt(np.mean(values**2))))


def summarize(context, residual):
    """Keep H-band scoring separate from unchanged K-band and full windows."""
    all_orders = list(range(len(context["orders"])))
    h_orders = [b for b, o in enumerate(context["orders"]) if int(o) in [34, 40]]
    masks = dict(score=context["score"], included=context["allowed"],
                 full_window=np.ones_like(context["allowed"], dtype=bool))
    result = {}
    for scope, rows in [("training", slice(0, 5)), ("evaluation", slice(5, None))]:
        result[scope] = {}
        for name, mask in masks.items():
            result[scope][name] = dict(
                h_orders=metric(residual, mask, rows, h_orders),
                hybrid_all_orders=metric(residual, mask, rows, all_orders),
                per_order={str(int(context["orders"][b])): metric(residual, mask, rows, [b])
                           for b in all_orders})
    return result


def star_only(block):
    """Instrumentally broaden the stellar counterfactual with no atmosphere."""
    result = []
    for star, width, offset in zip(block["shifted_star"], block["widths"], block["offsets"]):
        core = sampling(block["target"], block["grid"],
                        ipgauss(star, block["velocity"], width), offset)
        wing = sampling(block["target"], block["grid"],
                        ipgauss(star, block["velocity"], 2*width), offset)
        result.append(np.asarray((1-block["wing_fraction"])*core+block["wing_fraction"]*wing))
    return np.array(result)


def replace_star(context, cached):
    """Use the new intrinsic spectrum before rotation, transmission and IP."""
    meta = json.loads((OUT / "motion_clean_control.json").read_text())
    parameters = meta["parameters"]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    rv = (info.iloc[context["exposures"]]["RV [km/s]"].to_numpy()
          - info.iloc[0]["RV [km/s]"] + parameters["stellar_rv_reference_kms"])
    output = dict(context, blocks=[])
    for b, original in enumerate(context["blocks"]):
        block = dict(original)
        order = block["order"]
        if order in [34, 40]:
            np.testing.assert_allclose(cached[f"grid_o{order}"], block["grid"], rtol=0, atol=1e-10)
            raw = np.asarray(cached[f"flux_o{order}"], dtype=float)
            assert raw.shape == block["grid"].shape and np.all(np.isfinite(raw))
            assert np.all(raw > 0)
            star = convolve_rigid_rotation(jnp.asarray(raw), block["velocity"], 4., .6, 0.)
            star = ipgauss(star, block["velocity"], parameters["stellar_extra_sigma_kms"])
            block["shifted_star"] = np.array([
                sampling(block["grid"], block["grid"], star, velocity) for velocity in rv])
            block["train_forward"], block["forward"] = make_forward(block)
            logwave = np.log(context["wave"][b])
            x = 2*(logwave-logwave.min())/np.ptp(logwave)-1
            basis = np.polynomial.chebyshev.chebvander(x, 5)
            inverse = np.linalg.pinv(basis[context["allowed"][b]])
            prepare_fit_callbacks(block, context["allowed"][b], basis, inverse)
        output["blocks"].append(block)
    return output


def refit_strengths(context, initial, contrasts):
    """Refit H-band intensities only, keeping adopted weak-line prior widths."""
    theta = initial.copy()
    diagnostics = []
    for b, block in enumerate(context["blocks"]):
        if block["order"] not in [34, 40]:
            continue
        sl = block["parameter_slice"]
        d = contrasts[sl]
        widths = 1+9*d*d/(d*d+.005**2)
        lower = -1/block["upper"]
        start = np.maximum(theta[sl], lower+1e-10)
        target = jnp.asarray(context["y_over_c"][b, :5])

        def fun(z):
            return np.r_[np.asarray(block["data_residual"](z, target)), z/widths]

        def jac(z):
            return np.vstack([np.asarray(block["data_jacobian"](z, target)), np.diag(1/widths)])

        fit = least_squares(fun, start, jac=jac, bounds=(lower, np.inf), method="trf",
                            x_scale="jac", ftol=1e-10, xtol=1e-10, gtol=1e-8, max_nfev=300)
        if not fit.success:
            raise RuntimeError(f"Order {block['order']} strength fit failed: {fit.message}")
        theta[sl] = fit.x
        diagnostics.append(dict(order=block["order"], success=bool(fit.success),
                                nfev=int(fit.nfev), optimality=float(fit.optimality),
                                data_objective=float(np.sum(fun(fit.x)[:-len(fit.x)]**2)),
                                prior_objective=float(np.sum((fit.x/widths)**2)),
                                zero_intensity_count=int(np.sum(1+block["upper"]*fit.x < 1e-5))))
        print("Refitted", block["order"], diagnostics[-1], flush=True)
    return theta, diagnostics


def plot_comparison(context, components, residuals, stellar):
    """Show observed spectra, both stellar models, and conditional residuals."""
    frame = int(np.flatnonzero(context["exposures"] == 30)[0])
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    for pair, b in enumerate([1, 2]):
        x = 1e4/context["wave"][b]
        ax, rx = axes[2*pair:2*pair+2]
        observed = context["y_over_c"][b, frame]
        ax.plot(x, observed, color="black", lw=.6, label="Observed")
        for name, color, label in [("phoenix", "#2166ac", "PHOENIX total"),
                                   ("transformerpayne_refit", "#b2182b", "TransformerPayne total + strength refit")]:
            smooth = context["y_over_c"]-components[name]-residuals[name]
            ax.plot(x, components[name][b, frame]+smooth[b, frame], color=color, lw=.9, label=label)
        ax.plot(x, stellar["phoenix"][b, frame], "--", color="#2166ac", lw=1, label="PHOENIX star alone")
        ax.plot(x, stellar["transformerpayne"][b, frame], "--", color="#238b45", lw=1,
                label="TransformerPayne star alone")
        for name, color, label in [("phoenix", "#2166ac", "PHOENIX"),
                                   ("transformerpayne_fixed", "#e08214", "TransformerPayne: fixed strengths"),
                                   ("transformerpayne_refit", "#b2182b", "TransformerPayne: refitted strengths")]:
            rx.plot(x, 100*np.where(context["allowed"][b], residuals[name][b, frame], np.nan),
                    color=color, lw=.7, label=label)
        ax.set(ylabel="Flux / original continuum", title=f"Order {context['orders'][b]}: evaluation frame 30", ylim=(-.05, 1.15))
        rx.set(ylabel="Residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)")
        rx.axhline(0, color="0.4", lw=.6)
        for axis in [ax, rx]:
            shade_excluded(axis, x, context["allowed"][b])
            axis.set_xlim(x.min(), x.max())
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y", alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(facecolor="#ececec"))
    labels.append("Existing exclusions (unchanged)")
    axes[0].legend(handles, labels, fontsize=8, ncol=2, loc="lower left")
    axes[1].legend(fontsize=8, ncol=3)
    fig.suptitle("TransformerPayne trial: fixed masks, atmosphere, IP and stellar broadening\n"
                 "H-band only; degree-five residual projection retained; no model adoption", fontsize=13)
    fig.text(.5, .01, "Residual traces show all included pixels; reported comparison RMS uses the fixed scoring subset.",
             ha="center", fontsize=10)
    fig.tight_layout(rect=(0, .03, 1, .94))
    fig.savefig(OUT / "transformerpayne_comparison.png", dpi=170)
    plt.close(fig)


def main():
    started = time.monotonic()
    with np.load(CACHE) as data:
        cached = {key: data[key] for key in data.files}
    np.testing.assert_array_equal(cached["orders"], [34, 40])
    with np.load(OUT / "adopted_telluric_fit.npz") as data:
        adopted = {key: data[key] for key in data.files}
    initial = adopted["parameters"][adopted["line_parameter_indices"]]
    context = build_context()
    for key, saved_key in [("orders", "orders"), ("exposures", "exposures"),
                           ("wave", "wavelength_um"), ("allowed", "included_mask"),
                           ("score", "score_mask"), ("y_over_c", "y_over_c")]:
        np.testing.assert_array_equal(context[key], adopted[saved_key])
    components = {"phoenix": predict_components(context, initial)}
    error = float(np.max(np.abs(components["phoenix"]-adopted["component"])))
    np.testing.assert_allclose(components["phoenix"], adopted["component"], rtol=0, atol=1e-10)
    stellar = {"phoenix": np.array([star_only(block) for block in context["blocks"]])}
    alternative = replace_star(context, cached)
    stellar["transformerpayne"] = np.array([star_only(block) for block in alternative["blocks"]])
    components["transformerpayne_fixed"] = predict_components(alternative, initial)
    fixed_residual = context["project"](context["y_over_c"]-components["transformerpayne_fixed"])
    print("Fixed TP metrics", summarize(context, fixed_residual)["evaluation"]["score"], flush=True)
    theta, fits = refit_strengths(alternative, initial, adopted["line_contrast"])
    components["transformerpayne_refit"] = predict_components(alternative, theta)
    residuals = {name: context["project"](context["y_over_c"]-component)
                 for name, component in components.items()}
    np.testing.assert_allclose(residuals["phoenix"], adopted["residual_filtered"], rtol=0, atol=1e-10)
    for name in components:
        np.testing.assert_array_equal(components[name][0], components["phoenix"][0])
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
        supported_orders=[34, 40], unchanged_phoenix_order=13,
        comparison="Same Teff/logg, nominal solar composition, broadening and saved physical nuisance fits; replace intrinsic stellar flux before telluric multiplication and instrumental convolution",
        stellar_labels="Teff=5600 K, logg=4.5, solar abundances, vmic=1 km/s for TransformerPayne",
        strength_fit="Only H-order strengths refitted on five training frames, with positivity and the original adopted contrast-dependent priors frozen",
        working_residual_scale=WORKING_SIGMA, original_baseline_reconstruction_max_error=error,
        metrics={name: summarize(context, residual) for name, residual in residuals.items()},
        strength_fit_diagnostics=fits, seconds=time.monotonic()-started,
        caveats=["The public emulator ends at 20000 Angstrom; the K-band order is not extrapolated.",
                 "This changes atmosphere synthesis, line list, solar reference mixture, continuum definition and emulation together; it does not isolate neural-network architecture.",
                 "The current masks were defined using PHOENIX and earlier residuals and remain fixed in every case.",
                 "Full-window RMS includes excluded pixels and is an exploratory diagnostic, not a newly validated scoring set.",
                 "Original continuum and calibration nuisance fits remain fixed; degree-five corrections are analytically reprofiled for each prediction.",
                 "Stellar labels, rotation, extra broadening and velocity are not optimized for TransformerPayne.",
                 "No planet-injection test or model adoption is performed."],
    )
    sources = context["source_files"]+[Path(__file__), CACHE, OUT/"adopted_telluric_fit.npz",
                                      ROOT/"fit_line_strength.py", ROOT/"stellar_template.py"]
    if CACHE.with_suffix(".json").exists():
        sources.append(CACHE.with_suffix(".json"))
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT / "transformerpayne_comparison.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "transformerpayne_comparison.npz", orders=context["orders"],
                        exposures=context["exposures"], wavelength_um=context["wave"],
                        included_mask=context["allowed"], score_mask=context["score"],
                        y_over_c=context["y_over_c"], adopted_parameters=initial,
                        refitted_parameters=theta,
                        **{f"component_{k}": v for k, v in components.items()},
                        **{f"residual_{k}": v for k, v in residuals.items()},
                        **{f"stellar_only_{k}": v for k, v in stellar.items()})
    lines = ["TransformerPayne controlled stellar-template trial", "",
             "The public v2 intensity emulator can be used for this H-band trial, but the fixed-label substitution worsens the residual.",
             "Refitting telluric strengths recovers part of the loss but does not outperform the adopted PHOENIX model.",
             "Strong new stellar absorption predicted near 6667 and 6686 cm^-1 contributes to the deterioration in order40.",
             "This is not an optimized TransformerPayne stellar retrieval. The adopted model remains unchanged.",
             "Model specification: https://github.com/RozanskiT/transformer_payne", "",
             "All RMS values are percentages of the original fitted continuum, after degree-five projection.",
             "Same four evaluation frames and unchanged masks; order13 remains PHOENIX.", "",
             "Case | H score RMS | Order34 score RMS | Order40 score RMS | Hybrid3-order score RMS | H full-window RMS"]
    for name, entry in result["metrics"].items():
        values = entry["evaluation"]; score = values["score"]
        lines.append(f"{name} | {score['h_orders']['rms_percent']:.8f} | "
                     f"{score['per_order']['34']['rms_percent']:.8f} | {score['per_order']['40']['rms_percent']:.8f} | "
                     f"{score['hybrid_all_orders']['rms_percent']:.8f} | {values['full_window']['h_orders']['rms_percent']:.8f}")
    lines += ["", "Interpretation limits:"]+result["caveats"]
    (OUT / "TRANSFORMERPAYNE_COMPARISON_REPORT.txt").write_text("\n".join(lines)+"\n")
    plot_comparison(context, components, residuals, stellar)
    print("Completed", result["seconds"], "seconds", flush=True)


if __name__ == "__main__":
    main()
