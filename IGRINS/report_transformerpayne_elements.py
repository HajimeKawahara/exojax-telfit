"""Independently audit and summarize the saved elemental TransformerPayne trial."""

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

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CASES = ["adopted", "initial", "global_stellar", "global_refit",
         "elements_stellar", "elements_refit"]
LABELS = dict(adopted="Adopted PHOENIX + blase", initial="TP initial",
              global_stellar="TP global labels", global_refit="TP global labels + telluric refit",
              elements_stellar="TP independent elements", elements_refit="TP elements + telluric refit")
COLORS = dict(adopted="tab:blue", global_refit="tab:orange", elements_refit="tab:green")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_projection(values, wave, mask):
    """Use QR least squares rather than the fitter's pseudoinverse."""
    result = np.empty_like(values)
    for b, wavelength in enumerate(wave):
        x = 2*(np.log(wavelength)-np.log(wavelength).min())/np.ptp(np.log(wavelength))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        q, r = np.linalg.qr(basis[mask[b]], mode="reduced")
        coefficients = np.linalg.solve(r, q.T@values[b][:, mask[b]].T)
        result[b] = values[b]-(basis@coefficients).T
    return result


def audit(data, metadata):
    """Verify original masks, independent metrics, stage freezes and adoption."""
    manifest_path = OUT/"adopted_telluric_model.json"
    manifest = json.loads(manifest_path.read_text())
    adopted_path = ROOT/manifest["product"]
    if metadata["adopted_product"] != manifest["product"]:
        raise ValueError("The current adoption differs from this experiment's baseline.")
    assert digest(adopted_path) == manifest["product_sha256"]
    with np.load(adopted_path) as saved:
        adopted = {key: saved[key] for key in saved.files}
    with np.load(ROOT/"cache/transformerpayne_fit_context.npz") as saved:
        phoenix = saved["phoenix_stellar"]
    with np.load(OUT/"transformerpayne_stellar_control.npz") as previous:
        prior_star = previous["stellar_only_stellar_only"]
    tied_star_error = float(np.max(np.abs(data["stellar_only_initial"][1:]-prior_star[1:])))
    np.testing.assert_allclose(data["stellar_only_initial"][1:], prior_star[1:], atol=5e-6, rtol=0)
    for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c"]:
        np.testing.assert_array_equal(data[key], adopted[key])
    np.testing.assert_array_equal(data["component_adopted"], adopted["component"])
    np.testing.assert_array_equal(data["stellar_only_adopted"], adopted["stellar_only"])
    np.testing.assert_array_equal(data["exposures"], [0, 19, 39, 59, 78, 10, 30, 50, 70])
    errors = {}
    strong = np.max(1-phoenix[:, :5], axis=1) > .02
    masks = dict(score=data["score_mask"], included=data["included_mask"],
                 excluded_diagnostic=~data["included_mask"], strong_stellar_diagnostic=strong,
                 full_window=np.ones_like(strong))
    for name in CASES:
        component, stellar = data[f"component_{name}"], data[f"stellar_only_{name}"]
        assert np.isfinite(component).all() and component.min() >= 0
        assert np.isfinite(stellar).all() and stellar.min() >= 0
        np.testing.assert_array_equal(component[0], adopted["component"][0])
        np.testing.assert_array_equal(stellar[0], adopted["stellar_only"][0])
        residual = independent_projection(data["y_over_c"]-component,
                                          data["wavelength_um"], data["included_mask"])
        errors[name] = float(np.max(np.abs(residual-data[f"residual_{name}"])))
        np.testing.assert_allclose(residual, data[f"residual_{name}"], atol=1e-11, rtol=0)
        for scope, frames in [("training", slice(0, 5)), ("evaluation", slice(5, None))]:
            for selection, mask in masks.items():
                groups = {"h_orders": [1, 2], "hybrid_all_orders": [0, 1, 2]}
                for group, indices in groups.items():
                    values = np.concatenate([residual[b, frames][:, mask[b]].ravel() for b in indices])
                    reference = metadata["metrics"][name][scope][selection][group]
                    assert values.size == reference["n_samples"]
                    np.testing.assert_allclose(100*np.sqrt(np.mean(values**2)), reference["rms_percent"], atol=1e-10, rtol=0)
                for b, order in enumerate(data["orders"]):
                    values = residual[b, frames][:, mask[b]]
                    reference = metadata["metrics"][name][scope][selection]["per_order"][str(order)]
                    assert values.size == reference["n_samples"]
                    np.testing.assert_allclose(100*np.sqrt(np.mean(values**2)), reference["rms_percent"], atol=1e-10, rtol=0)
    nstar = 6+len(metadata["plan"]["elements"])
    hstrength = adopted["z"][-(data["parameters_global_stellar"].size-nstar):]
    for label in ["global", "elements"]:
        first, final = data[f"parameters_{label}_stellar"], data[f"parameters_{label}_refit"]
        np.testing.assert_array_equal(first[nstar:], hstrength)
        np.testing.assert_array_equal(first[:nstar], final[:nstar])
        for values in [first, final]:
            assert np.all(values >= data["parameter_lower"]-1e-10)
            assert np.all(values <= data["parameter_upper"]+1e-10)
        if label == "global":
            np.testing.assert_array_equal(first[6:nstar], np.full(nstar-6, first[2]))
    for b in [1, 2]:
        np.testing.assert_array_equal(data["calibration_mask"][b],
                                      np.min(data["calibration_telluric"][b, :5], axis=0) > .95)
    for relative, expected in metadata["source_sha256"].items():
        if digest(ROOT/relative) != expected:
            raise ValueError(f"Source hash changed: {relative}")
    assert all(row["success"] for row in metadata["fits"].values())
    return dict(created_utc=datetime.now(timezone.utc).isoformat(), passed=True,
                independent_q5_max_errors=errors, all_metric_groups_recomputed=True,
                original_data_masks_frames_preserved=True, order13_exact_adopted=True,
                stage1_telluric_strengths_fixed=True, stage2_stellar_parameters_fixed=True,
                global_control_abundances_tied=True, parameter_bounds_satisfied=True,
                tied_initial_h_stellar_prediction_max_error=tied_star_error,
                h_calibration_masks_recomputed=True,
                adopted_product=manifest["product"], adopted_product_sha256=digest(adopted_path),
                source_hashes_verified=True)


def shade_excluded(ax, nu, mask):
    order = np.argsort(nu)
    x, keep = nu[order], mask[order]
    edges = np.r_[x[0]-(x[1]-x[0])/2, (x[1:]+x[:-1])/2, x[-1]+(x[-1]-x[-2])/2]
    boundaries = np.flatnonzero(np.diff(np.r_[False, ~keep, False]))
    for left, right in boundaries.reshape(-1, 2):
        ax.axvspan(edges[left], edges[right], color=".8", alpha=.3, lw=0)


def plots(data, metadata):
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex="col")
    for column, b in enumerate([1, 2]):
        nu = 1e4/data["wavelength_um"][b]
        order = np.argsort(nu)
        x = nu[order]
        star_ax, spectrum_ax, residual_ax = axes[:, column]
        observed = np.mean(data["y_over_c"][b, 5:], axis=0)
        spectrum_ax.plot(x, observed[order], color=".15", lw=.75, label="Observed / original continuum")
        for case, color in COLORS.items():
            stellar = np.mean(data[f"stellar_only_{case}"][b, 5:], axis=0)
            residual = np.mean(data[f"residual_{case}"][b, 5:], axis=0)
            star_ax.plot(x, stellar[order], color=color, lw=1., label=LABELS[case])
            spectrum_ax.plot(x, (observed-residual)[order], color=color, lw=.8, label=LABELS[case])
            residual_ax.plot(x, 100*np.where(data["included_mask"][b], residual, np.nan)[order],
                             color=color, lw=.8, label=LABELS[case])
        star_ax.axhline(1, color=".5", lw=.6)
        residual_ax.axhline(0, color=".5", lw=.6)
        for ax in axes[:, column]:
            shade_excluded(ax, nu, data["included_mask"][b])
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.set_xlim(x.min(), x.max())
            ax.grid(alpha=.15)
        calibration = data["calibration_mask"][b][order]
        star_ax.plot(x[calibration], np.full(calibration.sum(), .03), "|", color="teal", ms=3,
                     transform=star_ax.get_xaxis_transform(), label="Stellar calibration pixels")
        score = data["score_mask"][b][order]
        residual_ax.plot(x[score], np.full(score.sum(), .035), "|", color="k", ms=3,
                         transform=residual_ax.get_xaxis_transform(), label="Original score pixels")
        scores = [metadata["metrics"][case]["evaluation"]["score"]["per_order"][str(data["orders"][b])]["rms_percent"]
                  for case in COLORS]
        star_ax.set_title(f"Order {data['orders'][b]}: score RMS " + " / ".join(f"{value:.3f}%" for value in scores))
        star_ax.set_ylabel("Stellar-only flux")
        spectrum_ax.set_ylabel("Continuum-scaled flux")
        residual_ax.set_ylabel("Mean residual (%)")
        residual_ax.set_xlabel(r"Observed wavenumber (cm$^{-1}$)")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8, loc="lower left")
    axes[2, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("TransformerPayne: tied metallicity versus independent elemental abundances\n"
                 "Curves are means of four evaluation frames; RMS uses individual samples. Gray: original exclusions.", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT/"transformerpayne_elements_fit.png", dpi=170)
    plt.close(fig)


def report(data, metadata, validation):
    m = metadata["metrics"]
    elements = metadata["plan"]["elements"]
    checkpoint = json.loads((ROOT/"cache/transformerpayne_checkpoint.json").read_text())
    def hscore(case):
        return m[case]["evaluation"]["score"]["h_orders"]["rms_percent"]
    old, tied, free = [hscore(case) for case in ["adopted", "global_refit", "elements_refit"]]
    lines = ["TransformerPayne independent-element experiment", "",
             f"Fit UTC: {metadata['created_utc']}",
             "Status: local comparison; no automatic adoption or new planet-response validation.", "",
             f"On the unchanged H-order evaluation score, RMS is {old:.6f}% for adopted blase,",
             f"{tied:.6f}% for the TransformerPayne global-metallicity control, and {free:.6f}%",
             "when individual elemental abundances are freed. Both TP cases include a subsequent telluric-strength refit.",
             f"Element freedom changes RMS by {free-tied:+.6f} percentage points relative to the matched TP control",
             f"and by {free-old:+.6f} percentage points relative to the current adopted model.", "",
             "The extra chemical freedom improves TransformerPayne, but it does not beat the provisionally adopted",
             "PHOENIX-plus-blase model on this sample. The adopted model remains in use.", "",
             "Plain explanation", "",
             "The earlier TP fits forced all metals to become richer or poorer together. Here fifteen common NIR",
             "elements can change separately, while all lines of one element respond together through the emulator.",
             "This is more physically linked than independently adjusting every spectral feature. It also cannot",
             "correct arbitrary errors in line positions, line data, the neural model or the assumed stellar atmosphere.",
             "The fitted abundances are effective parameters for this conditional spectral fit, not measurements of",
             "the star's chemical composition. A lower residual alone would not establish their physical validity.", "",
             "Matched protocol", "",
             "Both TP cases fit the same six shared values: Teff, logg, global [M/H], microturbulence, RV offset,",
             "and extra Gaussian width. The control starts from the previous telluric-clear stellar calibration.",
             "Both use the same model-selected training mask. The expanded case releases fifteen elemental abundances",
             "from the control solution; it has the same six global/kinematic bounds as the control.",
             "Elements: "+", ".join(elements)+".",
             "The internal parameters are absolute [X/H] with checkpoint box bounds. The empirical penalty is on",
             "[X/H]-[M/H], with width 0.3*(0.1+0.9*d^2/(d^2+0.005^2)) dex. d is a frozen model-only derivative",
             "on the training calibration pixels; small responses receive tighter regularization. Other metals",
             "remain tied to global [M/H], and helium is fixed. These are shrinkage scales, not chemical error bars.", "",
             "Stage 1 uses exposures 0,19,39,59,78 where current adopted telluric-only transmission exceeds0.95",
             "in all five frames. It profiles Q5 on those calibration pixels with current telluric strengths fixed.",
             f"Calibration pixels: order34={metadata['calibration_counts']['34']}; order40={metadata['calibration_counts']['40']}. K calibration is not used.",
             "Stage 2 freezes each fitted star and refits the same119 H-order telluric strengths on original included",
             "pixels, preserving weak-line priors. All final comparisons use original Q5 and original score masks",
             "on evaluation exposures10,30,50,70. No evaluation flux chooses elements, priors or calibration pixels.",
             "The atmosphere/columns, wavelength solution, original continua, old two-Gaussian IP and rotation4km/s",
             "remain fixed. The public TP checkpoint covers at most2.0micron, so only H orders34/40 change.",
             "K order13, including the4820cm^-1 feature, remains exactly the currently adopted blase prediction.", "",
             "RMS (% of original continuum units)", "",
             "Case | H score | Order34 | Order40 | Hybrid3 score | H included | H strong-stellar | H full-window"]
    for case in CASES:
        e = m[case]["evaluation"]
        values = [e["score"]["h_orders"]["rms_percent"],
                  e["score"]["per_order"]["34"]["rms_percent"], e["score"]["per_order"]["40"]["rms_percent"],
                  e["score"]["hybrid_all_orders"]["rms_percent"], e["included"]["h_orders"]["rms_percent"],
                  e["strong_stellar_diagnostic"]["h_orders"]["rms_percent"], e["full_window"]["h_orders"]["rms_percent"]]
        lines.append(case+" | "+" | ".join(f"{value:.6f}" for value in values))
    h40_mean = data["residual_elements_refit"][2, 5:].mean(axis=0)
    available = np.flatnonzero(data["included_mask"][2])
    worst = available[np.argmax(np.abs(h40_mean[available]))]
    worst_nu = 1e4/data["wavelength_um"][2, worst]
    old_mean = 100*data["residual_adopted"][2, 5:, worst].mean()
    lines += ["", f"A remaining order40 mismatch occurs at {worst_nu:.6f}cm^-1: the four-frame mean residual",
              f"is {100*h40_mean[worst]:+.6f}% with independent elements, versus {old_mean:+.6f}% for adopted blase.",
              f"That pixel is originally included; membership in the original scoring subset is {bool(data['score_mask'][2, worst])}.",
              "This is a diagnostic example, not a new exclusion or a fitted line identification."]
    lines += ["", "H-only is the primary TP comparison; hybrid3 also contains unchanged K pixels.",
              "Initial means the previous calibrated six stellar labels with current adopted telluric strengths,",
              "not the original solar-composition 5600K TransformerPayne trial.",
              "Strong-stellar diagnostics use the original PHOENIX depth>2% definition, not a new fitted mask.",
              "Excluded/full-window diagnostics retain the polynomial defined on original included pixels.", "",
              "Fitted global labels (after the stellar stage; fixed during strength refitting)", "",
              "Parameter | Global control | Independent elements"]
    for name in metadata["fits"]["global_stellar"]["global_labels"]:
        lines.append(f"{name} | {metadata['fits']['global_stellar']['global_labels'][name]:.8g} | "
                     f"{metadata['fits']['elements_stellar']['global_labels'][name]:.8g}")
    lines += ["", "Element | [X/H] dex | [X/H]-[M/H] dex | Prior width dex | Frozen response per dex"]
    efit = metadata["fits"]["elements_stellar"]
    for name in elements:
        lines.append(f"{name} | {efit['elemental_abundances_dex'][name]:.6f} | "
                     f"{efit['elemental_offsets_dex'][name]:+.6f} | "
                     f"{metadata['elemental_prior_width_dex'][name]:.6f} | {metadata['elemental_contrast'][name]:.6g}")
    largest = max(elements, key=lambda name: abs(efit["elemental_offsets_dex"][name])/metadata["elemental_prior_width_dex"][name])
    deviation = efit["elemental_offsets_dex"][largest]
    scale = metadata["elemental_prior_width_dex"][largest]
    lines += ["", "The abundance pattern is extreme and should not be read as measured stellar chemistry.",
              f"The largest offset in prior-width units is {largest}: {deviation:+.6f}dex, or {abs(deviation)/scale:.2f}",
              "assumed prior widths. This is not a statistical significance: the prior is empirical and the",
              "working diagonal noise model does not describe all correlated/model errors. Such large shifts",
              "show that the conditional fit is using abundance freedom to absorb substantial spectral mismatch."]
    lines += ["", "Convergence", ""]
    for name, fit in metadata["fits"].items():
        lines.append(f"{name}: {fit['message']} nfev={fit['nfev']}, optimality={fit['optimality']:.6g}, "
                     f"objective={fit['objective']:.8f}; active free bounds={fit['active_free_bounds']}.")
    lines += ["The solver uses explicitly scaled coordinates for temperatures, velocities and abundances.",
              "Solver success is a stopping flag, not proof of a precise or global optimum. The neural layers use",
              "float32 internally, so small parameter steps and a non-small gradient must be interpreted cautiously.",
              "Elemental directional-derivative checks: "+json.dumps(metadata["elemental_derivative_checks"]), "",
              "Validation", "",
              "Independent QR polynomial projection reproduces every stored residual. All training/evaluation",
              "metric groups and per-order counts were recomputed. Original data/masks/frames, fixed K predictions,",
              "fixed stage1 tellurics, frozen stage2 stellar parameters, bounds, source hashes and the current",
              "adopted product were checked. Maximum Q5 discrepancy: "+f"{max(validation['independent_q5_max_errors'].values()):.3e}.",
              "The adopted product is unchanged. See transformerpayne_elements_audit.json for the check record.", "",
              "Reading the figure", "",
              "transformerpayne_elements_fit.png has one column per supported H order. The top panel shows the",
              "stellar-only prediction. The middle compares data with the physical model plus its profiled Q5",
              "polynomial. The lower panel shows residuals on original included pixels. Gray shading marks the",
              "original exclusions; black ticks mark scoring pixels; teal ticks mark stellar-calibration pixels.",
              "Curves are means of four evaluation frames, while all RMS values use the individual frame samples.", "",
              "Limitations", ""]+metadata["limitations"]
    lines += ["", "Checkpoint", "",
              f"{checkpoint['repository']} / {checkpoint['filename']}",
              f"Revision: {checkpoint['revision']}",
              f"SHA256: {checkpoint['sha256']}",
              f"https://huggingface.co/{checkpoint['repository']}/blob/{checkpoint['revision']}/{checkpoint['filename']}",
              "", "Reproduction from repository root", "",
              "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 JAX_ENABLE_X64=true taskset -c 16-31 IGRINS/.venv-transformerpayne/bin/python IGRINS/fit_transformerpayne_elements.py --max-nfev 120",
              "OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python IGRINS/report_transformerpayne_elements.py",
              "taskset is an optional Linux CPU-affinity setting; adjust it for another machine.",
              "All code, data and generated products remain under Git-ignored IGRINS/.", "",
              "Product SHA256"]
    for name in ["transformerpayne_elements_fit.json", "transformerpayne_elements_fit.npz", "adopted_blase_telluric_fit.npz"]:
        lines.append(f"{name}: {digest(OUT/name)}")
    (OUT/"TRANSFORMERPAYNE_ELEMENTS_REPORT.txt").write_text("\n".join(lines)+"\n")


def main():
    path = OUT/"transformerpayne_elements_fit.npz"
    with np.load(path) as saved:
        data = {key: saved[key] for key in saved.files}
    metadata = json.loads(path.with_suffix(".json").read_text())
    validation = audit(data, metadata)
    validation["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in
                                    [Path(__file__), path, path.with_suffix(".json") ]}
    (OUT/"transformerpayne_elements_audit.json").write_text(json.dumps(validation, indent=2)+"\n")
    plots(data, metadata)
    report(data, metadata, validation)
    print("Elemental report and independent audit complete.")


if __name__ == "__main__":
    main()
