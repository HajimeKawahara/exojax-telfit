"""Optimize shared TransformerPayne stellar labels on the fixed IGRINS sample."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CONTEXT = ROOT / "cache/transformerpayne_fit_context.npz"
WORKING_SIGMA = .0045
STELLAR_NAMES = ["teff_K", "logg", "metallicity_dex", "vmic_kms",
                 "stellar_rv_offset_kms", "stellar_extra_sigma_kms"]
STELLAR_SCALE = np.array([200., .3, .2, 1., 1., 1.])


def export_context():
    """Use the existing ExoJAX environment once to export fixed physics."""
    import pandas as pd
    from fit_line_strength import build_context, predict_components

    context = build_context()
    with np.load(OUT / "adopted_telluric_fit.npz") as saved:
        adopted = {key: saved[key] for key in saved.files}
    initial = adopted["parameters"][adopted["line_parameter_indices"]]
    component = predict_components(context, initial)
    np.testing.assert_allclose(component, adopted["component"], rtol=0, atol=1e-10)
    meta = json.loads((OUT / "motion_clean_control.json").read_text())
    p = meta["parameters"]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    stellar_rv = p["stellar_rv_reference_kms"] + (
        info.iloc[context["exposures"]]["RV [km/s]"].to_numpy() - info.iloc[0]["RV [km/s]"])
    data = dict(orders=context["orders"], exposures=context["exposures"],
                wavelength_um=context["wave"], included_mask=context["allowed"],
                score_mask=context["score"], y_over_c=context["y_over_c"],
                continuum=context["continuum"], adopted_component=component,
                adopted_residual=adopted["residual_filtered"], stellar_rv=stellar_rv,
                stellar_initial=np.array([5600., 4.5, 0., 1., 0., p["stellar_extra_sigma_kms"]]))
    with np.load(OUT / "transformerpayne_comparison.npz") as comparison:
        data["initial_tp_component"] = comparison["component_transformerpayne_fixed"]
        data["previous_tp_refit_component"] = comparison["component_transformerpayne_refit"]
        data["initial_tp_stellar"] = comparison["stellar_only_transformerpayne"]
        data["phoenix_stellar"] = comparison["stellar_only_phoenix"]
    for b, block in enumerate(context["blocks"]):
        if block["order"] not in [34, 40]:
            continue
        order = block["order"]
        for key in ["grid", "velocity", "line_tau", "upper", "baseline_tau", "line_factors",
                    "airmass", "widths", "offsets", "target", "wing_fraction"]:
            data[f"o{order}_{key}"] = np.asarray(block[key])
        sl = block["parameter_slice"]
        data[f"o{order}_strength_initial"] = initial[sl]
        contrast = adopted["line_contrast"][sl]
        data[f"o{order}_prior_widths"] = 1 + 9 * contrast**2 / (contrast**2 + .005**2)
    np.savez_compressed(CONTEXT, **data)
    sources = context["source_files"] + [OUT / "adopted_telluric_fit.npz",
                                         OUT / "transformerpayne_comparison.npz",
                                         ROOT / "fit_line_strength.py"]
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  baseline_max_absolute_error=float(np.max(np.abs(component-adopted["component"]))),
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources},
                  product_sha256=hashlib.sha256(CONTEXT.read_bytes()).hexdigest())
    CONTEXT.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print("Exported", CONTEXT, flush=True)


def project(values, wave, included):
    """Apply the unchanged per-order degree-five projection."""
    result = np.empty_like(values)
    for b, wavelength in enumerate(wave):
        x = 2*(np.log(wavelength)-np.log(wavelength).min())/np.ptp(np.log(wavelength))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        inverse = np.linalg.pinv(basis[included[b]])
        result[b] = values[b] - (values[b, :, included[b]].T @ inverse.T) @ basis.T
    return result


def statistics(data, residual):
    """Keep the same scoring subset separate from broader diagnostics."""
    result = {}
    for scope, rows in [("training", slice(0, 5)), ("evaluation", slice(5, None))]:
        result[scope] = {}
        strong = np.max(1-data["phoenix_stellar"][:, :5], axis=1) > .02
        for name, mask in [("score", data["score_mask"]), ("included", data["included_mask"]),
                           ("excluded_diagnostic", ~data["included_mask"]),
                           ("strong_stellar_diagnostic", strong),
                           ("full_window", np.ones_like(data["included_mask"]))]:
            def metric(indices):
                values = np.concatenate([residual[b, rows][:, mask[b]].ravel() for b in indices])
                return dict(n_samples=values.size, rms_percent=float(100*np.sqrt(np.mean(values**2))))
            result[scope][name] = dict(h_orders=metric([1, 2]), hybrid_all_orders=metric([0, 1, 2]),
                                      per_order={str(int(order)): metric([b]) for b, order in enumerate(data["orders"])})
    return result


class Problem:
    """Chain four neural-label derivatives through the physical forward model."""

    def __init__(self, data):
        import jax
        import jax.numpy as jnp
        from exojax.postproc.response import ipgauss, sampling
        from exojax.postproc.spin_rotation import convolve_rigid_rotation
        from transformerpayne_labels import StellarEmulator

        self.data = data
        self.emulator = StellarEmulator({order: data[f"o{order}_grid"] for order in [34, 40]})
        self.blocks = []
        cursor = 6
        for b, order in [(1, 34), (2, 40)]:
            keys = ["grid", "velocity", "line_tau", "upper", "baseline_tau", "line_factors",
                    "airmass", "widths", "offsets", "target", "wing_fraction"]
            a = {key: jnp.asarray(data[f"o{order}_{key}"]) for key in keys}
            nline = len(a["upper"])
            sl = slice(cursor, cursor+nline)
            cursor += nline
            rv = jnp.asarray(data["stellar_rv"])
            included = data["included_mask"][b]
            x = 2*(np.log(data["wavelength_um"][b])-np.log(data["wavelength_um"][b]).min())/np.ptp(np.log(data["wavelength_um"][b]))-1
            basis = jnp.asarray(np.polynomial.chebyshev.chebvander(x, 5))
            inverse = jnp.asarray(np.linalg.pinv(np.asarray(basis)[included]))

            def make_functions(a, rv, basis, inverse, included):
                def star(raw, kin):
                    rotation = convolve_rigid_rotation(raw, a["velocity"], 4., .6, 0.)
                    broadened = ipgauss(rotation, a["velocity"], kin[1])
                    return jax.vmap(lambda v: sampling(a["grid"], a["grid"], broadened, v+kin[0]))(rv)

                def response(intrinsic):
                    def one(row, width, offset):
                        core = sampling(a["target"], a["grid"], ipgauss(row, a["velocity"], width), offset)
                        wing = sampling(a["target"], a["grid"], ipgauss(row, a["velocity"], 2*width), offset)
                        return (1-a["wing_fraction"])*core + a["wing_fraction"]*wing
                    return jax.vmap(one)(intrinsic, a["widths"], a["offsets"])

                def component(raw, kin, z):
                    delta = (a["line_factors"]*(a["upper"]*z)[None, :]) @ a["line_tau"]
                    transmission = jnp.exp(-a["airmass"][:, None]*(a["baseline_tau"]+delta))
                    return response(star(raw, kin)*transmission)

                def filtered(component):
                    return component-(component[:, included] @ inverse.T) @ basis.T

                def train_model(raw, kin, z):
                    return filtered(component(raw, kin, z))[:5, included].ravel()/WORKING_SIGMA

                # Model is linear in raw stellar flux, so this also propagates label tangents.
                label_tangent = jax.jit(jax.vmap(train_model, in_axes=(1, None, None), out_axes=1))
                return (jax.jit(component), jax.jit(train_model), label_tangent,
                        jax.jit(jax.jacfwd(train_model, argnums=1)),
                        jax.jit(jax.jacfwd(train_model, argnums=2)),
                        jax.jit(lambda raw, kin: response(star(raw, kin))))

            fns = make_functions(a, rv, basis, inverse, included)
            target = data["y_over_c"][b]
            target = target - (target[:, included] @ np.asarray(inverse).T) @ np.asarray(basis).T
            self.blocks.append(dict(order=order, index=b, sl=sl, a=a, component=fns[0],
                                    train_model=fns[1], label_tangent=fns[2], kin_jacobian=fns[3],
                                    strength_jacobian=fns[4], star_only=fns[5],
                                    target=target[:5, included].ravel()/WORKING_SIGMA,
                                    prior=data[f"o{order}_prior_widths"]))
        self.initial = np.r_[data["stellar_initial"], data["o34_strength_initial"], data["o40_strength_initial"]]
        self.lower = np.r_[self.emulator.lower, -5., .3, -1/data["o34_upper"], -1/data["o40_upper"]]
        self.upper = np.r_[self.emulator.upper, 5., 10., np.full(len(self.initial)-6, np.inf)]
        self.scale = np.r_[STELLAR_SCALE, np.ones(len(self.initial)-6)]
        self._label_cache = None
        self._flux = None
        self._dflux = None
        self.history = []
        self.started = time.monotonic()

    def stellar(self, labels, derivative=False):
        if self._label_cache is None or not np.array_equal(labels, self._label_cache):
            # Keep the AD-compiled primal identical in residuals and Jacobians.
            self._flux, self._dflux = self.emulator.evaluate(labels, with_jacobian=True)
            self._label_cache = np.array(labels, copy=True)
        return self._flux, self._dflux

    def residual(self, theta):
        flux, _ = self.stellar(theta[:4])
        rows = []
        for b in self.blocks:
            rows.append(b["target"]-np.asarray(b["train_model"](flux[b["order"]], theta[4:6], theta[b["sl"]])))
        rows.extend(theta[b["sl"]]/b["prior"] for b in self.blocks)
        value = np.concatenate(rows)
        self.history.append(dict(seconds=time.monotonic()-self.started, stellar=theta[:6].tolist(),
                                 objective=float(value@value)))
        if len(self.history) % 5 == 1:
            print("Evaluation", len(self.history), "objective", float(value@value), "stellar", theta[:6].tolist(), flush=True)
        return value

    def jacobian(self, theta):
        flux, dflux = self.stellar(theta[:4], derivative=True)
        rows = []
        for b in self.blocks:
            raw, draw, z = flux[b["order"]], dflux[b["order"]], theta[b["sl"]]
            row = np.zeros((len(b["target"]), len(theta)))
            row[:, :4] = -np.asarray(b["label_tangent"](draw, theta[4:6], z))
            row[:, 4:6] = -np.asarray(b["kin_jacobian"](raw, theta[4:6], z))
            row[:, b["sl"]] = -np.asarray(b["strength_jacobian"](raw, theta[4:6], z))
            rows.append(row)
        prior = np.zeros((len(theta)-6, len(theta)))
        prior[:, 6:] = np.diag(1/np.concatenate([b["prior"] for b in self.blocks]))
        return np.vstack(rows+[prior])

    def predictions(self, theta):
        flux, _ = self.stellar(theta[:4])
        component = self.data["adopted_component"].copy()
        stellar = self.data["phoenix_stellar"].copy()
        for b in self.blocks:
            component[b["index"]] = np.asarray(b["component"](flux[b["order"]], theta[4:6], theta[b["sl"]]))
            stellar[b["index"]] = np.asarray(b["star_only"](flux[b["order"]], theta[4:6]))
        return component, stellar


def fit_case(problem, initial, free, max_nfev, name):
    """Fit a fixed parameter subset using training data only."""
    from scipy.optimize import least_squares

    template = np.array(initial, copy=True)
    def expand(x):
        theta = template.copy()
        theta[free] = x
        return theta
    fit = least_squares(lambda x: problem.residual(expand(x)),
                        np.clip(template[free], problem.lower[free]+1e-9, problem.upper[free]-1e-9),
                        jac=lambda x: problem.jacobian(expand(x))[:, free],
                        bounds=(problem.lower[free], problem.upper[free]),
                        x_scale=problem.scale[free], method="trf", ftol=1e-7,
                        xtol=1e-6, gtol=1e-5, max_nfev=max_nfev)
    theta = expand(fit.x)
    report = dict(name=name, success=bool(fit.success), status=int(fit.status), message=fit.message,
                  nfev=fit.nfev, njev=fit.njev, optimality=float(fit.optimality), objective=float(2*fit.cost),
                  stellar=dict(zip(STELLAR_NAMES, theta[:6].tolist())),
                  active_bounds=np.flatnonzero(np.isclose(theta, problem.lower, rtol=0, atol=1e-5)
                                               | np.isclose(theta, problem.upper, rtol=0, atol=1e-5)).tolist(),
                  free_parameters=int(free.sum()), parameters=theta.tolist())
    print("Fit completed", json.dumps(report), flush=True)
    return theta, report


def plot_result(data, components, residuals, stars):
    """Show the fit and stellar changes, retaining every original exclusion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    frame = int(np.flatnonzero(data["exposures"] == 30)[0])
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), gridspec_kw={"height_ratios": [2, 1, 2, 1]})
    colors = dict(phoenix="#2166ac", initial="#999999", stellar_only="#e08214", joint="#b2182b")
    for pair, b in enumerate([1, 2]):
        x = 1e4/data["wavelength_um"][b]
        ax, rx = axes[2*pair:2*pair+2]
        excluded = ~data["included_mask"][b]
        edges = np.r_[x[0]-(x[1]-x[0])/2, (x[1:]+x[:-1])/2, x[-1]+(x[-1]-x[-2])/2]
        starts = np.flatnonzero(np.diff(np.r_[False, excluded, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, excluded, False].astype(int)) == -1)
        for axis in [ax, rx]:
            for lo, hi in zip(starts, ends):
                axis.axvspan(edges[lo], edges[hi], color="#eeeeee", zorder=0)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.set_xlim(x.min(), x.max())
            axis.grid(axis="y", alpha=.2)
        ax.plot(x, data["y_over_c"][b, frame], color="black", lw=.6, label="Observed")
        for name, label in [("phoenix", "PHOENIX adopted"), ("joint", "TransformerPayne optimized")]:
            fitted = data["y_over_c"]-residuals[name]
            ax.plot(x, fitted[b, frame], color=colors[name], lw=.9, label=label)
            ax.plot(x, stars[name][b, frame], color=colors[name], lw=1, ls="--", label=label+" star alone")
        for name, label in [("phoenix", "PHOENIX"), ("initial", "TP before optimization"),
                            ("stellar_only", "TP stellar fit"), ("joint", "TP joint fit")]:
            rx.plot(x, 100*np.where(data["included_mask"][b], residuals[name][b, frame], np.nan),
                    color=colors[name], lw=.7, label=label)
        ax.set(ylabel="Flux / original continuum", title=f"Order {data['orders'][b]}: evaluation frame 30", ylim=(-.05, 1.15))
        rx.set(ylabel="Residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)")
        rx.axhline(0, color="0.4", lw=.6)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles+[Patch(facecolor="#eeeeee")], labels+["Existing exclusions"], fontsize=8, ncol=2, loc="lower left")
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("TransformerPayne stellar-parameter optimization\nSame training frames, evaluation pixels, original IP and degree-five filtering", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94))
    fig.savefig(OUT / "transformerpayne_label_fit.png", dpi=170)
    plt.close(fig)


def main(args):
    metadata = json.loads(CONTEXT.with_suffix(".json").read_text())
    assert hashlib.sha256(CONTEXT.read_bytes()).hexdigest() == metadata["product_sha256"]
    with np.load(CONTEXT) as saved:
        data = {key: saved[key] for key in saved.files}
    problem = Problem(data)
    components, stars, parameters, fits = {}, {}, {}, {}
    components["phoenix"], stars["phoenix"] = data["adopted_component"], data["phoenix_stellar"]
    components["initial"], stars["initial"] = problem.predictions(problem.initial)
    baseline_error = float(np.max(np.abs(components["initial"]-data["initial_tp_component"])))
    if baseline_error > 5e-6:
        raise ValueError(f"TP initial reconstruction differs by {baseline_error}")
    parameters["initial"] = problem.initial
    # Check a scale-aware direction with finite steps suitable for float32 NN layers.
    rng = np.random.default_rng(71)
    direction = rng.normal(size=len(problem.initial))*problem.scale
    direction[6:] *= .1
    analytic = problem.jacobian(problem.initial) @ direction
    checks = []
    for step in [.01, .03]:
        plus = problem.residual(problem.initial+step*direction)
        minus = problem.residual(problem.initial-step*direction)
        numerical = (plus-minus)/(2*step)
        checks.append(dict(step=step, relative_l2=float(np.linalg.norm(numerical-analytic)/np.linalg.norm(analytic))))
    if min(row["relative_l2"] for row in checks) > .03:
        raise ValueError(f"Directional Jacobian check failed: {checks}")
    print("Initial checks", baseline_error, checks, flush=True)
    if args.check_only:
        return
    free = np.zeros(len(problem.initial), dtype=bool)
    free[:6] = True
    parameters["stellar_only"], fits["stellar_only"] = fit_case(problem, problem.initial, free, args.max_nfev, "stellar_only")
    np.savez_compressed(OUT / "transformerpayne_label_checkpoint.npz", parameters=parameters["stellar_only"])
    parameters["joint"], fits["joint"] = fit_case(problem, parameters["stellar_only"], np.ones_like(free), args.max_nfev, "joint")
    selected_joint_fit = "joint"
    if args.restart:
        alternate, report = fit_case(problem, problem.initial, np.ones_like(free), args.max_nfev, "joint_initial_restart")
        fits["joint_initial_restart"] = report
        parameters["joint_initial_restart"] = alternate
        if report["objective"] < fits["joint"]["objective"]:
            parameters["joint"] = alternate
            selected_joint_fit = "joint_initial_restart"
    for name in ["stellar_only", "joint"]:
        components[name], stars[name] = problem.predictions(parameters[name])
    residuals = {name: project(data["y_over_c"]-model, data["wavelength_um"], data["included_mask"])
                 for name, model in components.items()}
    np.testing.assert_allclose(residuals["phoenix"], data["adopted_residual"], atol=1e-10, rtol=0)
    for component in components.values():
        np.testing.assert_array_equal(component[0], data["adopted_component"][0])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
                  training_frames=data["exposures"][:5].tolist(), evaluation_frames=data["exposures"][5:].tolist(),
                  stellar_parameter_names=STELLAR_NAMES, stellar_lower=problem.lower[:6].tolist(),
                  stellar_upper=problem.upper[:6].tolist(), baseline_reconstruction_error=baseline_error,
                  directional_jacobian_checks=checks, fits=fits, selected_joint_fit=selected_joint_fit,
                  metrics={name: statistics(data, residual) for name, residual in residuals.items()},
                  history=problem.history, wall_seconds=time.monotonic()-problem.started,
                  caveats=["Only orders34/40 use TransformerPayne; order13 remains PHOENIX outside the public emulator wavelength range.",
                           "All stellar parameters and 119 H-order line strengths are shared across five training frames; evaluation flux never enters this optimization.",
                           "Masks, atmospheric columns/TP, IP, rotation4km/s, original continua and wavelength calibrations remain fixed; smooth Q5 coefficients are profiled.",
                           "Global metallicity shifts every metal abundance together relative to checkpoint solar defaults, leaving helium fixed.",
                           "No stellar priors were imposed beyond bounds. These conditional effective labels are not a calibrated stellar retrieval.",
                           "The original mask suppresses strong stellar features and was selected using PHOENIX/earlier residuals; label constraints can be weak.",
                           "Strong-stellar diagnostics use fixed PHOENIX-only depth >2% in any training frame; excluded/full-window diagnostics do not change the fit mask.",
                           "NN layers use float32; fit tolerances and finite-difference checks account for numerical granularity.",
                           "No planet injection test or model adoption is performed."])
    sources = [Path(__file__), ROOT/"transformerpayne_labels.py", CONTEXT, CONTEXT.with_suffix(".json"),
               ROOT/"cache/transformerpayne_checkpoint.json", OUT/"adopted_telluric_fit.npz"]
    result["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT / "transformerpayne_label_fit.json").write_text(json.dumps(result, indent=2) + "\n")
    arrays = {key: data[key] for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c"]}
    for name in components:
        arrays.update({f"component_{name}": components[name], f"residual_{name}": residuals[name], f"stellar_only_{name}": stars[name]})
    arrays.update({f"parameters_{name}": value for name, value in parameters.items()})
    np.savez_compressed(OUT / "transformerpayne_label_fit.npz", **arrays)
    plot_result(data, components, residuals, stars)
    lines = ["TransformerPayne stellar-label optimization", "",
             "RMS is percent of the original fitted continuum; identical four evaluation frames and fixed score pixels.",
             "No adoption. Only H orders34/40 are changed; K order13 remains PHOENIX.", "",
             "Case | H score RMS | Order34 | Order40 | Hybrid three-order RMS"]
    for name, value in result["metrics"].items():
        entry = value["evaluation"]["score"]
        lines.append(f"{name} | {entry['h_orders']['rms_percent']:.8f} | {entry['per_order']['34']['rms_percent']:.8f} | {entry['per_order']['40']['rms_percent']:.8f} | {entry['hybrid_all_orders']['rms_percent']:.8f}")
    lines += ["", "Fits:"]
    for name, value in fits.items():
        lines.append(f"{name}: {value['stellar']}; success={value['success']}; active bounds={value['active_bounds']}; objective={value['objective']:.8f}")
    lines += ["", "Interpretation limits:"] + result["caveats"]
    (OUT / "TRANSFORMERPAYNE_LABEL_FIT_REPORT.txt").write_text("\n".join(lines) + "\n")
    print("Finished label optimization", result["wall_seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--max-nfev", type=int, default=100)
    args = parser.parse_args()
    if args.export:
        export_context()
    else:
        main(args)
