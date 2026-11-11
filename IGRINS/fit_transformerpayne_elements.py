"""Compare tied metallicity with independent elemental TransformerPayne labels.

Run in IGRINS/.venv-transformerpayne. The current adopted blase product supplies
the fixed K order and the initial telluric strengths. No adopted file is changed.
"""

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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from fit_transformerpayne_labels import (
    CONTEXT, Problem as PhysicalProblem, STELLAR_NAMES, STELLAR_SCALE,
    WORKING_SIGMA, project, statistics,
)
from transformerpayne_labels import StellarEmulator

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
ELEMENTS = ["C", "N", "O", "Na", "Mg", "Al", "Si", "S", "K", "Ca",
            "Ti", "Cr", "Mn", "Fe", "Ni"]
N_ELEMENT = len(ELEMENTS)
N_STAR = 6+N_ELEMENT
PLAN = dict(
    elements=ELEMENTS,
    control="Six shared global/kinematic labels; all metal abundances tied to global [M/H]",
    expanded="The same six labels plus fifteen independent elemental [X/H] values",
    parameterization="Absolute [X/H] box bounds respect the checkpoint; regularization applies to [X/H]-[M/H]",
    element_prior="Gaussian penalty on [X/H]-[M/H] with s=0.3*(0.1+0.9*d^2/(d^2+0.005^2)) dex",
    contrast="Frozen maximum absolute derivative of the Q5-filtered component per dex on telluric-clear training pixels, at the same initial stellar labels and current adopted telluric strengths",
    element_selection="Fifteen declared common NIR elements; no observed-residual selection. Other metals stay tied to global [M/H]; helium fixed.",
    stellar_training="Minimum current adopted telluric-only transmission over first five frames >0.95; Q5 profiled on that calibration mask",
    strength_stage="Freeze each calibrated star, then refit the existing119 H-order strengths on the original included pixels with unchanged adopted weak-line priors",
    evaluation="Original Q5 and masks; last four frames; K order13 remains current adopted blase",
    optimizer="Bounded trust-region least squares in explicit scaled coordinates; identical physical bounds and objective for both controls",
    training_frames=[0, 19, 39, 59, 78], evaluation_frames=[10, 30, 50, 70],
    primary_prior_scale_dex=.3, adoption=False,
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ElementEmulator(StellarEmulator):
    """Use independent absolute abundances while retaining all checkpoint bounds."""

    def __init__(self, grids):
        super().__init__(grids)
        self.element_indices = np.asarray([self.names.index(name) for name in ELEMENTS])
        low = np.asarray(self.emulator.min_stellar_parameters)-self.solar
        high = np.asarray(self.emulator.max_stellar_parameters)-self.solar
        self.lower = np.r_[StellarEmulator.lower, low[self.element_indices]]
        self.upper = np.r_[StellarEmulator.upper, high[self.element_indices]]

    def parameters(self, labels):
        parameters = StellarEmulator.parameters(self, labels[:4])
        indices = jnp.asarray(self.element_indices)
        return parameters.at[indices].set(jnp.asarray(self.solar)[indices]+labels[4:])

    def evaluate(self, labels, with_jacobian=True):
        labels = np.asarray(labels, dtype=float)
        if labels.shape != (4+N_ELEMENT,) or np.any(labels < self.lower) or np.any(labels > self.upper):
            raise ValueError("Labels exceed checkpoint bounds.")
        p = np.asarray(self.parameters(jnp.asarray(labels)))
        if np.any(p < np.asarray(self.emulator.min_stellar_parameters)) or np.any(p > np.asarray(self.emulator.max_stellar_parameters)):
            raise ValueError("An absolute abundance exceeds checkpoint bounds.")
        if (self._last_labels is not None and np.array_equal(labels, self._last_labels)
                and (not with_jacobian or self._last_result[1] is not None)):
            return self._last_result
        fluxes, jacobians = {}, {} if with_jacobian else None
        for order, logwave in self.logwave.items():
            total = np.zeros((len(logwave), 2))
            gradient = np.zeros((len(logwave), 2, len(labels))) if with_jacobian else None
            for mu, weight in zip(self.mu, self.weights):
                for left in range(0, len(logwave), self.batch_size):
                    piece = logwave[left:left+self.batch_size]
                    padded = np.pad(piece, (0, self.batch_size-len(piece)), mode="edge")
                    if with_jacobian:
                        value, jac = self.intensity_with_jacobian(labels, padded, mu)
                        gradient[left:left+len(piece)] += 2*np.pi*mu*weight*np.asarray(jac)[:len(piece)]
                    else:
                        value = self.intensity(labels, padded, mu)
                    total[left:left+len(piece)] += 2*np.pi*mu*weight*np.asarray(value)[:len(piece)]
            if np.any(~np.isfinite(total)) or np.any(total <= 0):
                raise ValueError("Invalid integrated stellar intensity.")
            fluxes[order] = total[:, 0]/total[:, 1]
            if with_jacobian:
                jacobians[order] = (gradient[:, 0]-fluxes[order][:, None]*gradient[:, 1])/total[:, 1, None]
                if not np.isfinite(jacobians[order]).all():
                    raise ValueError("Invalid elemental derivative.")
        self._last_labels = labels.copy()
        self._last_result = fluxes, jacobians
        return self._last_result


def load_data():
    """Read current adoption through its manifest, preserving old experiment files."""
    context_meta = json.loads(CONTEXT.with_suffix(".json").read_text())
    assert digest(CONTEXT) == context_meta["product_sha256"]
    with np.load(CONTEXT) as saved:
        data = {key: saved[key] for key in saved.files}
    manifest_path = OUT/"adopted_telluric_model.json"
    manifest = json.loads(manifest_path.read_text())
    path = ROOT/manifest["product"]
    if "blase" not in path.name:
        raise ValueError("Adopt the requested blase model before running this trial.")
    assert digest(path) == manifest["product_sha256"]
    with np.load(path) as saved:
        adopted = {key: saved[key] for key in saved.files}
    for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c"]:
        np.testing.assert_array_equal(data[key], adopted[key])
    data["adopted_component"] = adopted["component"]
    data["adopted_residual"] = adopted["residual_filtered"]
    data["adopted_stellar"] = adopted["stellar_only"]
    z = adopted["z"]
    cursor = len(z)-len(data["o34_strength_initial"])-len(data["o40_strength_initial"])
    for order in [34, 40]:
        count = len(data[f"o{order}_strength_initial"])
        data[f"o{order}_strength_initial"] = z[cursor:cursor+count]
        cursor += count
    with np.load(OUT/"transformerpayne_stellar_control.npz") as previous:
        data["stellar_initial"] = previous["parameters_calibrated"][:6].copy()
    data["adopted_path"] = str(path)
    data["adopted_manifest"] = manifest
    return data


class ElementProblem:
    """Reuse the fixed ExoJAX response and chain all elemental derivatives."""

    def __init__(self, data, emulator, mask, tied=False, prior_width=None):
        physical = PhysicalProblem(dict(data, included_mask=mask))
        self.data, self.emulator, self.tied = data, emulator, tied
        self.blocks = physical.blocks
        for block in self.blocks:
            original = block["sl"]
            block["sl"] = slice(original.start+N_ELEMENT, original.stop+N_ELEMENT)
        self.initial = np.r_[physical.initial[:6], np.full(N_ELEMENT, physical.initial[2]), physical.initial[6:]]
        self.lower = np.r_[physical.lower[:6], emulator.lower[4:], physical.lower[6:]]
        self.upper = np.r_[physical.upper[:6], emulator.upper[4:], physical.upper[6:]]
        self.scale = np.r_[STELLAR_SCALE, np.full(N_ELEMENT, .3), np.ones(len(physical.initial)-6)]
        self.prior_width = np.ones(N_ELEMENT) if prior_width is None else np.asarray(prior_width)
        self.history = []
        self.started = time.monotonic()

    def canonical(self, theta):
        theta = np.asarray(theta).copy()
        if self.tied:
            theta[6:N_STAR] = theta[2]
        return theta

    def stellar(self, theta):
        theta = self.canonical(theta)
        return self.emulator.evaluate(np.r_[theta[:4], theta[6:N_STAR]], with_jacobian=True)

    def residual(self, theta):
        theta = self.canonical(theta)
        flux, _ = self.stellar(theta)
        residual = [block["target"]-np.asarray(block["train_model"](
            flux[block["order"]], theta[4:6], theta[block["sl"]])) for block in self.blocks]
        residual += [theta[b["sl"]]/b["prior"] for b in self.blocks]
        residual.append((theta[6:N_STAR]-theta[2])/self.prior_width)
        value = np.concatenate(residual)
        self.history.append(dict(seconds=time.monotonic()-self.started, objective=float(value@value),
                                 global_labels=theta[:6].tolist(), abundances_dex=theta[6:N_STAR].tolist()))
        if len(self.history) % 5 == 1:
            print("Evaluation", "tied" if self.tied else "elements", len(self.history),
                  "objective", float(value@value), "global", theta[:6].tolist(), flush=True)
        return value

    def jacobian(self, theta):
        theta = self.canonical(theta)
        flux, gradient = self.stellar(theta)
        rows = []
        for b in self.blocks:
            draw = gradient[b["order"]]
            propagated = -np.asarray(b["label_tangent"](draw, theta[4:6], theta[b["sl"]]))
            row = np.zeros((len(b["target"]), len(theta)))
            row[:, :4] = propagated[:, :4]
            if self.tied:
                row[:, 2] += propagated[:, 4:].sum(axis=1)
            else:
                row[:, 6:N_STAR] = propagated[:, 4:]
            row[:, 4:6] = -np.asarray(b["kin_jacobian"](flux[b["order"]], theta[4:6], theta[b["sl"]]))
            row[:, b["sl"]] = -np.asarray(b["strength_jacobian"](flux[b["order"]], theta[4:6], theta[b["sl"]]))
            rows.append(row)
        strength_prior = np.zeros((len(theta)-N_STAR, len(theta)))
        strength_prior[:, N_STAR:] = np.diag(1/np.concatenate([b["prior"] for b in self.blocks]))
        element_prior = np.zeros((N_ELEMENT, len(theta)))
        if not self.tied:
            element_prior[:, 2] = -1/self.prior_width
            element_prior[:, 6:N_STAR] = np.diag(1/self.prior_width)
        return np.vstack(rows+[strength_prior, element_prior])

    def predictions(self, theta):
        theta = self.canonical(theta)
        flux, _ = self.stellar(theta)
        component, stars = self.data["adopted_component"].copy(), self.data["adopted_stellar"].copy()
        for b in self.blocks:
            component[b["index"]] = np.asarray(b["component"](flux[b["order"]], theta[4:6], theta[b["sl"]]))
            stars[b["index"]] = np.asarray(b["star_only"](flux[b["order"]], theta[4:6]))
        return component, stars


def fit_case(problem, initial, free, name, max_nfev):
    template = problem.canonical(initial)
    scale = problem.scale[free]

    def expand(values):
        theta = template.copy(); theta[free] = values*scale
        return problem.canonical(theta)

    fit = least_squares(lambda x: problem.residual(expand(x)),
                        np.clip(template[free], problem.lower[free]+1e-8, problem.upper[free]-1e-8)/scale,
                        jac=lambda x: problem.jacobian(expand(x))[:, free]*scale[None, :],
                        bounds=(problem.lower[free]/scale, problem.upper[free]/scale), x_scale=1.,
                        method="trf", max_nfev=max_nfev, ftol=1e-7, xtol=1e-6, gtol=1e-5)
    theta = expand(fit.x)
    report = dict(name=name, success=bool(fit.success), nfev=fit.nfev, njev=fit.njev,
                  status=int(fit.status), message=fit.message, optimality=float(fit.optimality),
                  objective=float(2*fit.cost), free_parameters=int(free.sum()),
                  global_labels=dict(zip(STELLAR_NAMES, theta[:6].tolist())),
                  elemental_abundances_dex=dict(zip(ELEMENTS, theta[6:N_STAR].tolist())),
                  elemental_offsets_dex=dict(zip(ELEMENTS, (theta[6:N_STAR]-theta[2]).tolist())),
                  active_free_bounds=np.flatnonzero(free & (
                      np.isclose(theta, problem.lower, atol=1e-5, rtol=0)
                      | np.isclose(theta, problem.upper, atol=1e-5, rtol=0))).tolist())
    print("Fit completed", json.dumps(report), flush=True)
    if not fit.success:
        raise RuntimeError(f"Fit did not converge: {report}")
    return theta, report


def main(args):
    started = time.monotonic()
    (OUT/"transformerpayne_elements_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    data = load_data()
    adopted_hash = digest(data["adopted_path"])
    emulator = ElementEmulator({order: data[f"o{order}_grid"] for order in [34, 40]})
    physical = PhysicalProblem(data)
    mask = data["included_mask"].copy()
    telluric = np.ones_like(data["y_over_c"])
    for b in physical.blocks:
        telluric[b["index"]] = np.asarray(b["component"](
            np.ones_like(data[f"o{b['order']}_grid"]), physical.initial[4:6], physical.initial[b["sl"]]))
        mask[b["index"]] = np.min(telluric[b["index"], :5], axis=0) > .95
    probe = ElementProblem(data, emulator, mask)
    initial = probe.initial.copy()
    flux, gradients = probe.stellar(initial)
    contrast = np.zeros(N_ELEMENT)
    for b in probe.blocks:
        derivative = np.asarray(b["label_tangent"](
            gradients[b["order"]], initial[4:6], initial[b["sl"]]))*WORKING_SIGMA
        contrast = np.maximum(contrast, np.max(np.abs(derivative[:, 4:]), axis=0))
    widths = .3*(.1+.9*contrast**2/(contrast**2+.005**2))
    probe.prior_width = widths
    print("Element response and prior", list(zip(ELEMENTS, contrast.tolist(), widths.tolist())), flush=True)
    # Float32 layers require finite physical label steps for an informative check.
    direction = np.zeros(len(initial))
    direction[6:N_STAR] = np.random.default_rng(20261111).normal(size=N_ELEMENT)*.2
    analytic = probe.jacobian(initial)@direction
    derivative_checks = []
    for step in [.03, .06]:
        numerical = (probe.residual(initial+step*direction)-probe.residual(initial-step*direction))/(2*step)
        error = float(np.linalg.norm(numerical-analytic)/np.linalg.norm(analytic))
        derivative_checks.append(dict(step=step, relative_l2=error))
    if min(row["relative_l2"] for row in derivative_checks) > .03:
        raise ValueError(f"Elemental derivative check failed: {derivative_checks}")
    if args.check_only:
        print("Check complete", derivative_checks, flush=True)
        return
    components = dict(adopted=data["adopted_component"])
    stars = dict(adopted=data["adopted_stellar"])
    parameters, fits, histories = {}, {}, {}
    initial_component, initial_star = probe.predictions(initial)
    components["initial"], stars["initial"] = initial_component, initial_star
    control_start = initial.copy()
    for label, tied in [("global", True), ("elements", False)]:
        training = ElementProblem(data, emulator, mask, tied=tied, prior_width=widths)
        free = np.zeros(len(initial), dtype=bool)
        free[:6] = True
        if not tied:
            free[6:N_STAR] = True
        theta, fits[label+"_stellar"] = fit_case(training, control_start, free,
                                                   label+"_stellar", args.max_nfev)
        parameters[label+"_stellar"] = theta
        components[label+"_stellar"], stars[label+"_stellar"] = training.predictions(theta)
        np.savez_compressed(OUT/f"transformerpayne_elements_{label}_checkpoint.npz", parameters=theta)
        if tied:
            control_start = theta.copy()
        original = ElementProblem(data, emulator, data["included_mask"], tied=tied, prior_width=widths)
        strength_free = np.zeros(len(initial), dtype=bool); strength_free[N_STAR:] = True
        final, fits[label+"_refit"] = fit_case(original, theta, strength_free,
                                                label+"_refit", args.max_nfev)
        np.testing.assert_array_equal(final[:N_STAR], theta[:N_STAR])
        parameters[label+"_refit"] = final
        components[label+"_refit"], stars[label+"_refit"] = original.predictions(final)
        histories[label] = dict(stellar=training.history, strengths=original.history)
    residuals = {name: project(data["y_over_c"]-component, data["wavelength_um"], data["included_mask"])
                 for name, component in components.items()}
    np.testing.assert_allclose(residuals["adopted"], data["adopted_residual"], atol=1e-11, rtol=0)
    for component in components.values():
        np.testing.assert_array_equal(component[0], data["adopted_component"][0])
        assert np.isfinite(component).all() and component.min() >= 0
    arrays = {key: data[key] for key in ["orders", "exposures", "wavelength_um", "included_mask", "score_mask", "y_over_c"]}
    arrays.update(calibration_mask=mask, calibration_telluric=telluric, elemental_contrast=contrast,
                  elemental_prior_width=widths, parameter_lower=probe.lower, parameter_upper=probe.upper)
    for name in components:
        arrays.update({f"component_{name}": components[name], f"residual_{name}": residuals[name],
                       f"stellar_only_{name}": stars[name]})
    for name, theta in parameters.items():
        arrays[f"parameters_{name}"] = theta
    paths = [Path(__file__), ROOT/"transformerpayne_labels.py", ROOT/"fit_transformerpayne_labels.py",
             CONTEXT, CONTEXT.with_suffix(".json"), Path(data["adopted_path"]),
             OUT/"adopted_telluric_model.json", OUT/"transformerpayne_stellar_control.npz",
             ROOT/"cache/transformerpayne_checkpoint.json"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False, plan=PLAN,
                  adopted_product=str(Path(data["adopted_path"]).relative_to(ROOT)),
                  metrics={name: statistics(data, value) for name, value in residuals.items()},
                  fits=fits, history=histories, elemental_contrast=dict(zip(ELEMENTS, contrast.tolist())),
                  elemental_prior_width_dex=dict(zip(ELEMENTS, widths.tolist())),
                  parameter_names=STELLAR_NAMES+[name+"_H_dex" for name in ELEMENTS],
                  stellar_lower=probe.lower[:N_STAR].tolist(), stellar_upper=probe.upper[:N_STAR].tolist(),
                  calibration_counts={str(order): int(mask[b].sum()) for b, order in [(1, 34), (2, 40)]},
                  elemental_derivative_checks=derivative_checks,
                  source_sha256={str(path.relative_to(ROOT)): digest(path) for path in paths},
                  seconds=time.monotonic()-started,
                  limitations=["Only H orders34/40 support TransformerPayne; K order13 and the 4820 feature retain the adopted blase model.",
                               "Independent abundances are conditional effective labels, not a validated stellar abundance measurement.",
                               "Global labels can reach their broad training-domain bounds; elemental priors regularize degeneracy but do not calibrate uncertainty.",
                               "Float32 neural layers limit optimizer precision. Solver stopping conditions and finite-step checks must be read with the residual metrics.",
                               "The original data and masks have been explored. Q5 is profiled separately on evaluation spectra.",
                               "No new planet injection test is performed by this script, and no automatic adoption occurs."])
    np.savez_compressed(OUT/"transformerpayne_elements_fit.npz", **arrays)
    (OUT/"transformerpayne_elements_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    assert digest(data["adopted_path"]) == adopted_hash
    print("Element trial complete", result["seconds"], json.dumps(result["metrics"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-nfev", type=int, default=120)
    parser.add_argument("--check-only", action="store_true")
    main(parser.parse_args())
