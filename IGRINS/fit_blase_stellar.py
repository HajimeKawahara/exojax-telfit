"""Calibrate shared stellar line amplitudes with the actual blase JAX model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
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
import pandas as pd
from scipy.optimize import least_squares, minimize

from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation

from blase_stellar import build_clone
from fit_line_strength import build_context, make_forward, metrics, WORKING_SIGMA
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
ADOPTED_HASH = "2699c05089e79b5392b7393f4d9da1fedf9d9857323a9a514bb30424e8bee875"
PLAN = dict(
    model="PHOENIX * blase(amplitude ratios) / blase(reference), before rotation, stellar RV, telluric absorption and IP",
    training_frames=[0, 19, 39, 59, 78], evaluation_frames=[10, 30, 50, 70],
    stellar_calibration_mask="Minimum adopted telluric-only transmission over five training frames >0.95",
    line_selection="Model-only: maximum training calibration response to unit amplitude ratio >0.001 and calibration/full-window response L2 ratio >0.25",
    stellar_prior="q centered at1, width 0.5*(0.1+0.9*d^2/(d^2+0.005^2)); d is frozen maximum calibration response before Q5",
    stellar_bounds="0 <= amplitude ratio <= min(3, positive-profile limit)",
    fixed_stellar="Line centers and clone widths; rotation4 km/s, limb coefficient0.6, extra width and all RVs",
    stage1="Shared stellar amplitudes only, adopted269 telluric strengths fixed; Q5 profiled on calibration pixels",
    stage2="Freeze calibrated star; refit existing269 telluric strengths on original included pixels with adopted weak-line priors",
    reporting="All cases use original included-mask Q5 and original evaluation masks",
    selection_note="Primary empirical stellar prior declared before evaluation; no residual-dependent line selection or new residual masks",
    adoption=False,
)


def polynomial(wave, mask):
    x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    return jnp.asarray(basis), jnp.asarray(np.linalg.pinv(basis[mask]))


def forward_functions(block, clone, physical, relative_rv):
    """Keep all stellar corrections in the common stellar rest frame."""
    grid, velocity = block["grid"], block["velocity"]
    rv = jnp.asarray(physical["stellar_rv_reference_kms"]+relative_rv)
    line_tau, upper = jnp.asarray(block["line_tau"]), jnp.asarray(block["upper"])
    factors, tau0 = jnp.asarray(block["line_factors"]), jnp.asarray(block["baseline_tau"])
    air = jnp.asarray(block["airmass"])
    widths, offsets = jnp.asarray(block["widths"]), jnp.asarray(block["offsets"])
    target, fraction = block["target"], block["wing_fraction"]

    def shifted(q):
        raw = clone.corrected_raw(q)
        rotated = convolve_rigid_rotation(raw, velocity, 4., .6, 0.)
        broadened = ipgauss(rotated, velocity, physical["stellar_extra_sigma_kms"])
        return jax.vmap(lambda v: sampling(grid, grid, broadened, v))(rv)

    def response(rows, count):
        def one(row, width, offset):
            core = sampling(target, grid, ipgauss(row, velocity, width), offset)
            wing = sampling(target, grid, ipgauss(row, velocity, 2*width), offset)
            return (1-fraction)*core+fraction*wing
        return jax.vmap(one)(rows, widths[:count], offsets[:count])

    def component(q, z, count):
        tau = tau0[:count]+(factors[:count]*(upper*z)[None, :])@line_tau
        intrinsic = shifted(q)[:count]*jnp.exp(-air[:count, None]*tau)
        return response(intrinsic, count)

    return dict(full=jax.jit(lambda q, z: component(q, z, 9)),
                train=jax.jit(lambda q, z: component(q, z, 5)),
                star=jax.jit(lambda q: response(shifted(q), 9)),
                shifted=jax.jit(shifted))


def residual_functions(forward, wave, mask):
    basis, inverse = polynomial(wave, mask)

    def residual(q, z, target):
        rows = target-forward(q, z)
        projected = rows-(rows[:, mask]@inverse.T)@basis.T
        return projected[:, mask].ravel()/WORKING_SIGMA

    return jax.jit(residual), jax.jit(jax.jacfwd(residual, 0)), jax.jit(jax.jacfwd(residual, 1))


def derivative_statistics(forward, point, mask, chunk_size):
    """Reduce model derivatives in bounded chunks without a full Jacobian."""
    count = len(point)
    maximum, selected_power, full_power = (np.zeros(count) for _ in range(3))
    if count == 0:
        return maximum, selected_power, full_power
    point = jnp.asarray(point)
    size = min(count, chunk_size)
    directional = jax.jit(jax.vmap(lambda tangent: jax.jvp(
        forward, (point,), (tangent,))[1]))
    for start in range(0, count, size):
        stop = min(start+size, count)
        tangents = np.zeros((size, count))
        tangents[np.arange(stop-start), np.arange(start, stop)] = 1
        values = np.asarray(directional(jnp.asarray(tangents)))[:stop-start, :5]
        if np.any(mask):
            maximum[start:stop] = np.max(np.abs(values[..., mask]), axis=(1, 2))
            selected_power[start:stop] = np.sum(values[..., mask]**2, axis=(1, 2))
        full_power[start:stop] = np.sum(values**2, axis=(1, 2))
    return maximum, selected_power, full_power


def check_least_squares_gradient(residual, point, gradient, direction, step=1e-4,
                                 maximum_step=np.inf):
    """Check a squared-residual gradient without subtracting large objectives."""
    point, direction = jnp.asarray(point), jnp.asarray(direction)
    analytic = float(np.asarray(gradient) @ np.asarray(direction))
    values, tangent = jax.jvp(residual, (point,), (direction,))
    forward = float(np.asarray(values) @ np.asarray(tangent))
    if not np.isfinite([analytic, forward]).all():
        raise ValueError("Nonfinite directional gradient")
    np.testing.assert_allclose(analytic, forward, rtol=1e-7, atol=1e-8,
                               err_msg="Reverse-gradient and residual JVP disagree")
    checks, previous = [], None
    for trial in np.unique(np.minimum(step*np.array([1., 10., 100.]), maximum_step)):
        if trial <= 0:
            raise ValueError("A positive feasible finite-difference step is required")
        plus = np.asarray(residual(point+trial*direction))
        minus = np.asarray(residual(point-trial*direction))
        # Factoring the scalar costs avoids their cancellation. Larger steps
        # can also resolve residual-level rounding near a stationary point.
        central = float(((plus-minus)/(2*trial)) @ ((plus+minus)/2))
        candidates = [dict(method="central", step=float(trial), finite_difference=central)]
        if previous is not None:
            smaller, smaller_fd = previous
            ratio = (trial/smaller)**2
            # Remove the leading O(h^2) truncation term once a larger step
            # has reduced subtraction noise in the residual vector.
            extrapolated = (ratio*smaller_fd-central)/(ratio-1)
            candidates.append(dict(method="Richardson", step=float(smaller),
                                   larger_step=float(trial), finite_difference=float(extrapolated)))
        previous = trial, central
        matched = False
        for check in candidates:
            fd = check["finite_difference"]
            error = abs(fd-analytic)/max(abs(analytic), 1e-5)
            check.update(relative_error=error, absolute_error=abs(fd-analytic))
            checks.append(check)
            if np.isfinite(fd) and (error <= 1e-3 or abs(fd-analytic) <= 1e-6):
                matched = True
                break
        if matched:
            break
    else:
        raise ValueError(f"Reverse-gradient finite differences failed: {checks}, analytic={analytic}")
    return dict(gradient_directional_error=error, gradient_directional_fd=fd,
                gradient_directional_analytic=analytic,
                gradient_directional_absolute_error=abs(fd-analytic),
                gradient_directional_forward_ad=forward,
                gradient_directional_forward_ad_absolute_error=abs(forward-analytic),
                gradient_directional_steps=checks,
                gradient_directional_method="Feasible factored central differences with step adaptation and Richardson extrapolation, including prior; independently checked by residual JVP")


class FixedReferenceStar:
    """Retain PHOENIX where too few clear pixels constrain a stellar correction."""

    def __init__(self, grid, reason):
        self.raw_reference = jnp.asarray(load_stellar(np.asarray(grid)))
        self.n_lines = 0
        self.centers_angstrom = np.empty(0)
        self.amplitude_ratio_upper = np.empty(0)
        self.metadata = dict(model="Unmodified PHOENIX", reason=reason,
                             optimization=dict(success=True, skipped=True))

    def corrected_raw(self, q):
        if q.shape != (0,):
            raise ValueError("The fixed reference star has no fitted amplitudes")
        return self.raw_reference


class Problem:
    """Frozen data, line basis, masks and priors for paired calibration tests."""

    def __init__(self, context=None, *, allow_sparse_calibration=False,
                 derivative_chunk_size=None, clone_retries=0):
        self.diagnostic_flags = []
        if context is None:
            assert hashlib.sha256((OUT/"adopted_telluric_fit.npz").read_bytes()).hexdigest() == ADOPTED_HASH
            context = build_context()
            with np.load(OUT/"adopted_telluric_fit.npz") as saved:
                self.adopted_z = saved["parameters"][saved["line_parameter_indices"]]
                self.adopted_component = saved["component"].copy()
                contrast = saved["line_contrast"]
            physical = json.loads((OUT/"motion_clean_control.json").read_text())["parameters"]
        else:
            self.adopted_z = np.zeros(context["n_parameters"])
            self.adopted_component = context["reference_component"].copy()
            contrasts = []
            for b, block in enumerate(context["blocks"]):
                zero = jnp.zeros(len(block["records"]))
                if len(zero) == 0:
                    contrasts.append(np.empty(0))
                elif derivative_chunk_size is not None:
                    maximum, _, _ = derivative_statistics(block["train_forward"], zero,
                        context["allowed"][b], derivative_chunk_size)
                    contrasts.append(maximum/block["upper"])
                else:
                    derivative = np.asarray(jax.jit(jax.jacfwd(block["train_forward"]))(zero))
                    contrasts.append(np.max(np.abs(derivative[:, context["allowed"][b]]),
                                            axis=(0, 1))/block["upper"])
            contrast = np.concatenate(contrasts)
            physical = context["physical_parameters"]
        self.context = context
        self.strength_contrast = np.asarray(contrast)
        self.strength_prior = 1+9*contrast**2/(contrast**2+.005**2)
        info = pd.read_csv(ROOT/"data/20201214_info.csv")
        relative_rv = (info.iloc[context["exposures"]]["RV [km/s]"].to_numpy()
                       -info.iloc[0]["RV [km/s]"])
        self.blocks, calibration, telluric, count = [], [], [], 0
        for b, block in enumerate(context["blocks"]):
            unity_block = dict(block, shifted_star=np.ones_like(block["shifted_star"]))
            _, only_telluric = make_forward(unity_block)
            z = self.adopted_z[block["parameter_slice"]]
            transmission = np.asarray(only_telluric(z))
            mask = np.min(transmission[:5], axis=0) > .95
            if mask.sum() < 30:
                if not allow_sparse_calibration:
                    raise ValueError("Insufficient telluric-clear calibration pixels")
                reason = (f"Cube order {block['order']}: only {int(mask.sum())} telluric-clear "
                          "pixels; retain fixed PHOENIX without observational stellar corrections")
                self.diagnostic_flags.append(reason)
                clone = FixedReferenceStar(block["grid"], reason)
            else:
                for attempt in range(clone_retries+1):
                    clone = build_clone(np.asarray(block["grid"]), block["order"])
                    if clone.metadata["optimization"]["success"]:
                        break
                    print("Continuing PHOENIX clone", block["order"], attempt+1, flush=True)
                if not clone.metadata["optimization"]["success"]:
                    raise RuntimeError(f"PHOENIX clone did not converge for order {block['order']}")
            funcs = forward_functions(block, clone, physical, relative_rv)
            q = np.ones(clone.n_lines)
            if clone.n_lines == 0:
                d, coverage = np.empty(0), np.empty(0)
            elif derivative_chunk_size is not None:
                d, selected_power, full_power = derivative_statistics(
                    funcs["star"], q, mask, derivative_chunk_size)
                coverage = np.sqrt(selected_power/np.maximum(full_power, 1e-30))
            else:
                derivative = np.asarray(jax.jit(jax.jacfwd(funcs["star"]))(q))[:5]
                d = np.max(np.abs(derivative[:, mask]), axis=(0, 1))
                coverage = np.sqrt(np.sum(derivative[:, mask]**2, axis=(0, 1))
                                   /np.maximum(np.sum(derivative**2, axis=(0, 1)), 1e-30))
            free = (d > .001) & (coverage > .25)
            calibration_functions = residual_functions(funcs["train"], context["wave"][b], mask)
            original_functions = residual_functions(funcs["train"], context["wave"][b], context["allowed"][b])
            item = dict(index=b, order=block["order"], clone=clone, funcs=funcs,
                        stellar_slice=slice(count, count+clone.n_lines),
                        strength_slice=block["parameter_slice"], mask=mask, free=free,
                        contrast=d, coverage=coverage,
                        prior=.5*(.1+.9*d**2/(d**2+.005**2)),
                        upper=np.asarray(clone.amplitude_ratio_upper),
                        calibration_functions=calibration_functions, original_functions=original_functions)
            self.blocks.append(item)
            count += clone.n_lines
            calibration.append(mask); telluric.append(transmission)
            print("Stellar support", block["order"], "pixels", int(mask.sum()),
                  "free lines", int(free.sum()), "/", clone.n_lines, flush=True)
        self.n_stellar = count
        self.initial_stellar = np.ones(count)
        self.calibration_mask = np.asarray(calibration)
        self.calibration_telluric = np.asarray(telluric)
        reconstructed = self.predict(self.initial_stellar, self.adopted_z)
        self.reconstruction_error = float(np.max(np.abs(reconstructed-self.adopted_component)))
        np.testing.assert_allclose(reconstructed, self.adopted_component, atol=1e-10, rtol=0)

    def predict(self, q, z):
        return np.array([np.asarray(b["funcs"]["full"](q[b["stellar_slice"]], z[b["strength_slice"]]))
                         for b in self.blocks])

    def stellar(self, q):
        return np.array([np.asarray(b["funcs"]["star"](q[b["stellar_slice"]])) for b in self.blocks])

    def fit_stellar(self, target=None, initial=None, max_nfev=100):
        """Fit first-five-frame amplitudes, keeping adopted tellurics fixed."""
        target = self.context["y_over_c"] if target is None else np.asarray(target)
        q = self.initial_stellar.copy() if initial is None else np.asarray(initial).copy()
        self.last_stellar_state = q.copy()
        diagnostics = []
        for b in self.blocks:
            sl, free = b["stellar_slice"], b["free"]
            template = np.ones(sl.stop-sl.start)
            y = jnp.asarray(target[b["index"], :5])
            z = jnp.asarray(self.adopted_z[b["strength_slice"]])
            residual, jacobian, _ = b["calibration_functions"]
            prior = b["prior"][free]

            def expand(values):
                full = template.copy(); full[free] = values
                return full

            def fun(values):
                return np.r_[np.asarray(residual(expand(values), z, y)), (values-1)/prior]

            def jac(values):
                return np.vstack([np.asarray(jacobian(expand(values), z, y))[:, free], np.diag(1/prior)])

            if not free.any():
                q[sl] = template
                diagnostics.append(dict(order=b["order"], success=True, n_free=0, nfev=0))
                continue
            upper = b["upper"][free]
            fit = least_squares(fun, np.clip(q[sl][free], 1e-10, upper-1e-10), jac=jac,
                                bounds=(np.zeros(free.sum()), upper), method="trf", x_scale="jac",
                                max_nfev=max_nfev, ftol=1e-9, xtol=1e-9, gtol=1e-7)
            q[sl] = expand(fit.x)
            self.last_stellar_state = q.copy()
            diagnostics.append(dict(order=b["order"], success=bool(fit.success), nfev=fit.nfev,
                                    n_free=int(free.sum()), objective=float(2*fit.cost),
                                    optimality=float(fit.optimality), message=fit.message,
                                    near_zero_count=int(np.sum(fit.x < 1e-4)),
                                    upper_bound_count=int(np.sum(upper-fit.x < 1e-4))))
            print("Stellar fit", diagnostics[-1], flush=True)
        if not all(row["success"] for row in diagnostics):
            raise RuntimeError(f"Stellar fit did not converge: {diagnostics}")
        return q, diagnostics

    def fit_telluric(self, q, target=None, initial=None, max_nfev=100):
        """Refit original strengths with stellar amplitudes frozen."""
        target = self.context["y_over_c"] if target is None else np.asarray(target)
        z = self.adopted_z.copy() if initial is None else np.asarray(initial).copy()
        diagnostics = []
        for b, original in zip(self.blocks, self.context["blocks"]):
            sl = b["strength_slice"]
            fixed_q, y = jnp.asarray(q[b["stellar_slice"]]), jnp.asarray(target[b["index"], :5])
            residual, _, jacobian = b["original_functions"]
            prior, lower = self.strength_prior[sl], -1/original["upper"]

            if sl.stop == sl.start:
                diagnostics.append(dict(order=b["order"], success=True, skipped=True,
                                        n_free=0, nfev=0,
                                        message="No eligible telluric intensity corrections"))
                continue

            def fun(values):
                return np.r_[np.asarray(residual(fixed_q, values, y)), values/prior]

            def jac(values):
                return np.vstack([np.asarray(jacobian(fixed_q, values, y)), np.diag(1/prior)])

            if getattr(self, "telluric_solver", "trf") == "lbfgsb":
                scale = jnp.asarray(prior)

                def objective(scaled):
                    data = residual(fixed_q, scaled*scale, y)
                    return .5*(jnp.vdot(data, data)+jnp.vdot(scaled, scaled))

                value_gradient = jax.jit(jax.value_and_grad(objective))
                start = np.maximum(z[sl]/prior, lower/prior+1e-10)
                value, gradient = value_gradient(start)
                expected = .5*np.sum(fun(start*prior)**2)
                np.testing.assert_allclose(float(value), expected, rtol=1e-12, atol=1e-8)
                direction = np.random.default_rng(20261112).normal(size=len(start))
                direction[start-lower/prior < 1e-3] = 0
                direction_norm = np.linalg.norm(direction)
                if direction_norm > 0:
                    direction /= direction_norm
                def scaled_residual(values):
                    return jnp.concatenate((residual(fixed_q, values*scale, y), values))

                nonzero = direction != 0
                maximum_step = (.5*np.min((start-lower/prior)[nonzero]/abs(direction[nonzero]))
                                if nonzero.any() else np.inf)
                gradient_check = check_least_squares_gradient(
                    scaled_residual, start, gradient, direction, maximum_step=maximum_step)
                calls, steps = [0], [0]

                def value_and_gradient(values):
                    calls[0] += 1
                    value, gradient = value_gradient(values)
                    return float(value), np.asarray(gradient)

                def accepted(intermediate_result):
                    steps[0] += 1
                    state = z.copy()
                    state[sl] = intermediate_result.x*prior
                    self.last_telluric_state = state
                    checkpoint = getattr(self, "telluric_checkpoint_path", None)
                    if checkpoint is not None:
                        temporary = checkpoint.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, q=q, z=state)
                        temporary.replace(checkpoint)
                    if steps[0] <= 5 or steps[0] % 10 == 0:
                        print("Telluric L-BFGS-B step", dict(order=b["order"],
                              iteration=steps[0], nfev=calls[0],
                              objective=float(2*intermediate_result.fun)), flush=True)

                options = dict(maxiter=10*max_nfev, maxfun=30*max_nfev, maxcor=50,
                               ftol=1e-12, gtol=1e-7, maxls=40)
                fit = minimize(value_and_gradient, start, jac=True, method="L-BFGS-B",
                               bounds=list(zip(lower/prior, np.full(len(start), np.inf))),
                               callback=accepted, options=options)
                z[sl] = fit.x*prior
                self.last_telluric_state = z.copy()
                projected = np.asarray(fit.jac).copy()
                projected[(fit.x-lower/prior < 1e-8) & (projected > 0)] = 0
                diagnostics.append(dict(order=b["order"], success=bool(fit.success),
                    nfev=int(fit.nfev), nit=int(fit.nit), n_free=len(fit.x),
                    objective=float(2*fit.fun), optimality=float(np.max(np.abs(projected))),
                    message=str(fit.message), method="L-BFGS-B", options=options,
                    coordinates="z divided by frozen weak-line prior width",
                    objective_equivalence_error=abs(float(value)-expected),
                    **gradient_check))
                print("Telluric refit", diagnostics[-1], flush=True)
                if not fit.success:
                    raise RuntimeError(f"Telluric refit did not converge: {diagnostics[-1]}")
                continue

            solver = {}
            if len(z[sl]) > 512:
                options = getattr(self, "telluric_solver_options",
                                  {"atol": 1e-8, "btol": 1e-8, "maxiter": 200})

                def callback(intermediate_result):
                    state = z.copy()
                    state[sl] = intermediate_result.x
                    self.last_telluric_state = state
                    checkpoint = getattr(self, "telluric_checkpoint_path", None)
                    if checkpoint is not None:
                        temporary = checkpoint.with_suffix(".tmp.npz")
                        np.savez_compressed(temporary, q=q, z=state)
                        temporary.replace(checkpoint)
                    print("Telluric step", dict(order=b["order"],
                          iteration=int(intermediate_result.nit),
                          nfev=int(intermediate_result.nfev),
                          objective=float(2*intermediate_result.cost)), flush=True)

                solver = dict(tr_solver="lsmr", tr_options=dict(options), callback=callback)
            fit = least_squares(fun, np.maximum(z[sl], lower+1e-10), jac=jac,
                                bounds=(lower, np.inf), method="trf", x_scale="jac", max_nfev=max_nfev,
                                ftol=1e-9, xtol=1e-9, gtol=1e-7, **solver)
            z[sl] = fit.x
            self.last_telluric_state = z.copy()
            diagnostics.append(dict(order=b["order"], success=bool(fit.success), nfev=fit.nfev,
                                    n_free=len(fit.x), objective=float(2*fit.cost),
                                    optimality=float(fit.optimality), message=fit.message,
                                    tr_solver=solver.get("tr_solver", "exact"),
                                    tr_options=solver.get("tr_options", {})))
            print("Telluric refit", diagnostics[-1], flush=True)
        if not all(row["success"] for row in diagnostics):
            raise RuntimeError(f"Telluric refit did not converge: {diagnostics}")
        return z, diagnostics


def additional_metrics(context, residual, reference_star):
    result = {}
    strong = np.max(1-reference_star[:, :5], axis=1) > .02
    local = np.zeros_like(context["allowed"])
    nu = 1e4/context["wave"][0]
    local[0] = (nu >= 4819.74) & (nu <= 4820.10) & context["score"][0]
    for name, mask in [("strong_stellar", strong), ("full_window", np.ones_like(strong)),
                       ("stellar_4819_92", local)]:
        keep = np.broadcast_to(mask[:, None], residual[:, 5:].shape)
        result[name] = dict(n_samples=int(keep.sum()),
                            rms_percent=float(100*np.sqrt(np.mean(residual[:, 5:][keep]**2))))
    return result


def main(args):
    started = time.monotonic()
    (OUT/"blase_stellar_plan.json").write_text(json.dumps(PLAN, indent=2)+"\n")
    problem = Problem()
    context = problem.context
    q, stellar_fit = problem.fit_stellar(max_nfev=args.max_nfev)
    np.savez_compressed(OUT/"blase_stellar_checkpoint.npz", q=q)
    z, strength_fit = problem.fit_telluric(q, max_nfev=args.max_nfev)
    components = dict(adopted=problem.adopted_component,
                      stellar=problem.predict(q, problem.adopted_z), refit=problem.predict(q, z))
    stars = dict(adopted=problem.stellar(problem.initial_stellar), corrected=problem.stellar(q))
    results, residuals = {}, {}
    for name, value in components.items():
        assert np.isfinite(value).all() and value.min() >= 0
        results[name], residuals[name] = metrics(context, value)
        results[name]["diagnostic"] = additional_metrics(context, residuals[name], stars["adopted"])
    arrays = dict(orders=context["orders"], exposures=context["exposures"], wavelength_um=context["wave"],
                  included_mask=context["allowed"], score_mask=context["score"], y_over_c=context["y_over_c"],
                  calibration_mask=problem.calibration_mask, calibration_telluric=problem.calibration_telluric,
                  q=q, z=z, adopted_z=problem.adopted_z, initial_stellar=problem.initial_stellar)
    for name in components:
        arrays.update({f"component_{name}": components[name], f"residual_{name}": residuals[name]})
    for name, value in stars.items():
        arrays[f"stellar_only_{name}"] = value
    line_records, mask_counts = [], []
    for b in problem.blocks:
        i, sl = b["index"], b["stellar_slice"]
        strong = np.max(1-stars["adopted"][i, :5], axis=0) > .02
        mask_counts.append(dict(order=b["order"], calibration_pixels=int(b["mask"].sum()),
                                strong_stellar_calibration_pixels=int(np.sum(b["mask"] & strong)),
                                newly_used_pixels=int(np.sum(b["mask"] & ~context["allowed"][i])),
                                original_score_overlap=int(np.sum(b["mask"] & context["score"][i])),
                                original_included_pixels=int(context["allowed"][i].sum()),
                                free_lines=int(b["free"].sum()), candidate_lines=len(b["free"])))
        for j, center in enumerate(b["clone"].centers_angstrom):
            line_records.append(dict(order=b["order"], center_rest_angstrom=float(center),
                                     center_rest_cm1=float(1e8/center), fitted=bool(b["free"][j]),
                                     amplitude_ratio=float(q[sl][j]), prior_width=float(b["prior"][j]),
                                     calibration_contrast=float(b["contrast"][j]),
                                     calibration_response_coverage=float(b["coverage"][j])))
    sources = [Path(__file__), ROOT/"blase_stellar.py", ROOT/"vendor/blase/blase/jaxemulator.py",
               ROOT/"fit_line_strength.py", OUT/"adopted_telluric_fit.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False, plan=PLAN,
                  environment={name: version(name) for name in ["jax", "jaxlib", "numpy", "scipy"]},
                  metrics=results, reconstruction_error=problem.reconstruction_error,
                  fits=dict(stellar=stellar_fit, telluric=strength_fit), mask_counts=mask_counts,
                  line_records=line_records, clone_metadata=[b["clone"].metadata for b in problem.blocks],
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  seconds=time.monotonic()-started,
                  limitations=["Phenomenological feature amplitudes are not atomic oscillator strengths or measured stellar abundances.",
                               "Original continuum, atmospheric profile, columns, IP and wavelength calibration remain conditional fitted inputs.",
                               "All nine exposures were previously explored; held-out frames are not pristine independent model selection.",
                               "Q5 is profiled independently for every evaluation frame; the same operation must be applied to planet templates.",
                               "The small barycentric span cannot by itself separate a stationary planet feature from a shared stellar correction.",
                               "Additional unreported HITRAN strengths from the preceding experiment are not included in this controlled comparison."])
    np.savez_compressed(OUT/"blase_stellar_fit.npz", **arrays)
    (OUT/"blase_stellar_fit.json").write_text(json.dumps(result, indent=2)+"\n")
    print("Metrics", json.dumps(results), flush=True)
    assert hashlib.sha256((OUT/"adopted_telluric_fit.npz").read_bytes()).hexdigest() == ADOPTED_HASH
    print("Completed blase stellar fit", result["seconds"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-nfev", type=int, default=100)
    main(parser.parse_args())
