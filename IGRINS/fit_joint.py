"""Joint stellar/telluric fit with an exposure reserved for conditional validation.

All assets and outputs stay under IGRINS. Run prepare_stellar.py and
prepare_atmosphere.py first. The atmosphere is either one shared effective
slab or a fixed meteorological profile. Continuum coefficients are eliminated
by linear least squares, including their dependence in the JAX Jacobian.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector
from exojax.postproc.response import ipgauss, ipgauss_variable_sampling, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import Patm, c, m_u

from atmosphere_profile import load_profile
from hitran_cache import load_mdb
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent


def cross_section(opa, temperature, pressure, self_pressure, shifted):
    """Existing ExoJAX opacity, optionally with the HITRAN air line shift."""
    if not shifted:
        return opa.xsvector(temperature, pressure, self_pressure)
    db = opa.mdb
    strength = line_strength(temperature, db.logsij0, db.nu_lines, db.elower,
                             db.qr_interp_lines(temperature, 296.), 296.)
    width = gamma_hitran(pressure, temperature, self_pressure, db.n_air,
                         db.gamma_air, db.gamma_self) + gamma_natural(db.A)
    sigma = doppler_sigma(db.nu_lines, temperature, db.molmass)
    return xsvector(opa.opainfo - (db.delta_air * (pressure - self_pressure) / Patm)[:, None],
                    sigma, width, strength)


def profile_tau(opa, grid, profile, species, shifted=False, self_broadening=False):
    """Cache fixed-profile vertical optical depths, retaining raw provenance."""
    pressure = np.asarray(profile["pressure_bar"])
    temperature = np.asarray(profile["temperature_K"])
    vmr = np.asarray(profile["vmr"][species])
    column = np.asarray(profile["number_column_cm2"]) * vmr
    if np.any(column < 0) or not np.all((temperature >= 150) & (temperature <= 400)):
        raise ValueError("Invalid columns or temperatures outside cached partition sums.")
    # Self widths are fixed at the reference profile; fitted scale changes column only.
    self_pressure = pressure * vmr if self_broadening else np.zeros_like(pressure)
    digest = hashlib.sha256()
    for value in (grid, pressure, temperature, column, self_pressure,
                  opa.mdb.nu_lines, opa.mdb.logsij0, opa.mdb.gamma_air,
                  opa.mdb.gamma_self, opa.mdb.n_air, opa.mdb.delta_air):
        digest.update(np.asarray(value).tobytes())
    digest.update(str(shifted).encode())
    path = ROOT / "cache" / f"tau_{species}_{digest.hexdigest()[:20]}.npz"
    if path.exists():
        with np.load(path) as saved:
            return saved["tau"], float(np.sum(column))
    xs = jax.jit(lambda t, p, ps: cross_section(opa, t, p, ps, shifted))
    tau = np.zeros(len(grid))
    for t, p, ps, n in zip(temperature, pressure, self_pressure, column):
        tau += n * np.asarray(xs(t, p, ps))
    np.savez_compressed(path, tau=tau, column=column)
    return tau, float(np.sum(column))


def metrics(residual, mask):
    values = np.asarray(residual)[mask]
    return {"n_pixels": int(len(values)),
            "rms": float(np.sqrt(np.mean(values ** 2))),
            "mad": float(1.4826 * np.median(np.abs(values - np.median(values))))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=[13, 34, 40])
    parser.add_argument("--train", type=int, nargs="+", default=[0, 78])
    parser.add_argument("--holdout", type=int, default=39)
    parser.add_argument("--pixels", type=int, nargs=2, default=[600, 1500])
    parser.add_argument("--atmosphere", choices=["slab", "profile"], default="profile")
    parser.add_argument("--no-star", action="store_true")
    parser.add_argument("--layers", type=int, default=20)
    parser.add_argument("--dv", type=float, default=.75)
    parser.add_argument("--shift", action="store_true")
    parser.add_argument("--self-broadening", action="store_true",
                        help="Use self widths at the reference composition (frozen when scaling columns)")
    parser.add_argument("--species", nargs="+", default=["H2O", "CO2"])
    parser.add_argument("--initial", type=Path)
    parser.add_argument("--pixel-mask", type=Path,
                        help="NPZ with orders, wavelength_um and a fixed per-order boolean mask")
    parser.add_argument("--fix-star", action="store_true",
                        help="Keep the stellar RV and width at the --initial values")
    parser.add_argument("--ip", choices=["gaussian", "mixture"], default="gaussian")
    parser.add_argument("--temperature-offset", type=float, default=0.,
                        help="Fixed additive temperature perturbation of the pressure profile, in K")
    parser.add_argument("--temperature-nodes", type=float, nargs="*", default=[],
                        help="Pressure nodes in bar with free temperature corrections shared across exposures")
    parser.add_argument("--temperature-bound", type=float, default=20.)
    parser.add_argument("--water-shape", choices=["fixed", "tilt", "nodes"], default="fixed")
    parser.add_argument("--pressure-scale", type=float, default=1.,
                        help="Fixed multiplier of pressure boundaries and hydrostatic columns")
    parser.add_argument("--line-cutoff", type=float, default=1e-28)
    parser.add_argument("--line-margin", type=float, default=3.)
    parser.add_argument("--continuum-degree", type=int, default=3)
    parser.add_argument("--quadratic-wavelength", action="store_true")
    parser.add_argument("--direct-ip", action="store_true",
                        help="Evaluate Gaussian response directly at observed wavelengths")
    parser.add_argument("--ip-wavelength", action="store_true",
                        help="Fit an order-dependent linear slope of log IP width; implies direct IP")
    parser.add_argument("--ip-epoch", action="store_true",
                        help="Fit one relative IP width per exposure, shared across orders")
    parser.add_argument("--ip-asymmetry", action="store_true",
                        help="Fit a shared separation of the centered Gaussian mixture components")
    parser.add_argument("--opacity-override", type=Path,
                        help="NPZ with grid_oN and SPECIES_oN fixed vertical optical depths")
    parser.add_argument("--freeze-shared", action="store_true",
                        help="Reuse a matching --initial training fit and calibrate only the held-out exposure")
    parser.add_argument("--loss", choices=["linear", "soft_l1"], default="linear")
    parser.add_argument("--loss-scale", type=float, default=.005,
                        help="Fixed-data-scale transition for the optional robust nonlinear objective")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--tag", default="")
    parser.add_argument("--check-layers", action="store_true",
                        help="Compare fixed-parameter 10/20/40 layer predictions")
    args = parser.parse_args()
    dynamic_profile = bool(args.temperature_nodes or args.water_shape != "fixed")
    args.direct_ip |= args.ip_wavelength
    if dynamic_profile and args.atmosphere != "profile":
        raise ValueError("Flexible profiles require --atmosphere profile.")
    if args.ip_asymmetry and args.ip != "mixture":
        raise ValueError("IP asymmetry requires a Gaussian mixture.")
    if not 0 < args.temperature_bound < 30 or not 0 < args.pressure_scale < 2:
        raise ValueError("Invalid temperature bound or pressure scale.")
    if not 0 <= args.continuum_degree <= 7:
        raise ValueError("Use a continuum degree between zero and seven.")
    if args.temperature_nodes and (np.any(np.asarray(args.temperature_nodes) <= 0) or
                                   np.any(np.diff(args.temperature_nodes) <= 0)):
        raise ValueError("Temperature pressure nodes must be positive and increasing.")
    if dynamic_profile and args.check_layers:
        raise ValueError("Check flexible-profile layer convergence with separate fits.")
    if args.opacity_override and (dynamic_profile or args.atmosphere != "profile" or args.check_layers):
        raise ValueError("Opacity overrides require a fixed profile without layer checks.")
    started = time.monotonic()
    exposures = list(args.train) + [args.holdout]
    if len(set(exposures)) != len(exposures) or args.dv <= 0:
        raise ValueError("Exposure sets must be disjoint; grid spacing must be positive.")
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    lo, hi = args.pixels
    if not (0 <= lo < hi <= cube.shape[-1]):
        raise ValueError("Invalid pixel interval.")
    profile = load_profile(args.layers)
    if args.temperature_offset and args.atmosphere != "profile":
        raise ValueError("Temperature offset requires a pressure profile.")
    profile["temperature_K"] = profile["temperature_K"] + args.temperature_offset
    for key in ["pressure_bar", "delta_pressure_bar", "pressure_boundaries_bar", "number_column_cm2"]:
        profile[key] = profile[key] * args.pressure_scale
    if dynamic_profile:
        from profile_opacity import (prepare_profile_opacity, pressure_node_profile,
                                     redistribution_weights)
    override = None
    if args.opacity_override:
        with np.load(args.opacity_override) as saved:
            override = {key: saved[key] for key in saved.files}
    if "H2O" not in args.species:
        raise ValueError("Include H2O so each exposure has a water scaling parameter.")
    if any(s not in profile["vmr"] for s in args.species):
        raise ValueError("Missing molecular profile.")
    pixel = np.arange(hi - lo)
    # Whole wavelength blocks, with guards, rather than interleaved pixels.
    calibration_mask = ((pixel // 80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    score_mask = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    allowed = np.ones((len(args.orders), hi-lo), dtype=bool)
    mask_metadata = None
    if args.pixel_mask:
        with np.load(args.pixel_mask) as saved:
            for b, order in enumerate(args.orders):
                matches = np.flatnonzero(saved["orders"] == order)
                if len(matches) != 1:
                    raise ValueError(f"Mask must contain order {order} exactly once.")
                index = matches[0]
                np.testing.assert_allclose(saved["wavelength_um"][index],
                                           wavelength[order, lo:hi][::-1], rtol=0, atol=1e-12)
                if saved["mask"].dtype != np.bool_:
                    raise ValueError("Pixel mask must be boolean.")
                allowed[b] = saved["mask"][index]
        mask_metadata = dict(path=str(args.pixel_mask.resolve()),
                             sha256=hashlib.sha256(args.pixel_mask.read_bytes()).hexdigest())
    if np.any(np.sum(allowed & calibration_mask, axis=1) < 20) or np.any(np.sum(allowed & score_mask, axis=1) < 20):
        raise ValueError("Insufficient included calibration or scoring pixels.")
    n_continuum = args.continuum_degree + 1
    basis = jnp.asarray(np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, hi-lo), args.continuum_degree))
    blocks = []
    for b, order in enumerate(args.orders):
        wave = wavelength[order, lo:hi][::-1].copy()
        nu = 1e4 / wave
        flux = cube[order, exposures, lo:hi][:, ::-1].copy()
        if not np.all(np.isfinite(flux)) or not np.all(np.diff(nu) > 0):
            raise ValueError("Use finite spectra with a strictly increasing wavenumber grid.")
        scales = np.percentile(flux[:, allowed[b]], 95, axis=1)
        scales[-1] = np.percentile(flux[-1, allowed[b] & calibration_mask], 95)
        y = flux / scales[:, None]
        minimum, maximum = nu[0] * np.exp(-220/c), nu[-1] * np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/args.dv)) + 1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c * np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1) * dv
        stellar = np.ones(n) if args.no_star else load_stellar(grid)
        # Fixed v sin i = 4 km/s; fit one additional Gaussian stellar width.
        stellar = np.asarray(convolve_rigid_rotation(jnp.asarray(stellar), velocity, 4., .6, 0.))
        opacities, tau, columns, tables = [], [], [], []
        for species in args.species:
            db = load_mdb(species, grid[0]-args.line_margin, grid[-1]+args.line_margin, crit=args.line_cutoff)
            if not len(db.nu_lines):
                raise ValueError(f"No {species} lines; adjust --species for this order.")
            opa = OpaDirect(db, grid)
            print(f"Order {order}: {species}, {len(db.nu_lines)} lines", flush=True)
            if args.atmosphere == "profile":
                optical_depth, column = profile_tau(opa, grid, profile, species, args.shift, args.self_broadening)
                if override is not None and f"{species}_o{order}" in override:
                    np.testing.assert_allclose(override[f"grid_o{order}"], grid, rtol=0, atol=1e-10)
                    optical_depth = override[f"{species}_o{order}"]
                    if optical_depth.shape != grid.shape or np.any(~np.isfinite(optical_depth)) or np.any(optical_depth < 0):
                        raise ValueError("Invalid override optical depth.")
                    if f"column_{species}_o{order}" in override:
                        np.testing.assert_allclose(override[f"column_{species}_o{order}"], column, rtol=1e-10)
                if dynamic_profile:
                    tables.append(prepare_profile_opacity(opa, grid, profile, species, cross_section,
                                                          args.shift, args.self_broadening))
            else:
                optical_depth = np.zeros(len(grid))
                column = float(np.sum(profile["number_column_cm2"] * profile["vmr"][species]))
            opacities.append(opa)
            tau.append(optical_depth)
            columns.append(column)
        blocks.append(dict(order=order, wave=wave, nu=nu, flux=flux, y=jnp.asarray(y),
                           scales=scales, grid=grid, velocity=velocity, stellar=jnp.asarray(stellar),
                           opacities=opacities, tau=jnp.asarray(tau), columns=np.asarray(columns), tables=tables))

    names, initial, lower, upper = [], [], [], []
    def add(name, value, low, high):
        index = len(names)
        names.append(name); initial.append(value); lower.append(low); upper.append(high)
        return index
    if not args.no_star:
        star_rv = add("stellar_rv_reference_kms", float(info.iloc[exposures[0]]["RV [km/s]"]), -80, 80)
        star_sigma = add("stellar_extra_sigma_kms", 2., .2, 8.)
    if args.ip == "mixture":
        wing_fraction = add("ip_wing_fraction", .02, 0., .5)
    if args.ip_asymmetry:
        ip_separation = add("ip_mixture_separation_kms", 0., -5., 5.)
    temperature_indices = [add(f"temperature_delta_node{i}_K", 0., -args.temperature_bound,
                               args.temperature_bound) for i in range(len(args.temperature_nodes))]
    water_shape_indices = [add(f"water_shape_node{i}_dex", 0., -.7, .7)
                           for i in range({"fixed": 0, "tilt": 1, "nodes": 2}[args.water_shape])]
    if args.atmosphere == "slab":
        slab_t = add("slab_temperature_K", 260., 180., 330.)
        slab_p = add("slab_pressure_bar", .4, .03, 1.1)
    abundance = {s: add(f"log10_scale_{s}", 0., -1., 1.) for s in args.species if s != "H2O"}
    water = [add(f"log10_water_e{e}", 0., -1.5, 1.5) for e in exposures]
    epoch_rv = [None] + [add(f"calibration_rv_e{e}_kms", 0., -2., 2.) for e in exposures[1:]]
    epoch_ip = ([None] + [add(f"log_ip_width_e{e}", 0., -.15, .15) for e in exposures[1:]]
                if args.ip_epoch else [None] * len(exposures))
    instruments = []
    quadratic, ip_slope = [], []
    for block in blocks:
        order = block["order"]
        instruments.append((add(f"ip_sigma_o{order}_kms", 2.5, 1., 6.),
                            add(f"calibration_rv_o{order}_kms", .5, -5., 5.),
                            add(f"stretch_o{order}_kms", 0., -3., 3.)))
        quadratic.append(add(f"quadratic_o{order}_kms", 0., -2., 2.) if args.quadratic_wavelength else None)
        ip_slope.append(add(f"ip_log_slope_o{order}", 0., -.3, .3) if args.ip_wavelength else None)
    if args.initial:
        previous_fit = json.loads(args.initial.read_text())
        previous = previous_fit["parameters"]
        initial = [previous.get(k, v) for k, v in zip(names, initial)]
    if args.fix_star and (args.no_star or not args.initial or
                          any(key not in previous for key in [names[star_rv], names[star_sigma]])):
        raise ValueError("--fix-star requires an initial fit with stellar RV and width.")
    lower, upper, initial = map(np.asarray, (lower, upper, initial))
    initial = np.clip(initial, lower + 1e-7, upper - 1e-7)
    relative_rv = info.iloc[exposures]["RV [km/s]"].to_numpy() - info.iloc[exposures[0]]["RV [km/s]"]
    airmass = info.iloc[exposures]["Air Mass"].to_numpy()
    x = jnp.linspace(-1., 1., hi-lo)
    masks = np.broadcast_to(allowed[:, None, :], (len(blocks), len(exposures), hi-lo)).copy()
    masks[:, -1] &= calibration_mask

    def profile_state(p):
        delta = (pressure_node_profile(profile["pressure_bar"], args.temperature_nodes,
                                        p[jnp.asarray(temperature_indices)])
                 if temperature_indices else jnp.zeros(args.layers))
        if args.water_shape == "tilt":
            shape = p[water_shape_indices[0]] * jnp.log10(profile["pressure_bar"] / .4)
        elif args.water_shape == "nodes":
            shape = pressure_node_profile(profile["pressure_bar"], [.1, .35, .7],
                                           jnp.array([p[water_shape_indices[0]], p[water_shape_indices[1]], 0.]))
        else:
            shape = jnp.zeros(args.layers)
        column = profile["number_column_cm2"] * profile["vmr"]["H2O"]
        weights = redistribution_weights(column, shape) if dynamic_profile else jnp.ones(args.layers)
        return delta, weights

    def evaluate(p, tau_override=None):
        predictions, continua, components = [], [], []
        if dynamic_profile:
            temperature_delta, water_weights = profile_state(p)
        for b, (block, instrument) in enumerate(zip(blocks, instruments)):
            if args.atmosphere == "profile":
                if tau_override is not None:
                    tau = tau_override[b]
                elif dynamic_profile:
                    tau = jnp.stack([table.tau(temperature_delta, water_weights if species == "H2O" else None)
                                     for table, species in zip(block["tables"], args.species)])
                else:
                    tau = block["tau"]
            else:
                tau = jnp.stack([cross_section(opa, p[slab_t], p[slab_p],
                                               p[slab_p]*column/np.sum(profile["number_column_cm2"])
                                               if args.self_broadening else 0., args.shift) * column
                                 for opa, column in zip(block["opacities"], block["columns"])])
            star = block["stellar"] if args.no_star else ipgauss(block["stellar"], block["velocity"], p[star_sigma])
            rows, cs, parts = [], [], []
            beta, rv, stretch = (p[i] for i in instrument)
            for e, exposure in enumerate(exposures):
                factors = jnp.stack([10. ** p[water[e] if s == "H2O" else abundance[s]] for s in args.species])
                transmission = jnp.exp(-airmass[e] * jnp.sum(factors[:, None] * tau, axis=0))
                shifted_star = star if args.no_star else sampling(block["grid"], block["grid"], star, p[star_rv] + relative_rv[e])
                intrinsic = shifted_star * transmission
                offset = 0. if epoch_rv[e] is None else p[epoch_rv[e]]
                width = beta * (1. if epoch_ip[e] is None else jnp.exp(p[epoch_ip[e]]))
                correction = stretch*x + (0. if quadratic[b] is None else p[quadratic[b]]*(2*x*x-1))
                target_nu = block["nu"] * (1 + correction/c)
                separation = p[ip_separation] if args.ip_asymmetry else 0.
                fraction = p[wing_fraction] if args.ip == "mixture" else 0.
                if args.direct_ip:
                    variable_width = width * (jnp.ones_like(x) if ip_slope[b] is None else jnp.exp(p[ip_slope[b]]*x))
                    sampled = ipgauss_variable_sampling(target_nu, block["grid"], intrinsic,
                                                         variable_width, rv+offset-fraction*separation)
                    if args.ip == "mixture":
                        sampled = ((1-fraction)*sampled + fraction*ipgauss_variable_sampling(
                            target_nu, block["grid"], intrinsic, 2*variable_width,
                            rv+offset+(1-fraction)*separation))
                else:
                    core = ipgauss(intrinsic, block["velocity"], width)
                    sampled = sampling(target_nu, block["grid"], core, rv+offset-fraction*separation)
                    if args.ip == "mixture":
                        wing = ipgauss(intrinsic, block["velocity"], 2*width)
                        sampled = ((1-fraction)*sampled + fraction*sampling(
                            target_nu, block["grid"], wing, rv+offset+(1-fraction)*separation))
                design = basis * sampled[:, None]
                weight = jnp.asarray(masks[b,e], dtype=jnp.float64)
                normal = design.T @ (weight[:, None] * design)
                coefficients = jnp.linalg.solve(normal + 1e-12*jnp.eye(n_continuum), design.T @ (weight * block["y"][e]))
                continuum = basis @ coefficients
                rows.append(sampled * continuum); cs.append(continuum); parts.append(sampled)
            predictions.append(jnp.stack(rows)); continua.append(jnp.stack(cs)); components.append(jnp.stack(parts))
        return jnp.stack(predictions), jnp.stack(continua), jnp.stack(components)

    observed = jnp.stack([block["y"] for block in blocks])
    training = masks.copy(); training[:, -1] = False
    calibration = np.zeros(observed.shape, dtype=bool); calibration[:, -1] = allowed & calibration_mask
    scoring = np.zeros(observed.shape, dtype=bool); scoring[:, -1] = allowed & score_mask
    forward = jax.jit(lambda p: evaluate(p)[0])
    train_residual = jax.jit(lambda p: (forward(p) - observed)[training])
    validation_residual = jax.jit(lambda p: (forward(p) - observed)[calibration])
    heldout_indices = [water[-1], epoch_rv[-1]]
    if args.ip_epoch:
        heldout_indices.append(epoch_ip[-1])
    fixed_indices = [star_rv, star_sigma] if args.fix_star else []
    train_indices = np.array([i for i in range(len(names)) if i not in heldout_indices + fixed_indices])
    if args.freeze_shared:
        if not args.initial or not previous_fit["training_fit"]["success"]:
            raise ValueError("Frozen prediction requires a successful initial training fit.")
        source_args = previous_fit["provenance"]["command_arguments"]
        ignored_args = {"holdout", "initial", "tag", "max_nfev", "check_layers", "freeze_shared"}
        for key, value in vars(args).items():
            if key in ignored_args:
                continue
            value = str(value) if isinstance(value, Path) else value
            if source_args.get(key, parser.get_default(key)) != value:
                raise ValueError(f"Frozen model configuration mismatch: {key}")
        if any(names[i] not in previous for i in train_indices):
            raise ValueError("Missing shared parameters in initial training fit.")
        for i in list(train_indices) + fixed_indices:
            value = previous[names[i]]
            if not lower[i] <= value <= upper[i]:
                raise ValueError(f"Frozen parameter outside bounds: {names[i]}")
            initial[i] = value

    def optimize(p0, indices, residual, label):
        indices = jnp.asarray(indices)
        base = jnp.asarray(p0)
        fun = jax.jit(lambda q: residual(base.at[indices].set(q)))
        jac = jax.jit(jax.jacfwd(fun))
        q0 = np.asarray(p0)[np.asarray(indices)]
        print(f"Compiling {label}: {len(q0)} parameters", flush=True)
        fun(q0).block_until_ready(); jac(q0).block_until_ready()
        counter = 0
        def scipy_fun(q):
            nonlocal counter
            counter += 1
            value = np.asarray(fun(q))
            if counter % 10 == 0:
                print(f"{label} {counter}: scaled RMS {np.sqrt(np.mean(value**2)):.6f}", flush=True)
            return value
        fit = least_squares(scipy_fun, q0, jac=lambda q: np.asarray(jac(q)), method="trf",
                            bounds=(lower[np.asarray(indices)], upper[np.asarray(indices)]),
                            x_scale="jac", loss=args.loss, f_scale=args.loss_scale,
                            max_nfev=args.max_nfev, ftol=1e-7, xtol=1e-7, gtol=1e-7)
        best = np.asarray(base.at[indices].set(fit.x))
        direction = np.random.default_rng(2026).normal(size=len(q0)) * np.maximum(np.abs(fit.x), 1)
        step = 1e-6
        finite = (np.asarray(fun(fit.x+step*direction))-np.asarray(fun(fit.x-step*direction)))/(2*step)
        exact = np.asarray(jac(fit.x)) @ direction
        error = np.linalg.norm(exact-finite)/max(np.linalg.norm(exact), 1e-30)
        matrix = np.asarray(jac(fit.x))
        column_norm = np.linalg.norm(matrix, axis=0)
        singular = np.linalg.svd(matrix/np.maximum(column_norm, 1e-30), compute_uv=False)
        return best, dict(success=bool(fit.success), message=fit.message, nfev=fit.nfev,
                          optimality=float(fit.optimality), jacobian_relative_error=float(error),
                          optimized_parameters=[names[i] for i in np.asarray(indices)],
                          normalized_jacobian_singular_values=singular.tolist())

    # Search only the stellar RV, with fixed atmospheric/instrumental initialization.
    if not args.no_star and not args.initial:
        print("Scanning stellar velocity...", flush=True)
        trials = []
        for rv in np.arange(-40., 41., 4.):
            candidate = initial.copy(); candidate[star_rv] = rv
            residual = np.asarray(train_residual(candidate))
            trials.append((np.sum(residual ** 2), rv))
        initial[star_rv] = min(trials)[1]
        print("Initial stellar RV:", initial[star_rv], flush=True)
    if args.freeze_shared:
        best = initial.copy()
        training_fit = dict(previous_fit["training_fit"], reused=True, nfev=0,
                            source_path=str(args.initial), optimized_parameters=[],
                            reused_parameters=[names[i] for i in train_indices])
    else:
        best, training_fit = optimize(initial, train_indices, train_residual, "training")
    best, holdout_fit = optimize(best, heldout_indices, validation_residual, "heldout calibration")
    predicted, continuum, component = map(np.asarray, evaluate(jnp.asarray(best)))
    if not np.all(np.isfinite(predicted)) or np.any(continuum <= 0):
        raise ValueError("Nonfinite model or nonpositive fitted continuum.")
    normalized = (np.asarray(observed)-predicted)/continuum
    near_bound = (best-lower < .001*(upper-lower)) | (upper-best < .001*(upper-lower))
    result = dict(atmosphere=args.atmosphere, stellar_template=not args.no_star,
                  orders=args.orders, train_exposures=args.train, heldout_exposure=args.holdout,
                  pixels=args.pixels, layers=args.layers, dv_kms=args.dv, pressure_shift=args.shift,
                  reference_self_broadening=args.self_broadening,
                  pixel_mask=mask_metadata, fixed_parameters=[names[i] for i in fixed_indices],
                  ip=args.ip, ip_wing_width_ratio=2. if args.ip == "mixture" else None,
                  temperature_offset_K=args.temperature_offset,
                  temperature_node_pressures_bar=args.temperature_nodes, water_shape=args.water_shape,
                  pressure_scale=args.pressure_scale, line_cutoff=args.line_cutoff,
                  line_margin_cm1=args.line_margin, species=args.species,
                  continuum_degree=args.continuum_degree, quadratic_wavelength=args.quadratic_wavelength,
                  direct_ip=args.direct_ip, ip_wavelength=args.ip_wavelength,
                  ip_epoch=args.ip_epoch, ip_asymmetry=args.ip_asymmetry,
                  nonlinear_loss=args.loss, nonlinear_loss_scale=args.loss_scale,
                  parameters=dict(zip(names, best.tolist())), bounds_hit=[n for n,h in zip(names, near_bound) if h],
                  training_fit=training_fit, heldout_calibration_fit=holdout_fit,
                  training_metrics=metrics(normalized, training),
                  heldout_score_metrics=metrics(normalized, scoring),
                  training_fixed_scale_metrics=metrics(np.asarray(observed)-predicted, training),
                  heldout_fixed_scale_metrics=metrics(np.asarray(observed)-predicted, scoring),
                  profile_metadata=profile["metadata"], seconds=time.monotonic()-started,
                  used_pressure_boundaries_bar=np.asarray(profile["pressure_boundaries_bar"]).tolist(),
                  profile_metadata_note="Metadata describe the original reanalysis reference; actual scaled pressure boundaries and fitted columns are recorded separately.",
                  opacity_override=(dict(path=str(args.opacity_override), sha256=hashlib.sha256(args.opacity_override.read_bytes()).hexdigest())
                                    if args.opacity_override else None),
                  assumptions=["Fixed PHOENIX stellar Teff/logg/abundance and v sin i=4 km/s; Gaussian extra width",
                               "CSV relative RV has observer-frame sign, verified against Astropy",
                               "Reanalysis profile with optional smooth shared temperature/water-shape corrections; dry species shared, water per exposure",
                               "Self widths omitted by default; optional reference self widths frozen when scaling columns",
                               "No planetary model; no residual clipping; no pixel uncertainty array",
                               "Held-out exposure nuisance parameters and continua calibrated on alternate 80-pixel blocks with 5-pixel guards",
                               "Validation is conditional prediction, not an entirely unseen exposure",
                               "Polynomial continuum eliminated separately for each order/exposure",
                               "Profile redistribution preserves molecular column on fixed pressure layers; background mean mass fixed"])
    per_spectrum = []
    for b, block in enumerate(blocks):
        for e, exposure in enumerate(exposures):
            per_spectrum.append(dict(order=block["order"], exposure=exposure,
                                     **metrics(normalized[b,e], allowed[b]),
                                     score=metrics(normalized[b,e], scoring[b,e]) if e == len(exposures)-1 else None))
    result["per_spectrum"] = per_spectrum
    result["vertical_columns_cm-2"] = {
        s: [float(blocks[0]["columns"][j] * 10**best[water[e] if s == "H2O" else abundance[s]])
            for e in range(len(exposures))] for j,s in enumerate(args.species)}
    if dynamic_profile:
        delta, water_weights = profile_state(jnp.asarray(best))
        exact_taus = [jnp.stack([table.direct_tau(delta, water_weights if species == "H2O" else None)
                                for table, species in zip(block["tables"], args.species)]) for block in blocks]
        exact_prediction, _, exact_component = map(np.asarray, evaluate(jnp.asarray(best), exact_taus))
        difference = exact_component - component
        reference_water = profile["number_column_cm2"] * profile["vmr"]["H2O"]
        mass_column = profile["number_column_cm2"] * profile["mean_molecular_weight"]
        reference_dry = profile["number_column_cm2"] * (1-profile["vmr"]["H2O"])
        water_columns = np.asarray([reference_water*np.asarray(water_weights)*10**best[i] for i in water])
        dry_columns = (mass_column[None, :] - 18.01528*water_columns)/28.9647
        if np.any(dry_columns <= 0):
            raise ValueError("Water redistribution exceeds the hydrostatic mass budget.")
        result["fitted_profile"] = dict(
            pressure_bar=np.asarray(profile["pressure_bar"]).tolist(),
            temperature_K=(np.asarray(profile["temperature_K"])+np.asarray(delta)).tolist(),
            temperature_delta_K=np.asarray(delta).tolist(), water_layer_weights=np.asarray(water_weights).tolist(),
            water_layer_columns_cm2=water_columns.tolist(),
            mass_conserving_water_vmr=(water_columns/(water_columns+dry_columns)).tolist(),
            mass_conserving_dry_column_ratio=(dry_columns/reference_dry).tolist(),
            max_dry_column_fractional_change=float(np.max(np.abs(dry_columns/reference_dry-1))),
            total_water_column_conservation_relative_error=float(
                np.sum(reference_water*np.asarray(water_weights))/np.sum(reference_water)-1))
        result["opacity_interpolation_check"] = dict(
            rms_component_difference=float(np.sqrt(np.mean(difference**2))),
            max_component_difference=float(np.max(np.abs(difference))),
            exact_training_fixed_scale_metrics=metrics(np.asarray(observed)-exact_prediction, training),
            exact_heldout_fixed_scale_metrics=metrics(np.asarray(observed)-exact_prediction, scoring))
        result["profile_opacity_tables"] = [[table.metadata for table in block["tables"]] for block in blocks]
    if args.check_layers and args.atmosphere == "profile":
        convergence = {}
        for layers in [10, 20, 40]:
            other = load_profile(layers)
            other["temperature_K"] = other["temperature_K"] + args.temperature_offset
            for key in ["pressure_bar", "delta_pressure_bar", "pressure_boundaries_bar", "number_column_cm2"]:
                other[key] = other[key]*args.pressure_scale
            taus = [jnp.stack([profile_tau(opa, b["grid"], other, s, args.shift, args.self_broadening)[0]
                              for opa,s in zip(b["opacities"], args.species)]) for b in blocks]
            alternative_component = np.asarray(evaluate(jnp.asarray(best), taus)[2])
            difference = alternative_component - component
            convergence[str(layers)] = {"rms_model_difference": float(np.sqrt(np.mean(difference**2))),
                                         "max_model_difference": float(np.max(np.abs(difference)))}
        result["layer_convergence"] = convergence
    tag = args.tag or f"joint_{'nostar' if args.no_star else 'star'}_{args.atmosphere}"
    out = ROOT / "results"
    result["provenance"] = dict(
        fit_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        initial_path=str(args.initial) if args.initial else None,
        initial_sha256=hashlib.sha256(args.initial.read_bytes()).hexdigest() if args.initial else None,
        command_arguments={key: str(value) if isinstance(value, Path) else value for key,value in vars(args).items()})
    (out/f"{tag}.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(out/f"{tag}.npz", wavelength_um=np.stack([b["wave"] for b in blocks]),
                        observed=np.asarray(observed), model=predicted, continuum=continuum,
                        component=component, residual_continuum=normalized,
                        train_mask=training, calibration_mask=calibration, score_mask=scoring,
                        included_mask=allowed,
                        exposures=exposures, orders=args.orders, parameters=best)
    fig, axes = plt.subplots(len(blocks), 2, figsize=(14, 3*len(blocks)), squeeze=False)
    for b, block in enumerate(blocks):
        axes[b,0].plot(block["wave"], np.asarray(observed)[b,0]/continuum[b,0], color="0.3", lw=.7, label="Data / continuum")
        axes[b,0].plot(block["wave"], component[b,0], color="tab:orange", lw=.8, label="Stellar x telluric, then IP")
        axes[b,0].set_title(f"Order {block['order']}, training exposure {exposures[0]}")
        axes[b,0].set_ylabel("Normalized flux")
        axes[b,1].plot(block["wave"], 100*normalized[b,-1], color="0.65", lw=.7)
        axes[b,1].plot(block["wave"][scoring[b,-1]], 100*normalized[b,-1,scoring[b,-1]], ".", color="tab:blue", ms=2, label="Held-out scoring pixels")
        axes[b,1].axhline(0, color="tab:orange", lw=.7)
        axes[b,1].set_title(f"Conditional prediction, exposure {exposures[-1]}")
        axes[b,1].set_ylabel("Residual (%)")
        for ax in axes[b]:
            ax.set_xlabel("Vacuum wavelength (micron)")
    axes[0,0].legend(fontsize=8); axes[0,1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out/f"{tag}.png", dpi=150); plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
