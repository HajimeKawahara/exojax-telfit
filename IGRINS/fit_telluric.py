"""Fit a small IGRINS order segment with existing ExoJAX Voigt profiles.

Run from any directory: python IGRINS/fit_telluric.py --order 34
This is a telluric feasibility experiment, not a planetary retrieval.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import Patm, c

from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent
COLUMNS = {"H2O": 1e22, "CO2": 6e21, "CH4": 3e19, "N2O": 5e18, "CO": 2e18}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=34)
    parser.add_argument("--exposure", type=int, default=0)
    parser.add_argument("--pixels", type=int, nargs=2, default=[600, 1500])
    parser.add_argument("--dv", type=float, default=0.75, help="Model step in km/s")
    parser.add_argument("--shift", action="store_true", help="Apply HITRAN air pressure shifts")
    parser.add_argument("--ip", choices=["gaussian", "mixture"], default="gaussian")
    parser.add_argument("--optimizer", choices=["trf", "lm"], default="trf")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--initial", type=Path, help="Warm-start from a previous result JSON")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()
    start = time.monotonic()
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    if not (0 <= args.order < len(wavelength) and 0 <= args.exposure < cube.shape[1]):
        raise ValueError("Order or exposure is outside the data cube.")
    lo, hi = args.pixels
    if not (0 <= lo < hi <= cube.shape[2]) or args.dv <= 0:
        raise ValueError("Invalid pixel interval or model spacing.")
    wave = wavelength[args.order, lo:hi][::-1].copy()
    flux = cube[args.order, args.exposure, lo:hi][::-1].copy()
    valid = np.isfinite(wave) & np.isfinite(flux) & (wave > 0)
    wave, flux = wave[valid], flux[valid]
    nu = 1e4 / wave  # Assume vacuum microns; no barycentric correction applied.
    scale = float(np.percentile(flux, 95))
    y = flux / scale
    x = np.linspace(-1, 1, len(y))
    basis = jnp.asarray(np.polynomial.chebyshev.chebvander(x, 3))
    pad = 200.0  # km/s; exceeds calibration bounds plus IP kernel support.
    nmin, nmax = nu[0] * np.exp(-pad / c), nu[-1] * np.exp(pad / c)
    ng = int(np.ceil(np.log(nmax / nmin) * c / args.dv)) + 1
    ng += ng % 2
    grid = np.geomspace(nmin, nmax, ng)
    dv = float(c * np.log(grid[1] / grid[0]))
    kernel_velocity = jnp.arange(-int(80 / dv), int(80 / dv) + 1) * dv
    calculators = []
    for name, column in COLUMNS.items():
        db = load_mdb(name, grid[0] - 3, grid[-1] + 3, crit=1e-28)
        if not len(db.nu_lines):
            continue
        opa = OpaDirect(db, grid)
        initial_tau = np.asarray(opa.xsvector(270., 0.55)) * column
        if np.max(initial_tau) < 0.002:
            continue
        calculators.append((name, opa))
    names = [name for name, _ in calculators]
    print("Selected species:", [(n, len(o.mdb.nu_lines)) for n, o in calculators], flush=True)
    if not names:
        raise ValueError("No relevant telluric lines in this interval.")
    parameter_names = ["temperature_K", "pressure_bar", "ip_sigma_kms", "rv_kms", "stretch_kms"]
    initial = [270., .55, 2.83, 0., 0.]
    lower = [180., .03, 1., -100., -10.]
    upper = [330., 1.1, 8., 100., 10.]
    for name in names:
        parameter_names.append("log10_column_" + name)
        initial.append(np.log10(COLUMNS[name]))
        lower.append(np.log10(COLUMNS[name]) - 2)
        upper.append(np.log10(COLUMNS[name]) + 2)
    ncol = len(names)
    parameter_names += ["continuum_c0", "continuum_c1", "continuum_c2", "continuum_c3"]
    initial += [np.log(max(np.median(y), .01)), 0., 0., 0.]
    lower += [-3., -3., -3., -3.]
    upper += [2., 3., 3., 3.]
    if args.ip == "mixture":
        parameter_names.append("ip_wing_fraction")
        initial.append(.05)
        lower.append(0.)
        upper.append(.5)
    if args.initial:
        previous = json.loads(args.initial.read_text())["parameters"]
        initial = [previous.get(k, v) for k, v in zip(parameter_names, initial)]
    lower, upper = np.asarray(lower), np.asarray(upper)
    initial = np.clip(initial, lower + 1e-7, upper - 1e-7)

    def model(p):
        temperature, pressure, beta, rv, stretch = p[:5]
        tau = jnp.zeros(len(grid))
        for i, (_, opa) in enumerate(calculators):
            if args.shift:
                db = opa.mdb
                strength = line_strength(temperature, db.logsij0, db.nu_lines,
                                         db.elower, db.qr_interp_lines(temperature, 296.), 296.)
                width = gamma_hitran(pressure, temperature, 0., db.n_air,
                                     db.gamma_air, db.gamma_self) + gamma_natural(db.A)
                sigma = doppler_sigma(db.nu_lines, temperature, db.molmass)
                xs = xsvector(opa.opainfo - (db.delta_air * pressure / Patm)[:, None],
                              sigma, width, strength)
            else:
                xs = opa.xsvector(temperature, pressure)
            tau = tau + 10. ** p[5 + i] * xs
        transmission = jnp.exp(-tau)
        broadened = ipgauss(transmission, kernel_velocity, beta)
        if args.ip == "mixture":
            broadened = (1 - p[-1]) * broadened + p[-1] * ipgauss(transmission, kernel_velocity, 2 * beta)
        observed_transmission = sampling(nu * (1 + stretch * x / c), grid, broadened, rv)
        continuum = jnp.exp(basis @ p[5 + ncol:9 + ncol])
        return continuum * observed_transmission, continuum, observed_transmission

    forward = jax.jit(lambda p: model(p)[0])
    residual = jax.jit(lambda p: forward(p) - y)
    jacobian = jax.jit(jax.jacfwd(residual))
    print("Compiling model and exact Jacobian...", flush=True)
    residual(jnp.asarray(initial)).block_until_ready()
    jacobian(jnp.asarray(initial)).block_until_ready()
    compilation_seconds = time.monotonic() - start
    # A coarse RV scan avoids selecting the wrong minimum from a zero RV guess.
    if not args.initial:
        trials = []
        for rv in np.arange(-90., 91., 3.):
            p = initial.copy()
            p[3] = rv
            # Refit only the continuum offset for this inexpensive initialization.
            m = np.asarray(forward(p))
            p[5 + ncol] += np.log(max(np.dot(y, m) / np.dot(m, m), .01))
            trials.append((float(np.sum(np.asarray(residual(p)) ** 2)), p))
        initial = min(trials, key=lambda pair: pair[0])[1]
        initial = np.clip(initial, lower + 1e-7, upper - 1e-7)
    evaluations = 0

    def fun(p):
        nonlocal evaluations
        evaluations += 1
        r = np.asarray(residual(p))
        if evaluations % 10 == 0:
            print(f"Evaluation {evaluations}: scaled RMS={np.sqrt(np.mean(r*r)):.6f}", flush=True)
        return r

    opt_start = time.monotonic()
    if args.optimizer == "lm":
        span = upper - lower
        def physical(z):
            return lower + span * expit(z)
        def transformed_jac(z):
            s = expit(z)
            return np.asarray(jacobian(physical(z))) * (span * s * (1 - s))[None, :]
        z0 = logit(np.clip((initial - lower) / span, 1e-5, 1 - 1e-5))
        fit = least_squares(lambda z: fun(physical(z)), z0, jac=transformed_jac,
                            method="lm", x_scale="jac", max_nfev=args.max_nfev,
                            ftol=1e-8, xtol=1e-8, gtol=1e-8)
        best = physical(fit.x)
    else:
        fit = least_squares(fun, initial, jac=lambda p: np.asarray(jacobian(p)),
                            bounds=(lower, upper), method="trf", x_scale="jac",
                            max_nfev=args.max_nfev, ftol=1e-8, xtol=1e-8, gtol=1e-8)
        best = fit.x
    prediction, continuum, transmission = map(np.asarray, model(jnp.asarray(best)))
    r = (y - prediction) / continuum
    jac = np.asarray(jacobian(best))
    # Check derivatives on the actual data/model, not on a duplicate implementation.
    direction = np.random.default_rng(2026).normal(size=len(best))
    direction *= np.maximum(np.abs(best), 1.)
    epsilon = 1e-5
    fd = (np.asarray(forward(best + epsilon * direction)) - np.asarray(forward(best - epsilon * direction))) / (2 * epsilon)
    exact = jac @ direction
    derivative_error = float(np.linalg.norm(fd - exact) / max(np.linalg.norm(exact), 1e-30))
    metadata = pd.read_csv(ROOT / "data/20201214_info.csv").iloc[args.exposure].to_dict()
    near_bound = (best - lower < .001 * (upper - lower)) | (upper - best < .001 * (upper - lower))
    result = {
        "order_index": args.order, "exposure_index": args.exposure, "pixels": args.pixels,
        "wavelength_range_um": [float(wave.min()), float(wave.max())],
        "pressure_shift": args.shift, "ip": args.ip, "optimizer": args.optimizer,
        "grid_dv_kms": dv, "initial_result": str(args.initial) if args.initial else None,
        "parameters": dict(zip(parameter_names, best.tolist())),
        "bounds_hit": [n for n, hit in zip(parameter_names, near_bound) if hit],
        "success": bool(fit.success), "message": fit.message, "nfev": fit.nfev,
        "optimality": float(fit.optimality), "n_pixels": len(y),
        "rms_continuum": float(np.sqrt(np.mean(r*r))),
        "mad_continuum": float(1.4826 * np.median(np.abs(r - np.median(r)))),
        "p95_absolute_continuum": float(np.percentile(np.abs(r), 95)),
        "lag1_correlation": float(np.corrcoef(r[:-1], r[1:])[0, 1]),
        "rms_fixed_scale": float(np.sqrt(np.mean((y - prediction)**2))),
        "sum_squared_residual_fixed_scale": float(np.sum((y - prediction)**2)),
        "scale_flux": scale, "metadata": metadata,
        "nominal_inverse_metadata_snr": float(1 / metadata["Med. SNR"]),
        "jacobian_directional_relative_error": derivative_error,
        "compilation_seconds": compilation_seconds,
        "optimization_seconds": time.monotonic() - opt_start,
        "line_counts": {n: len(o.mdb.nu_lines) for n, o in calculators},
        "assumptions": ["Vacuum micron wavelengths; free calibration offset and stretch",
                        "Single effective atmospheric slab; zero molecular self-pressure",
                        "No stellar or planet spectrum; no pixel uncertainty array",
                        "Smooth continuum applied after IP convolution",
                        "No residual clipping; all finite pixels in chosen window fitted"],
    }
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    tag = args.tag or f"o{args.order:02d}_e{args.exposure:02d}_{'shift' if args.shift else 'baseline'}_{args.ip}_{args.optimizer}"
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(out / f"{tag}.npz", wavelength_um=wave, flux=flux,
                        model=prediction*scale, continuum=continuum*scale,
                        transmission=transmission, residual_continuum=r,
                        initial_parameters=initial, best_parameters=best)
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True, height_ratios=[3, 1])
    axes[0].plot(wave, y / continuum, color="0.3", lw=.65, label="Data / fitted continuum")
    axes[0].plot(wave, transmission, color="tab:orange", lw=.9, label="ExoJAX telluric model")
    axes[0].set_ylabel("Normalized flux")
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"WASP-77A | order {args.order}, exposure {args.exposure} | {args.ip}, shift={args.shift} | RMS={100*result['rms_continuum']:.2f}%")
    axes[1].plot(wave, 100*r, color="0.3", lw=.6)
    axes[1].axhline(0, color="tab:orange", lw=.7)
    axes[1].set_ylabel("Residual (%)")
    axes[1].set_xlabel("Wavelength (micron; vacuum assumed)")
    fig.tight_layout()
    fig.savefig(out / f"{tag}.png", dpi=160)
    plt.close(fig)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
