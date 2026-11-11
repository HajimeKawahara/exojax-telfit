"""Transfer the frozen telluric model across discovery exposures.

The reserved exposures are never selected for fitting, plotting, or statistics.
This is an in-sample diagnostic of time dependence, not held-out validation.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path
import time

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
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
RESERVED = [10, 30, 50, 70]


def correlation(x, y):
    """Describe associations without treating correlated pixels as trials."""
    return dict(pearson=float(pearsonr(x, y).statistic),
                spearman=float(spearmanr(x, y).statistic))


def main():
    started = time.monotonic()
    source_path = OUT / "focus_ip.json"
    source = json.loads(source_path.read_text())
    p = source["parameters"]
    orders = np.asarray(source["orders"])
    lo, hi = source["pixels"]
    exposures = np.asarray([e for e in range(79) if e not in RESERVED])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    # Select discovery exposures before any flux normalization or statistics.
    flux = cube[orders[:, None], exposures[None, :], lo:hi][:, :, ::-1].copy()
    del cube
    with np.load(OUT / "telluric_mask.npz") as saved:
        np.testing.assert_array_equal(saved["orders"], orders)
        allowed = saved["mask"].copy()
        wave = saved["wavelength_um"].copy()
    scales = np.stack([np.percentile(f[:, m], 95, axis=1)
                       for f, m in zip(flux, allowed)])
    observed = flux / scales[:, :, None]
    profile = load_profile(source["layers"])
    basis = jnp.asarray(np.polynomial.chebyshev.chebvander(
        np.linspace(-1, 1, hi-lo), 3))
    weights = jnp.asarray(allowed, dtype=jnp.float64)
    x = jnp.linspace(-1, 1, hi-lo)
    blocks = []
    for b, order in enumerate(orders):
        nu = 1e4 / wave[b]
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/source["dv_kms"])) + 1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, p["stellar_extra_sigma_kms"])
        tau = {}
        for species in ["H2O", "CO2"]:
            db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
            opa = OpaDirect(db, grid)
            tau[species] = jnp.asarray(profile_tau(
                opa, grid, profile, species, source["pressure_shift"],
                source["reference_self_broadening"])[0])
        blocks.append(dict(grid=grid, velocity=velocity, star=star, tau=tau,
                           calibrated_nu=nu*(1+p[f"stretch_o{order}_kms"]*x/c),
                           sigma=p[f"ip_sigma_o{order}_kms"],
                           rv=p[f"calibration_rv_o{order}_kms"]))

    def evaluate(q, y, airmass, relative_rv):
        models, continua, parts = [], [], []
        for b, block in enumerate(blocks):
            shifted = sampling(block["grid"], block["grid"], block["star"],
                               p["stellar_rv_reference_kms"]+relative_rv)
            tau = (10**q[0]*block["tau"]["H2O"]
                   + 10**p["log10_scale_CO2"]*block["tau"]["CO2"])
            intrinsic = shifted*jnp.exp(-airmass*tau)
            sigma = block["sigma"]*jnp.exp(q[2])
            fraction = p["ip_wing_fraction"]
            broadened = ((1-fraction)*ipgauss(intrinsic, block["velocity"], sigma)
                         + fraction*ipgauss(intrinsic, block["velocity"], 2*sigma))
            sampled = sampling(block["calibrated_nu"], block["grid"], broadened,
                               block["rv"]+q[1])
            design = basis*sampled[:, None]
            normal = design.T@(weights[b, :, None]*design)
            coefficients = jnp.linalg.solve(normal+1e-12*jnp.eye(4),
                                             design.T@(weights[b]*y[b]))
            continuum = basis@coefficients
            models.append(sampled*continuum)
            continua.append(continuum)
            parts.append(sampled)
        return jnp.stack(models), jnp.stack(continua), jnp.stack(parts)

    forward = jax.jit(evaluate)
    residual = jax.jit(lambda q, y, am, rv: (evaluate(q, y, am, rv)[0]-y)[allowed])
    jacobian = jax.jit(jax.jacfwd(residual, argnums=0))
    am = info.iloc[exposures]["Air Mass"].to_numpy()
    relative_rv = (info.iloc[exposures]["RV [km/s]"].to_numpy()
                   - info.iloc[0]["RV [km/s]"])
    reference_q = np.array([p["log10_water_e0"], 0., 0.])
    reconstructed = np.asarray(forward(reference_q, observed[:, 0], am[0], relative_rv[0])[2])
    with np.load(OUT / "focus_ip.npz") as saved:
        reconstruction_error = float(np.max(np.abs(reconstructed-saved["component"][:, 0])))
    if reconstruction_error > 1e-11:
        raise ValueError(f"Model reconstruction mismatch: {reconstruction_error}")
    print("Compiling derivatives...", flush=True)
    exact_jacobian = np.asarray(jacobian(reference_q, observed[:, 0], am[0], relative_rv[0]))
    direction = np.array([.03, .5, .05])
    step = 1e-6
    finite = (np.asarray(residual(reference_q+step*direction, observed[:, 0], am[0], relative_rv[0]))
              - np.asarray(residual(reference_q-step*direction, observed[:, 0], am[0], relative_rv[0])))/(2*step)
    exact = exact_jacobian@direction
    jacobian_error = float(np.linalg.norm(exact-finite)/np.linalg.norm(exact))
    if jacobian_error > 1e-3:
        raise ValueError(f"Jacobian mismatch: {jacobian_error}")
    output, records = {}, []
    for variable_ip in [False, True]:
        label = "variable_ip" if variable_ip else "fixed_ip"
        model = np.empty_like(observed)
        continuum = np.empty_like(observed)
        parameters = np.empty((len(exposures), 3))
        count = 3 if variable_ip else 2
        lower = np.array([-1.5, -2., np.log(.7)])[:count]
        upper = np.array([1.5, 2., np.log(1.3)])[:count]
        for e, exposure in enumerate(exposures):
            start = (output["fixed_ip"]["parameters"][e].copy() if variable_ip
                     else np.array([np.interp(exposure, [0, 20, 78],
                         [p["log10_water_e0"], p["log10_water_e20"], p["log10_water_e78"]]), 0., 0.]))
            def unpack(q):
                return jnp.asarray(q) if variable_ip else jnp.r_[jnp.asarray(q), 0.]
            def fun(q):
                return np.asarray(residual(unpack(q), observed[:, e], am[e], relative_rv[e]))
            def jac(q):
                return np.asarray(jacobian(unpack(q), observed[:, e], am[e], relative_rv[e]))[:, :count]
            fit = least_squares(fun, start[:count], jac=jac, bounds=(lower, upper),
                                x_scale="jac", max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-7)
            q = np.asarray(unpack(fit.x))
            prediction, cont, _ = map(np.asarray, forward(q, observed[:, e], am[e], relative_rv[e]))
            model[:, e], continuum[:, e], parameters[e] = prediction, cont, q
            rms = float(np.sqrt(np.mean(((observed[:, e]-prediction)[allowed])**2)))
            records.append(dict(model=label, exposure=int(exposure), success=bool(fit.success),
                                nfev=int(fit.nfev), optimality=float(fit.optimality),
                                log10_water=float(q[0]), calibration_rv_kms=float(q[1]),
                                ip_width_scale=float(np.exp(q[2])), fixed_scale_rms=rms))
            if e % 15 == 0:
                print(label, int(exposure), rms, np.exp(q[2]), flush=True)
        output[label] = dict(model=model, continuum=continuum, parameters=parameters)
    fixed_residual = (observed-output["fixed_ip"]["model"])/output["fixed_ip"]["continuum"]
    variable_residual = (observed-output["variable_ip"]["model"])/output["fixed_ip"]["continuum"]
    slant_water = am*10**output["fixed_ip"]["parameters"][:, 0]
    ip_scale = np.exp(output["variable_ip"]["parameters"][:, 2])
    feature_rows = []
    features = [(40, 1.501046034), (40, 1.494855416), (34, 1.570498667),
                (13, 2.076825837), (40, 1.496697779), (34, 1.572284718)]
    for order, target in features:
        b = int(np.flatnonzero(orders == order)[0])
        k = int(np.argmin(abs(wave[b]-target)))
        values = np.mean(fixed_residual[b, :, k-1:k+2], axis=1)
        flexible = np.mean(variable_residual[b, :, k-1:k+2], axis=1)
        slope, intercept = np.polyfit(slant_water, values, 1)
        feature_rows.append(dict(order=order, wavelength_um=float(wave[b, k]),
                                 original_pixel=int(hi-1-k),
                                 mean_residual_percent=float(100*np.mean(values)),
                                 scatter_residual_percent=float(100*np.std(values)),
                                 variable_ip_mean_residual_percent=float(100*np.mean(flexible)),
                                 slant_water_slope_percent=float(100*slope),
                                 correlations=dict(airmass=correlation(am, values),
                                                   slant_water=correlation(slant_water, values),
                                                   ip_scale=correlation(ip_scale, values)),
                                 residual_percent=(100*values).tolist(),
                                 variable_ip_residual_percent=(100*flexible).tolist()))
    statistics = {}
    for label, data in output.items():
        selected = np.broadcast_to(allowed[:, None], observed.shape)
        residual_flux = observed-data["model"]
        statistics[label] = dict(rms_percent=float(100*np.sqrt(np.mean(residual_flux[selected]**2))),
                                 order_rms_percent=[float(100*np.sqrt(np.mean(residual_flux[b, :, allowed[b]]**2)))
                                                    for b in range(len(orders))])
    result = dict(scope="Frozen shared focus_ip model transferred to 75 discovery exposures",
                  reserved_exposures_not_selected=RESERVED, exposures=exposures.tolist(),
                  reconstruction_max_error=reconstruction_error, statistics=statistics,
                  jacobian_relative_error=jacobian_error,
                  slant_water_range=[float(slant_water.min()), float(slant_water.max())],
                  ip_width_scale_range=[float(ip_scale.min()), float(ip_scale.max())],
                  ip_width_scale_correlations=dict(airmass=correlation(am, ip_scale),
                                                   time=correlation(exposures, ip_scale)),
                  features=feature_rows, fits=records,
                  assumptions=["All allowed pixels calibrate per-exposure parameters; this is not validation",
                               "No reserved exposure contributes to flux statistics or plots",
                               "Variable IP is one Gaussian-width multiplier shared across orders per exposure",
                               "IP wing fraction, molecular profile and stellar model stay fixed",
                               "Feature residual is the mean of three pixels, not an equivalent-width measurement",
                               "Time, airmass and water column covary; correlations do not uniquely identify a cause",
                               "No error bars supplied; RMS differences are descriptive"],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [source_path, OUT/"telluric_mask.npz", Path(__file__)]},
                  seconds=time.monotonic()-started)
    (OUT/"transfer_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    pd.DataFrame(records).to_csv(OUT/"transfer_fits.csv", index=False)
    np.savez_compressed(OUT/"transfer_diagnostics.npz", orders=orders, exposures=exposures,
                        wavelength_um=wave, observed=observed, allowed=allowed, airmass=am,
                        slant_water=slant_water, fixed_residual=fixed_residual,
                        variable_residual=variable_residual, ip_width_scale=ip_scale,
                        **{f"{label}_{key}": value for label, data in output.items() for key, value in data.items()})
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), constrained_layout=True)
    for ax, feature in zip(axes.flat, feature_rows):
        ax.scatter(slant_water, feature["residual_percent"], c=exposures, cmap="viridis", s=14,
                   label="Fixed IP")
        ax.scatter(slant_water, feature["variable_ip_residual_percent"], c="tab:orange", s=10,
                   marker="x", label="Per-exposure IP width")
        ax.axhline(0, color=".6", lw=.7)
        ax.set(xlabel="Fitted slant H2O column / reference vertical column", ylabel="Residual (%)",
               title=f"Order {feature['order']} | {feature['wavelength_um']:.6f} micron")
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Residual transfer across 75 discovery exposures; color = exposure index")
    fig.savefig(OUT/"transfer_features.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    axes[0].plot(exposures, slant_water, ".", label="Slant H2O")
    axes[0].plot(exposures, am, ".", label="Airmass")
    axes[0].set_ylabel("Column scale / airmass")
    axes[0].legend()
    axes[1].plot(exposures, ip_scale, ".")
    axes[1].axhline(1, color=".5", lw=.7)
    axes[1].set_ylabel("IP width multiplier")
    for label in ["fixed_ip", "variable_ip"]:
        selected = [row for row in records if row["model"] == label]
        axes[2].plot(exposures, [100*row["fixed_scale_rms"] for row in selected], ".", label=label)
    axes[2].set(xlabel="Exposure index (10, 30, 50, 70 reserved)", ylabel="Fixed-scale RMS (%)")
    axes[2].legend()
    fig.savefig(OUT/"transfer_time.png", dpi=150)
    plt.close(fig)
    print(json.dumps({key: result[key] for key in ["statistics", "ip_width_scale_range", "seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
