"""Estimate short-time residual scatter under the frozen current physical model.

The evaluator is adapted from diagnose_transfer.py and checked against the
source prediction. Reserved exposures are excluded before flux statistics.
Adjacent differences are an empirical scatter proxy, not a pure-noise estimate.
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


def main():
    started = time.monotonic()
    source_path = OUT / "motion_clean_control.json"
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
    with np.load(OUT / "telluric_mask_motion_clean.npz") as saved:
        np.testing.assert_array_equal(saved["orders"], orders)
        allowed = saved["mask"].copy()
        wave = saved["wavelength_um"].copy()
        reference_depth = saved["telluric_depth"].copy()
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
    with np.load(OUT / "motion_clean_control.npz") as saved:
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
    model = np.empty_like(observed)
    continuum = np.empty_like(observed)
    parameters = np.empty((len(exposures), 3))
    records = []
    reference_exposures = sorted(source["train_exposures"] + [source["heldout_exposure"]])
    for e, exposure in enumerate(exposures):
        initial = np.asarray([np.interp(exposure, reference_exposures,
            [p.get(name.format(e=k), 0.) for k in reference_exposures])
            for name in ["log10_water_e{e}", "calibration_rv_e{e}_kms", "log_ip_width_e{e}"]])

        def fun(q):
            return np.asarray(residual(q, observed[:, e], am[e], relative_rv[e]))

        def jac(q):
            return np.asarray(jacobian(q, observed[:, e], am[e], relative_rv[e]))

        fit = least_squares(fun, initial, jac=jac,
                            bounds=([-1.5, -2., -.15], [1.5, 2., .15]),
                            x_scale="jac", max_nfev=60, ftol=1e-8, xtol=1e-8, gtol=1e-7)
        prediction, cont, _ = map(np.asarray, forward(fit.x, observed[:, e], am[e], relative_rv[e]))
        model[:, e], continuum[:, e], parameters[e] = prediction, cont, fit.x
        records.append(dict(exposure=int(exposure), success=bool(fit.success),
                            nfev=int(fit.nfev), optimality=float(fit.optimality),
                            log10_water=float(fit.x[0]), calibration_rv_kms=float(fit.x[1]),
                            ip_width_scale=float(np.exp(fit.x[2])),
                            rms_percent=float(100*np.sqrt(np.mean(fun(fit.x)**2)))))
        if e % 15 == 0:
            print(int(exposure), records[-1]["rms_percent"], flush=True)
    if not all(row["success"] for row in records) or np.any(continuum <= 0):
        raise ValueError("Unsuccessful fit or nonpositive continuum")
    pair_left = np.flatnonzero(np.diff(exposures) == 1)
    pair_right = pair_left + 1
    assert len(pair_left) == 70
    assert not np.isin(exposures, RESERVED).any()
    residual_flux = observed-model
    residual_continuum = residual_flux/continuum
    time_minutes = (info.iloc[exposures]["Time (BJD)"].to_numpy()-info.iloc[0]["Time (BJD)"])*1440
    # These fixed reference depths are selection metadata, not fitted new masks.
    bins = dict(all=allowed,
                weak_reference_depth_lt_1pct=allowed & (reference_depth < .01),
                reference_depth_1_to_2pct=allowed & (reference_depth >= .01) & (reference_depth < .02),
                line_reference_depth_ge_2pct=allowed & (reference_depth >= .02))
    category_path = OUT/"nonstellar_spectral_selection.npz"
    with np.load(category_path) as saved:
        np.testing.assert_array_equal(saved["orders"], orders)
        np.testing.assert_array_equal(saved["category"] >= 0, allowed)
        for index, name in enumerate(["H2O_dominant", "CO2_dominant", "mixed", "weak_line_or_continuum"]):
            bins[name] = saved["category"] == index
    cadence = time_minutes[pair_right]-time_minutes[pair_left]
    short_pairs = cadence < 1.5*np.median(cadence)
    statistics = {}
    for name, values in [("fixed_flux_scale", residual_flux), ("continuum_normalized", residual_continuum)]:
        statistics[name] = {}
        for group, selected in bins.items():
            statistics[name][group] = describe(values, selected, pair_left, pair_right)
            statistics[name][group]["by_order"] = {
                str(order): describe(values[b:b+1], selected[b:b+1], pair_left, pair_right)
                for b, order in enumerate(orders)}
        statistics[name]["cadence_filtered_all"] = describe(
            values, allowed, pair_left[short_pairs], pair_right[short_pairs])
    snr = info.iloc[exposures]["Med. SNR"].to_numpy()
    reciprocal_snr = 100/snr
    result = dict(
        source="motion_clean_control",
        scope="Descriptive refit of 75 non-reserved exposures with fixed shared parameters and mask",
        reserved_exposures_excluded=RESERVED,
        exposures=exposures.tolist(), orders=orders.tolist(),
        retained_pixels_per_order=allowed.sum(axis=1).tolist(),
        adjacent_native_pairs=exposures[np.c_[pair_left, pair_right]].tolist(),
        adjacent_time_separation_minutes={"median": float(np.median(time_minutes[pair_right]-time_minutes[pair_left])),
                                          "range": [float(np.min(time_minutes[pair_right]-time_minutes[pair_left])),
                                                    float(np.max(time_minutes[pair_right]-time_minutes[pair_left]))]},
        cadence_filtered_pair_count=int(np.sum(short_pairs)),
        reconstruction_max_error=reconstruction_error, jacobian_relative_error=jacobian_error,
        statistics=statistics, fits=records,
        info_reciprocal_median_snr_percent={"median": float(np.median(reciprocal_snr)),
                                           "range": [float(reciprocal_snr.min()), float(reciprocal_snr.max())]},
        definitions={
            "difference_proxy": "(r[e+1]-r[e])/sqrt(2), only native adjacent exposure indices",
            "temporal_scatter": "r minus its 75-exposure mean at each retained pixel",
            "robust_sigma": "1.4826 times median absolute deviation around the pooled median",
            "temporal_lag1_correlation": "Pearson correlation of per-pixel time-demeaned residuals in native adjacent pairs",
            "spectral_lag1_correlation": "Pearson correlation over neighboring detector pixels retained in the same bin; no gaps bridged",
            "normalization": "Fixed flux scale is each exposure/order's observed retained-pixel 95th percentile; continuum normalized also divides by its fitted continuum",
            "reference_depth_bins": "Original mask metadata telluric_depth, fixed before this analysis; bins partition all retained pixels"},
        limitations=[
            "Per-exposure water, RV, common IP-width multiplier and cubic continua use all retained pixels; this is not held-out validation",
            "Exposure zero also has its three local nuisance parameters refitted with shared parameters fixed",
            "The shared source model already used five discovery exposures and development exposure 20",
            "Adjacent differences contain short-timescale atmosphere, calibration and extraction changes as well as measurement noise",
            "Temporal correlations can suppress differences; difference RMS is not a pure-noise estimate or a strict noise upper bound",
            "Nuisance fitting removes 15 local modes per 1467-pixel exposure and slightly suppresses fitted residual scatter",
            "The scalar metadata SNR has no order/pixel error model and only permits a rough scale comparison",
            "Static and time-varying residuals are statistical components, not uniquely identified physical causes",
            "No temporal resampling, residual clipping, new stellar mask or reserved-exposure analysis is used"],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [source_path, OUT/"telluric_mask_motion_clean.npz", category_path, Path(__file__)]},
        seconds=time.monotonic()-started)
    (OUT/"nonstellar_noise.json").write_text(json.dumps(result, indent=2)+"\n")
    pd.DataFrame(records).to_csv(OUT/"nonstellar_noise_fits.csv", index=False)
    np.savez_compressed(OUT/"nonstellar_noise.npz", orders=orders, exposures=exposures,
                        wavelength_um=wave, allowed=allowed, observed=observed, model=model,
                        continuum=continuum, parameters=parameters, residual_fixed_scale=residual_flux,
                        residual_continuum=residual_continuum, pair_left=pair_left, pair_right=pair_right,
                        time_minutes=time_minutes, reference_depth=reference_depth)
    figure, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    differences = (residual_flux[:, pair_right]-residual_flux[:, pair_left])/np.sqrt(2)
    for b, ax in enumerate(axes):
        ax.plot(time_minutes, 100*np.sqrt(np.mean(residual_flux[b][:, allowed[b]]**2, axis=1)),
                ".-", ms=3, lw=.8, label="Residual RMS")
        ax.plot((time_minutes[pair_left]+time_minutes[pair_right])/2,
                100*np.sqrt(np.mean(differences[b][:, allowed[b]]**2, axis=1)),
                ".-", ms=3, lw=.8, label="Adjacent difference / sqrt(2)")
        ax.plot(time_minutes, reciprocal_snr, color=".5", lw=.8, ls="--",
                label="1 / metadata median SNR (rough scale only)")
        ax.set(ylabel="Fixed-scale RMS (%)", title=f"Order {orders[b]}")
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Time since first exposure (minutes)")
    figure.suptitle("Current model; 75 non-reserved exposures; short-time scatter is not pure noise")
    figure.savefig(OUT/"nonstellar_noise.png", dpi=150)
    plt.close(figure)
    print(json.dumps({"all_fixed_scale": statistics["fixed_flux_scale"]["all"],
                      "seconds": result["seconds"]}, indent=2), flush=True)


def correlation(x, y):
    """Return a descriptive correlation without independent-sample inference."""
    if np.size(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(np.ravel(x), np.ravel(y))[0, 1])


def describe(residual, mask, pair_left, pair_right):
    """Summarize a fixed set of pixels and native adjacent exposure pairs."""
    if not mask.any():
        return {"pixels_per_exposure": 0}
    selected = np.broadcast_to(mask[:, None], residual.shape)
    centered = residual-residual.mean(axis=1, keepdims=True)
    differences = (residual[:, pair_right]-residual[:, pair_left])/np.sqrt(2)
    difference_selection = np.broadcast_to(mask[:, None], differences.shape)
    spectral_pairs = mask[:, :-1] & mask[:, 1:]
    spectral_selection = np.broadcast_to(spectral_pairs[:, None], differences[:, :, :-1].shape)
    raw_spectral_selection = np.broadcast_to(spectral_pairs[:, None], residual[:, :, :-1].shape)
    result = dict(pixels_per_exposure=int(mask.sum()),
                  difference_sample_count=int(difference_selection.sum()),
                  mean_pattern_sse_fraction=float(np.sum((residual.mean(axis=1)[mask])**2)*residual.shape[1]/np.sum(residual[selected]**2)),
                  temporal_lag1_correlation=correlation(centered[:, pair_left][difference_selection],
                                                        centered[:, pair_right][difference_selection]),
                  residual_spectral_lag1_correlation=correlation(residual[:, :, :-1][raw_spectral_selection],
                                                                 residual[:, :, 1:][raw_spectral_selection]),
                  difference_spectral_lag1_correlation=correlation(differences[:, :, :-1][spectral_selection],
                                                                   differences[:, :, 1:][spectral_selection]))
    for name, values in [("residual", residual[selected]),
                         ("temporal_scatter", centered[selected]),
                         ("adjacent_difference", differences[difference_selection])]:
        result[name] = dict(rms_percent=float(100*np.sqrt(np.mean(values**2))),
                            robust_sigma_percent=float(100*1.4826*np.median(np.abs(values-np.median(values)))),
                            mean_percent=float(100*np.mean(values)))
    return result


if __name__ == "__main__":
    main()
