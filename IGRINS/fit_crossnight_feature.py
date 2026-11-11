"""Compare observer-fixed and stellar-moving local absorption hypotheses.

The extra weak Gaussian is a diagnostic nuisance opacity, multiplied before
IP convolution. Shared Dec14 TP profiles are proxies for other nights, not
retrieved terrestrial structures. Main fit masks and models are unchanged.
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

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
OUT = ROOT/"results"
NIGHTS = ["06", "14", "21"]
FEATURES = [(1.501046034, [1.50074508, 1.50125903]),
            (1.494855416, [1.49494904, 1.49579716])]


def main():
    source = json.loads((OUT/"focus_ip.json").read_text())
    p = source["parameters"]
    info14 = pd.read_csv(ROOT/"data/20201214_info.csv")
    calibrations = {offset: json.loads((OUT/f"crossnight_localization{'_offset1' if offset else ''}.json").read_text())["calibrations"]["40"]
                    for offset in [0, 1]}
    inventory, datasets = [], {}
    for night in NIGHTS:
        with (ROOT/"data"/f"cube{night}v3.pic").open("rb") as stream:
            wavelength, cube = pickle.load(stream)
        info = pd.read_csv(ROOT/"data"/f"202012{night}_info.csv")
        nearest = int(np.argmin(abs(info["Air Mass"].to_numpy()-1.2)))
        indices = np.arange(nearest-2, nearest+3)
        if night == "14" and np.isin(indices, [10, 30, 50, 70]).any():
            raise ValueError("Reserved frames must not enter localization")
        for offset in ([0, 1] if night == "21" else [0]):
            spectra = cube[40, indices+offset]
            scaled = spectra/np.percentile(spectra, 95, axis=1)[:, None]
            datasets[(night, offset)] = dict(wave=wavelength[40], flux=np.median(scaled, axis=0),
                                             airmass=float(np.median(info.iloc[indices]["Air Mass"])),
                                             rv=float(np.median(info.iloc[indices]["RV [km/s]"])))
        inventory.append(dict(night=night, selected_metadata=indices.tolist(),
                              median_stellar_rv_kms=datasets[(night, 0)]["rv"],
                              frame_mapping="Ambiguous local offset0/+1:51cube frames,50CSV rows" if night == "21" else "Matched frame counts"))
    cases = [("06_14", ["06", "14"], 0, t) for t in [0., -10., 10.]]
    cases += [("all3", NIGHTS, 0, t) for t in [0., -10., 10.]]
    cases += [("all3_offset1", NIGHTS, 1, 0.)]
    plan = dict(target_half_window_kms=9., anchor_half_window_kms=12., intrinsic_gaussian_sigma_bounds_kms=[.2, 6.],
                extra_optical_depth_bounds=[1e-6, .15], night_ip_sigma_bounds_kms=[1., 4.5],
                hypotheses=["observer", "stellar"], continuum_degree=2,
                temperature_tests="Offsets0/-10/+10 K applied to non-Dec14 proxy profiles only",
                cases=[dict(label=label, nights=nights, night21_index_offset=offset, other_night_temperature_offset_K=t)
                       for label, nights, offset, t in cases],
                warning="New local diagnostic only; not a change to any main fitted spectrum or comparison mask")
    (OUT/"crossnight_physical_plan.json").write_text(json.dumps(plan, indent=2)+"\n")
    results, saved = [], {}
    for target, anchor_waves in FEATURES:
        # The reference feature coordinate uses the absolute Dec14 wavelength
        # calibration already present in the physical forward model.
        with np.load(OUT/"telluric_mask.npz") as mask:
            reference_wave = mask["wavelength_um"][2]
        ref_x = float(np.interp(target, reference_wave[::-1], np.linspace(-1, 1, len(reference_wave))[::-1]))
        target_true = target/((1+p["stretch_o40_kms"]*ref_x/c)*(1+p["calibration_rv_o40_kms"]/c))
        minimum_wave = min([target]+anchor_waves)*np.exp(-150/c)
        maximum_wave = max([target]+anchor_waves)*np.exp(150/c)
        n = int(np.ceil(np.log(maximum_wave/minimum_wave)*c/.2))+1
        n += n % 2
        grid = np.geomspace(1e4/maximum_wave, 1e4/minimum_wave, n)
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        feature_velocity = jnp.asarray(c*((1e4/grid)/target_true-1))
        stellar = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        stellar = ipgauss(stellar, velocity, p["stellar_extra_sigma_kms"])
        taus = {}
        for t in [0., -10., 10.]:
            profile = load_profile(20)
            profile["temperature_K"] = profile["temperature_K"]+t
            rows = []
            for species in ["H2O", "CO2"]:
                db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
                if not len(db.nu_lines):
                    rows.append(np.zeros(len(grid)))
                else:
                    rows.append(profile_tau(OpaDirect(db, grid), grid, profile, species, True, False)[0])
            taus[t] = jnp.asarray(rows)
        for label, nights, offset21, temperature in cases:
            blocks = []
            for night in nights:
                data = datasets[(night, offset21 if night == "21" else 0)]
                wave = data["wave"]
                cal = calibrations[offset21][night]
                offset = cal["offset_kms"]+cal["slope_kms"]*(wave-cal["wavelength_midpoint_um"])/cal["wavelength_span_um"]
                wave_aligned = wave/(1+offset/c)
                target_mask = abs(c*(wave_aligned/target-1)) < 9.
                selected = target_mask.copy()
                for anchor in anchor_waves:
                    selected |= abs(c*(wave_aligned/anchor-1)) < 12.
                wave = wave[selected]
                flux = data["flux"][selected]
                y = flux/np.percentile(flux, 95)
                x = np.interp(wave_aligned[selected], reference_wave[::-1], np.linspace(-1, 1, len(reference_wave))[::-1])
                calibrated_nu = (1e4/wave_aligned[selected])*(1+p["stretch_o40_kms"]*x/c)
                continuum_x = 2*(wave-wave.min())/np.ptp(wave)-1
                basis = jnp.asarray(np.polynomial.chebyshev.chebvander(continuum_x, 2))
                star_rv = p["stellar_rv_reference_kms"]+data["rv"]-info14.iloc[0]["RV [km/s]"]
                star = sampling(grid, grid, stellar, star_rv)
                blocks.append(dict(night=night, wave=wave, nu=calibrated_nu, y=jnp.asarray(y), basis=basis,
                                   star=star, airmass=data["airmass"], tau=taus[0. if night == "14" else temperature],
                                   target_mask=target_mask[selected], rv_shift=data["rv"]-datasets[("14", 0)]["rv"]))
            observed = np.concatenate([np.asarray(block["y"]) for block in blocks])
            target_pixels = np.concatenate([block["target_mask"] for block in blocks])
            count = len(nights)
            initial = np.r_[2.5, 0., np.tile([-.15, 2.3, .05], count)]
            lower = np.r_[.2, -7., np.tile([-1.5, 1., 1e-6], count)]
            upper = np.r_[6., 7., np.tile([1., 4.5, .15], count)]
            for hypothesis in ["observer", "stellar"]:
                shifts = [block["rv_shift"] if hypothesis == "stellar" else 0. for block in blocks]
                def evaluate(q, with_feature=True):
                    predictions, continua, nofeature = [], [], []
                    for j, block in enumerate(blocks):
                        water, sigma, amplitude = q[2+3*j:5+3*j]
                        tau = block["airmass"]*(10**water*block["tau"][0]+10**p["log10_scale_CO2"]*block["tau"][1])
                        extra = amplitude*jnp.exp(-.5*((feature_velocity-q[1]-shifts[j])/q[0])**2)
                        intrinsic = block["star"]*jnp.exp(-tau-extra)
                        fraction = p["ip_wing_fraction"]
                        broad = ((1-fraction)*ipgauss(intrinsic, velocity, sigma)
                                 + fraction*ipgauss(intrinsic, velocity, 2*sigma))
                        sampled = sampling(block["nu"], grid, broad, p["calibration_rv_o40_kms"])
                        design = block["basis"]*sampled[:, None]
                        coefficients = jnp.linalg.solve(design.T@design+1e-12*jnp.eye(3), design.T@block["y"])
                        continuum = block["basis"]@coefficients
                        predictions.append(sampled*continuum)
                        continua.append(continuum)
                    return jnp.concatenate(predictions), jnp.concatenate(continua)
                fun = jax.jit(lambda q: evaluate(q)[0]-jnp.asarray(observed))
                jac = jax.jit(jax.jacfwd(fun))
                fun(initial).block_until_ready(); jac(initial).block_until_ready()
                trials = []
                for center_start in [-2., 0., 2.]:
                    q0 = initial.copy(); q0[1] = center_start
                    fit = least_squares(lambda q: np.asarray(fun(q)), q0, jac=lambda q: np.asarray(jac(q)),
                                        bounds=(lower, upper), x_scale="jac", max_nfev=150,
                                        ftol=1e-9, xtol=1e-9, gtol=1e-8)
                    trials.append(fit)
                fit = min(trials, key=lambda result: result.cost)
                model, continuum = map(np.asarray, evaluate(fit.x))
                residual = observed-model
                hits = np.flatnonzero((fit.x-lower < .002*(upper-lower)) | (upper-fit.x < .002*(upper-lower))).tolist()
                row = dict(target_wavelength_um=target, anchor_wavelengths_um=anchor_waves,
                           label=label, nights=nights, night21_index_offset=offset21,
                           other_night_temperature_offset_K=temperature, hypothesis=hypothesis,
                           success=bool(fit.success), nfev=int(fit.nfev), bounds_hit_indices=hits,
                           parameters=dict(intrinsic_sigma_kms=float(fit.x[0]), reference_center_kms=float(fit.x[1]),
                                           nights={night:dict(log10_water=float(fit.x[2+3*j]), ip_sigma_kms=float(fit.x[3+3*j]),
                                                             extra_peak_optical_depth=float(fit.x[4+3*j]),
                                                             feature_center_kms=float(fit.x[1]+shifts[j])) for j, night in enumerate(nights)}),
                           all_window_sse=float(np.sum(residual**2)), target_window_sse=float(np.sum(residual[target_pixels]**2)),
                           n_pixels=len(observed), n_target_pixels=int(target_pixels.sum()),
                           all_window_rms_percent=float(100*np.sqrt(np.mean(residual**2))),
                           target_rms_percent=float(100*np.sqrt(np.mean(residual[target_pixels]**2))))
                results.append(row)
                key = f"{target:.9f}_{label}_{temperature:g}_{hypothesis}"
                saved[key] = dict(observed=observed, model=model, continuum=continuum,
                                  wave=np.concatenate([block["wave"] for block in blocks]), target_mask=target_pixels,
                                  night=np.concatenate([np.full(len(block["wave"]), int(block["night"])) for block in blocks]))
                print(key, row["all_window_rms_percent"], row["target_rms_percent"], row["bounds_hit_indices"], flush=True)
    report = dict(plan=plan, inventory=inventory, fits=results,
                  assumptions=["The existing stellar template/RV/width remain fixed; the extra Gaussian is a local nuisance feature",
                               "H2O abundance, Gaussian IP width and extra Gaussian amplitude are independent for each night",
                               "Gaussian intrinsic width and reference center are shared across nights under each motion hypothesis",
                               "The extra optical depth multiplies the known stellar and telluric model before IP convolution",
                               "Only the predefined target/anchor windows enter this diagnostic, not the main fit mask",
                               "Other-night TP profiles are Dec14 proxies; prescribed temperature variants expose part of that systematic",
                               "Same-window SSE is descriptive because no pixel variance arrays are supplied",
                               "Dec21 frame indexing is ambiguous and is supplementary to the matched Dec06-Dec14 comparison"],
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/"crossnight_physical.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT/"crossnight_physical.npz", **{f"{key}__{field}":value for key, data in saved.items() for field, value in data.items()})
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for b, (target, _) in enumerate(FEATURES):
        for j, label in enumerate(["06_14", "all3"]):
            ax = axes[b, j]
            for hypothesis in ["observer", "stellar"]:
                key = f"{target:.9f}_{label}_0_{hypothesis}"
                data = saved[key]
                residual = (data["observed"]-data["model"])/data["continuum"]
                for night in np.unique(data["night"]):
                    use = (data["night"] == night)&data["target_mask"]
                    ax.plot(c*(data["wave"][use]/target-1), 100*residual[use],
                            "o-" if hypothesis == "stellar" else "x--", ms=3, lw=.8,
                            label=f"Dec {night:02d}, {hypothesis}")
            ax.axhline(0, color=".5", lw=.6)
            ax.set(xlabel="Observed velocity from reference feature (km/s)", ylabel="Residual / fitted continuum (%)",
                   title=f"{target:.6f} micron | {label}")
            ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Local physical feature comparison at the unperturbed proxy TP")
    fig.savefig(OUT/"crossnight_physical.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
