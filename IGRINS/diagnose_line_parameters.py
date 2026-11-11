"""Audit line metadata and local parameter sensitivities in discovery spectra.

Line changes are diagnostics at the frozen reference atmosphere. They are not
adopted corrections, and reserved validation exposures are not accessed.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.optimize import lsq_linear

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import Patm, c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
FIELDS = {"nu": (3, 15), "sw": (15, 25), "gamma_air": (35, 40),
          "gamma_self": (40, 45), "elower": (45, 55), "n_air": (55, 59),
          "delta_air": (59, 67)}
UNCERTAINTY_FIELDS = ["nu", "sw", "gamma_air", "gamma_self", "n_air", "delta_air"]
RELATIVE_RANGES = {0: "unreported/unavailable", 1: "default/constant", 2: "average/estimate",
                   3: ">=20%", 4: "10-20%", 5: "5-10%", 6: "2-5%", 7: "1-2%",
                   8: "0.1-1%", 9: "<0.1%"}
ABSOLUTE_RANGES = {0: ">=1 cm-1 or unreported", 1: "0.1-1 cm-1", 2: "0.01-0.1 cm-1",
                   3: "0.001-0.01 cm-1", 4: "0.0001-0.001 cm-1", 5: "1e-5-1e-4 cm-1",
                   6: "1e-6-1e-5 cm-1", 7: "1e-7-1e-6 cm-1", 8: "1e-8-1e-7 cm-1",
                   9: "1e-9-1e-8 cm-1"}


def line_record(species, center):
    lines = (ROOT/"cache"/f"{species}.par").read_text().splitlines()
    selected = min(lines, key=lambda row: abs(float(row[3:15])-center))
    if abs(float(selected[3:15])-center) > 1e-7:
        raise ValueError("Requested line absent from cache")
    result = {name: float(selected[a:b]) for name, (a, b) in FIELDS.items()}
    result.update(species=species, isotope=selected[2], ierr=selected[127:133],
                  iref=selected[133:145], mixing_flag=selected[145],
                  global_upper=selected[67:82], global_lower=selected[82:97],
                  local_upper=selected[97:112], local_lower=selected[112:127])
    result["uncertainty_codes"] = dict(zip(UNCERTAINTY_FIELDS, map(int, result["ierr"])))
    result["reference_codes"] = dict(zip(UNCERTAINTY_FIELDS,
                                        [int(result["iref"][i:i+2]) for i in range(0, 12, 2)]))
    result["uncertainty_ranges"] = {key: (ABSOLUTE_RANGES if key in ["nu", "delta_air"] else RELATIVE_RANGES)[code]
                                     for key, code in result["uncertainty_codes"].items()}
    return result


def main():
    source = json.loads((OUT/"focus_ip.json").read_text())
    p = source["parameters"]
    with np.load(OUT/"transfer_diagnostics.npz") as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(OUT/"telluric_mask.npz") as saved:
        mask_data = {key: saved[key] for key in saved.files}
    exposures, orders = data["exposures"], data["orders"]
    assert not np.isin(exposures, [10, 30, 50, 70]).any()
    residual = data["fixed_residual"]
    mean = residual.mean(axis=1)
    sse = np.sum(residual**2, axis=1)
    static_sse = len(exposures)*mean**2
    included = data["allowed"]
    static_fraction = float(static_sse[included].sum()/sse[included].sum())
    candidates = []
    for b, order in enumerate(orders):
        covariate = data["slant_water"] if order == 40 else data["airmass"]
        centered = covariate-covariate.mean()
        slope = np.sum(residual[b]*centered[:, None], axis=0)/np.sum(centered**2)
        correlation = slope*np.std(covariate)/np.std(residual[b], axis=0)
        effect = abs(slope)*np.ptp(covariate)
        effect[~included[b]] = 0
        peaks, _ = find_peaks(effect, distance=10)
        for k in sorted(peaks, key=lambda k: effect[k], reverse=True)[:6]:
            component = data["fixed_ip_model"][b, :, k]/data["fixed_ip_continuum"][b, :, k]
            observed = data["observed"][b, :, k]/data["fixed_ip_continuum"][b, :, k]
            candidates.append(dict(order=int(order), pixel_index=int(k),
                                   wavelength_um=float(data["wavelength_um"][b, k]),
                                   correlation=float(correlation[k]),
                                   fitted_change_percent=float(100*effect[k]),
                                   mean_residual_percent=float(100*mean[b, k]),
                                   maximum_telluric_depth_percent=float(100*mask_data["telluric_depth"][b, k]),
                                   model_flux_range=[float(component.min()), float(component.max())],
                                   observed_flux_range=[float(observed.min()), float(observed.max())]))

    profile = load_profile(source["layers"])
    info = pd.read_csv(ROOT/"data/20201214_info.csv")
    relative_rv = info.iloc[exposures]["RV [km/s]"].to_numpy()-info.iloc[0]["RV [km/s]"]
    airmass = data["airmass"]
    water = 10**data["fixed_ip_parameters"][:, 0]
    epoch_rv = data["fixed_ip_parameters"][:, 1]
    targets = [(13, "CO2", 4815.175659), (13, "CO2", 4818.022683),
               (40, "H2O", 6677.330001), (40, "H2O", 6681.995047)]
    blocks = {}
    for order in set(target[0] for target in targets):
        b = int(np.flatnonzero(orders == order)[0])
        nu = 1e4/data["wavelength_um"][b]
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/source["dv_kms"]))+1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, p["stellar_extra_sigma_kms"])
        stars = jnp.stack([sampling(grid, grid, star, p["stellar_rv_reference_kms"]+rv)
                           for rv in relative_rv])
        taus = {}
        for species in ["H2O", "CO2"]:
            db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
            opa = OpaDirect(db, grid)
            taus[species] = profile_tau(opa, grid, profile, species, True, False)[0]
        total = water[:, None]*taus["H2O"]+10**p["log10_scale_CO2"]*taus["CO2"]
        calibrated_nu = nu*(1+p[f"stretch_o{order}_kms"]*np.linspace(-1, 1, len(nu))/c)
        blocks[order] = dict(b=b, grid=grid, velocity=velocity, stars=stars,
                             tau=jnp.asarray(total), calibrated_nu=calibrated_nu)

    reports, response_arrays = [], {}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for ax, (order, species, center) in zip(axes.flat, targets):
        record = line_record(species, center)
        block = blocks[order]
        b = block["b"]
        db = load_mdb(species, center-1e-7, center+1e-7, crit=0.)
        assert len(db.nu_lines) == 1
        opa = OpaDirect(db, block["grid"])
        columns = profile["number_column_cm2"]*profile["vmr"][species]
        def optical_depth(q):
            layers = []
            for temperature, pressure, column in zip(profile["temperature_K"], profile["pressure_bar"], columns):
                strength = line_strength(temperature, db.logsij0, db.nu_lines, db.elower,
                                         db.qr_interp_lines(temperature, 296.), 296.)*jnp.exp(q[0])
                gamma = gamma_hitran(pressure, temperature, 0., db.n_air,
                                     db.gamma_air*jnp.exp(q[2]), db.gamma_self)+gamma_natural(db.A)
                sigma = doppler_sigma(db.nu_lines, temperature, db.molmass)
                shift = (db.delta_air+q[3])*pressure/Patm+db.nu_lines*q[1]/c
                layers.append(column*xsvector(opa.opainfo-shift[:, None], sigma, gamma, strength))
            return jnp.sum(jnp.stack(layers), axis=0)
        tau_zero = optical_depth(jnp.zeros(4))
        species_scale = jnp.asarray(water if species == "H2O" else np.full(len(exposures), 10**p["log10_scale_CO2"]))
        def spectra(q):
            delta = optical_depth(q)-tau_zero
            intrinsic = block["stars"]*jnp.exp(-jnp.asarray(airmass)[:, None]*(block["tau"]+species_scale[:, None]*delta))
            beta = p[f"ip_sigma_o{order}_kms"]
            fraction = p["ip_wing_fraction"]
            broad = jax.vmap(lambda row: (1-fraction)*ipgauss(row, block["velocity"], beta)
                            + fraction*ipgauss(row, block["velocity"], 2*beta))(intrinsic)
            return jax.vmap(lambda row, offset: sampling(block["calibrated_nu"], block["grid"], row,
                                                       p[f"calibration_rv_o{order}_kms"]+offset))(broad, jnp.asarray(epoch_rv))
        evaluate = jax.jit(spectra)
        prediction = np.asarray(evaluate(jnp.zeros(4)))
        reference = data["fixed_ip_model"][b]/data["fixed_ip_continuum"][b]
        reconstruction_error = float(np.max(abs(prediction-reference)))
        if reconstruction_error > 1e-11:
            raise ValueError(f"Reconstruction mismatch: {reconstruction_error}")
        steps = np.array([.01, .05, .01, .001])
        responses = []
        for j in range(4):
            q = np.zeros(4); q[j] = steps[j]
            responses.append((np.asarray(evaluate(q))-np.asarray(evaluate(-q)))/(2*steps[j]))
        responses = np.stack(responses, axis=-1)
        velocity_from_line = c*(block["calibrated_nu"]*(1+p[f"calibration_rv_o{order}_kms"]/c)-center)/center
        window = (abs(velocity_from_line) < 20) & included[b]
        # Remove the time mean of every pixel, without modifying the fixed mask.
        # This isolates the varying part; it is a diagnostic nuisance projection.
        dynamic_response = responses-responses.mean(axis=0, keepdims=True)
        dynamic_residual = residual[b]-residual[b].mean(axis=0, keepdims=True)
        matrix = dynamic_response[:, window].reshape(-1, 4)
        target = dynamic_residual[:, window].ravel()
        correction, _, _, singular = np.linalg.lstsq(matrix, target, rcond=None)
        column_norms = np.linalg.norm(matrix, axis=0)
        normalized_matrix = matrix/column_norms
        normalized_singular = np.linalg.svd(normalized_matrix, compute_uv=False)
        explained = matrix@correction
        original_rms = np.sqrt(np.mean(target**2))
        remaining_rms = np.sqrt(np.mean((target-explained)**2))
        codes = record["uncertainty_codes"]
        relative_upper = {3: .3, 4: .2, 5: .1, 6: .05, 7: .02, 8: .01, 9: .001}
        strength_limit = relative_upper.get(codes["sw"], .1)
        width_limit = relative_upper.get(codes["gamma_air"], .1)
        center_limit = c*10**(-codes["nu"]+1)/center
        shift_limit = 10**(-codes["delta_air"]+1)
        bounds = np.array([strength_limit, center_limit, width_limit, shift_limit])
        bounded = lsq_linear(matrix, target, bounds=(-bounds, bounds), tol=1e-12)
        bounded_rms = np.sqrt(np.mean((target-matrix@bounded.x)**2))
        exact_delta = np.asarray(evaluate(bounded.x))-prediction
        linear_delta = np.einsum("epk,k->ep", responses, bounded.x)
        linearization_error = float(np.max(abs(exact_delta[:, window]-linear_delta[:, window])))
        single = []
        parameter_names = ["log_intensity", "center_velocity_kms", "log_air_width", "air_shift_cm1_atm"]
        for j, name in enumerate(parameter_names):
            response = matrix[:, j]
            coefficient = float(response@target/(response@response))
            single.append(dict(parameter=name, fitted_linear_coefficient=coefficient,
                               dynamic_rms_percent=float(100*np.sqrt(np.mean((target-coefficient*response)**2)))))
        trial_maxima = {}
        for j, name in enumerate(["intensity_2percent", "center_0.05kms", "air_width_5percent", "air_shift_0.005cm1_atm"]):
            delta = [.02, .05, .05, .005][j]
            trial_maxima[name] = float(100*np.max(abs(responses[:, window, j]*delta)))
        reports.append(dict(order=order, line=record, reconstruction_max_error=reconstruction_error,
                             included_window_pixels=int(window.sum()),
                             dynamic_rms_percent=float(100*original_rms),
                             joint_linear_correction=dict(zip(parameter_names, correction.tolist())),
                             joint_linear_remaining_rms_percent=float(100*remaining_rms),
                             bounded_linear_correction=dict(zip(parameter_names, bounded.x.tolist())),
                             diagnostic_absolute_bounds=dict(zip(parameter_names, bounds.tolist())),
                             bounded_linear_remaining_rms_percent=float(100*bounded_rms),
                             bounded_exact_vs_linear_max_flux_difference_percent=100*linearization_error,
                             bound_note="Use upper uncertainty-bin values as symmetric diagnostic bounds; unreported intensity gets exploratory +/-10%, not a reported uncertainty",
                             response_condition_number=float(singular[0]/singular[-1]),
                             unit_normalized_response_condition_number=float(normalized_singular[0]/normalized_singular[-1]),
                             center_pressure_shift_response_cosine=float(normalized_matrix[:, 1]@normalized_matrix[:, 3]),
                             single_parameter_tests=single, trial_max_flux_changes_percent=trial_maxima,
                             note="Linearized response at fixed TP/IP; inferred corrections are confounded and not adopted"))
        key = f"o{order}_{species}_{center:.6f}"
        response_arrays[key] = responses
        near = abs(velocity_from_line) < 20
        mean_residual = residual[b].mean(axis=0)
        ax.plot(velocity_from_line[near], 100*mean_residual[near], color=".75", lw=.8)
        selected_residual = np.where(included[b], mean_residual, np.nan)
        ax.plot(velocity_from_line[near], 100*selected_residual[near], color="k", label="Mean included residual")
        for j, name in enumerate(["+2% intensity", "+0.05 km/s center", "+5% air width", "+0.005 cm-1/atm shift"]):
            ax.plot(velocity_from_line[near], 100*responses[:, near, j].mean(axis=0)*[.02, .05, .05, .005][j], label=name)
        ax.axhline(0, color=".6", lw=.6)
        ax.set(xlabel="Velocity from zero-pressure line center (km/s)", ylabel="Flux change / residual (%)",
               title=f"{species} {center:.6f} cm-1 | order {order}")
        ax.legend(fontsize=7)
        print(key, reports[-1]["joint_linear_correction"], original_rms, remaining_rms, flush=True)

    result = dict(reserved_exposures=[10, 30, 50, 70], selected_exposures=exposures.tolist(),
                  airmass_range=[float(airmass.min()), float(airmass.max())],
                  slant_water_range=[float(data["slant_water"].min()), float(data["slant_water"].max())],
                  static_residual_sse_fraction=static_fraction,
                  order_static_residual_sse_fraction={str(order): float(static_sse[b, included[b]].sum()/sse[b, included[b]].sum())
                                                     for b, order in enumerate(orders)},
                  varying_residual_candidates=candidates, line_tests=reports,
                  interpretation=["Mean-square decomposition does not identify stellar versus terrestrial causes",
                                  "Airmass correlation does not imply absorption: check whether modeled transmission has a line",
                                  "The 1.501104 micron candidate exceeds continuum and has <1% modeled telluric absorption",
                                  "Large diagnostic coefficients invalidate the small-perturbation approximation and are not line corrections",
                                  "Temperature/profile and continuum/IP errors can project onto the same line responses"],
                  documentation=["https://hitran.org/docs/uncertainties/", "https://hitran.org/docs/definitions-and-units/",
                                 "https://hitran.org/static/hapi/hapi_manual.pdf"],
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__), OUT/"transfer_diagnostics.npz", OUT/"focus_ip.json",
                                              ROOT/"cache/H2O.par", ROOT/"cache/CO2.par"]})
    (OUT/"line_parameter_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"line_parameter_responses.npz", **response_arrays)
    fig.suptitle("Single-line perturbations at the fixed reference atmosphere; not adopted fits")
    fig.savefig(OUT/"line_parameter_responses.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
