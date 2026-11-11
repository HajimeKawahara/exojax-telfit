"""Audit the 4383 cm-1 blend with frozen stellar and atmospheric backgrounds."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from exojax.database.core.broadening import doppler_sigma, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import vvoigt
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import Patm

from atmosphere_profile import load_profile
from blase_stellar import build_clone
from fit_blase_stellar import forward_functions, polynomial
from fit_joint import profile_tau
from fit_line_strength import build_context
from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    path = OUT / "order7_stellar_fit.npz"
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path) as saved:
        saved = {key: saved[key] for key in saved.files}
    meta = json.loads(path.with_suffix(".json").read_text())
    ctx = build_context("order7_physical_e10", "order7_physical_e",
                        ("H2O", "CH4", "N2O", "CO", "CO2"))
    b = ctx["blocks"][0]
    grid, velocity, target = b["grid"], b["velocity"], b["target"]
    wave = ctx["wave"][0]
    nu = 1e4 / wave
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    rv = info.iloc[ctx["exposures"]]["RV [km/s]"].to_numpy() - info.iloc[0]["RV [km/s]"]
    clone = build_clone(np.asarray(grid), 5)
    funcs = forward_functions(b, clone, ctx["physical_parameters"], rv)
    star = funcs["shifted"](jnp.asarray(saved["q"]))
    factors = jnp.asarray(b["line_factors"])
    correction = (factors * (b["upper"] * saved["z"])[None, :]) @ jnp.asarray(b["line_tau"])
    tau_final = jnp.asarray(b["baseline_tau"]) + correction
    air = jnp.asarray(b["airmass"])
    fraction = b["wing_fraction"]

    @jax.jit
    def response(rows):
        def one(row, width, offset):
            core = sampling(target, grid, ipgauss(row, velocity, width), offset)
            wing = sampling(target, grid, ipgauss(row, velocity, 2*width), offset)
            return (1-fraction)*core+fraction*wing
        return jax.vmap(one)(rows, jnp.asarray(b["widths"]), jnp.asarray(b["offsets"]))

    exact = np.asarray(response(star * jnp.exp(-air[:, None] * tau_final)))
    error = float(np.max(np.abs(exact - saved["component_refit"][0])))
    assert error < 1e-11, error
    profile = load_profile(ctx["physical_metadata"]["layers"])
    # Per-species transmissions are descriptive; their separately convolved product
    # is not the transmission obtained by convolving the intrinsic product.
    isolated = {}
    for molecule in ctx["species"]:
        physical = ctx["physical_metadata"]
        db = load_mdb(molecule, float(grid[0])-physical["line_margin_cm1"],
                      float(grid[-1])+physical["line_margin_cm1"], physical["line_cutoff"])
        tau = profile_tau(OpaDirect(db, grid), grid, profile, molecule,
                          physical["pressure_shift"], physical["reference_self_broadening"])[0]
        indices = [i for i, rec in enumerate(ctx["records"]) if rec["species"] == molecule]
        if indices:
            scale = factors[:, indices[0]]
            delta = (factors[:, indices] * (b["upper"][indices] * saved["z"][indices])[None, :]) @ jnp.asarray(b["line_tau"][indices])
        else:
            scale = 10**ctx["physical_parameters"][f"log10_scale_{molecule}"]
            delta = 0.
        isolated[molecule] = np.asarray(response(jnp.exp(-air[:, None]*(scale[:, None]*tau+delta)))) if indices else np.asarray(response(jnp.exp(-air[:, None]*(scale*tau))))

    index = next(i for i, rec in enumerate(ctx["records"]) if rec["species"] == "CH4" and abs(rec["nu"]-4382.9378) < 1e-8)
    record = ctx["records"][index]
    with np.load(ROOT / "cache/CH4.npz") as cache:
        a = {key: float(cache[key][record["raw_index"]]) for key in ["nu", "sw", "a", "gamma_air", "gamma_self", "elower", "n_air", "delta_air", "line_mass", "iso"]}
        qi = np.where(cache["partition_isotopes"] == a["iso"])[0][0]
        q_grid, q_values = cache["temperatures"].copy(), cache["partition_sums"][qi].copy()
    temperatures = jnp.asarray(profile["temperature_K"])
    pressures = jnp.asarray(profile["pressure_bar"])
    columns = jnp.asarray(np.asarray(profile["number_column_cm2"])*np.asarray(profile["vmr"]["CH4"]))
    qr = jnp.interp(temperatures, q_grid, q_values)/jnp.interp(296., q_grid, q_values)
    strengths = line_strength(temperatures, jnp.log(a["sw"]), a["nu"], a["elower"], qr, 296.)
    sigma = doppler_sigma(a["nu"], temperatures, a["line_mass"])
    adopted_factor = float(1+b["upper"][index]*saved["z"][index])
    adopted_tau = jnp.asarray(b["line_tau"][index])*adopted_factor
    local = (nu >= 4382.4) & (nu <= 4383.5)
    core = (nu >= 4382.7) & (nu <= 4383.1)
    tail = (nu >= 4383.2) & (nu <= 4383.45)
    basis, inverse = polynomial(wave, ctx["allowed"][0])
    observed = jnp.asarray(saved["y_over_c"][0])

    @jax.jit
    def line_tau(p):
        # Parameters: multiplier of the adopted strength, center offset (cm-1),
        # gamma_air multiplier, n_air offset, delta_air offset (cm-1/atm).
        gamma = (296./temperatures)**(a["n_air"]+p[3]) * a["gamma_air"]*p[2]*pressures/Patm + gamma_natural(a["a"])
        centers = a["nu"]+p[1]+(a["delta_air"]+p[4])*pressures/Patm
        per_layer = vvoigt(grid[None, :]-centers[:, None], sigma, gamma)
        return adopted_factor*p[0]*jnp.sum(per_layer*(strengths*columns)[:, None], axis=0)

    origin = np.array([1., 0., 1., 0., 0.])
    line_error = float(np.max(np.abs(np.asarray(line_tau(origin))-adopted_tau)))
    assert line_error < 1e-12, line_error

    @jax.jit
    def residual(p):
        change = factors[:, index, None]*(line_tau(p)-adopted_tau)[None, :]
        model = response(star*jnp.exp(-air[:, None]*(tau_final+change)))
        raw = observed-model
        return raw-(raw @ inverse.T) @ basis.T

    jac = jax.jit(jax.jacfwd(residual))

    def metrics(r):
        return {name: {split: float(100*np.sqrt(np.mean(r[rows][:, mask]**2)))
                       for split, rows in [("training", slice(0, 5)), ("evaluation", slice(5, 9))]}
                for name, mask in [("local", local), ("core", core), ("tail", tail), ("full_order", np.ones_like(local))]}

    baseline_r = np.asarray(residual(origin))
    baseline_j = np.asarray(jac(origin))
    sensitivity_steps = [0.2/adopted_factor, .001, .05, .186, .01]
    sensitivity = {}
    for k, (name, step) in enumerate(zip(["strength_add_0_2_hitran", "center_add_0_001_cm1", "gamma_add_5_percent", "n_air_add_20_percent", "delta_air_add_0_01_cm1_per_atm"], sensitivity_steps)):
        p = origin.copy(); p[k] += step
        r = np.asarray(residual(p))
        delta = r-baseline_r
        sensitivity[name] = dict(step=step, local_max_model_change_percent=float(100*np.max(np.abs(delta[5:, local]))), local_rms_model_change_percent=float(100*np.sqrt(np.mean(delta[5:, local]**2))))

    experiments = {}
    configurations = [
        ("strength_only", [0], [0.], [3.]),
        ("metadata_shape_bounds", [0, 1, 2, 3, 4], [0., -.001, .95, -.186, -.01], [3., .001, 1.05, .186, .01]),
        ("generous_shape_bounds", [0, 1, 2, 3, 4], [0., -.03, .5, -.5, -.03], [3., .03, 2., .5, .03]),
    ]
    for name, free, lower, upper in configurations:
        def expand(values):
            p = origin.copy(); p[free] = values; return p
        def objective(values):
            return np.asarray(residual(expand(values)))[:5, local].ravel()/.0045
        def derivative(values):
            return np.asarray(jac(expand(values)))[:5, local][:, :, free].reshape(-1, len(free))/.0045
        fit = least_squares(objective, origin[free], jac=derivative, bounds=(lower, upper),
                            x_scale="jac", max_nfev=100, ftol=1e-10, xtol=1e-10, gtol=1e-8)
        parameters = expand(fit.x)
        r = np.asarray(residual(parameters))
        experiments[name] = dict(parameters=parameters.tolist(), hitran_strength_factor=float(adopted_factor*parameters[0]),
                                 free_indices=free, lower=lower, upper=upper, success=bool(fit.success),
                                 nfev=int(fit.nfev), optimality=float(fit.optimality), metrics=metrics(r))
        print(name, experiments[name], flush=True)

    rows = []
    for i in np.flatnonzero(local):
        rows.append(dict(pixel=int(i), nu_cm1=float(nu[i]), residual_mean_percent=float(100*np.mean(baseline_r[5:, i])),
                         stellar_transmission=float(np.mean(saved["stellar_only_refit"][0, 5:, i])),
                         telluric_transmission=float(np.mean(saved["telluric_only_refit"][0, 5:, i])),
                         species_absorption_percent={s: float(100*(1-np.mean(t[5:, i]))) for s, t in isolated.items()}))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=source_hash,
                  adoption=False, reconstructed_model_max_error=error, reconstructed_line_max_error=line_error,
                  description="One CH4 line varies before multiplication by the fixed star and before the exact instrumental response. Five training exposures only; four held-out evaluation exposures.",
                  windows={"local": [4382.4, 4383.5], "core": [4382.7, 4383.1], "tail": [4383.2, 4383.45]},
                  line_record=record, line_parameters=a, adopted_strength_factor=adopted_factor,
                  parameter_order=["adopted_strength_multiplier", "center_offset_cm1", "gamma_air_multiplier", "n_air_offset", "delta_air_offset_cm1_per_atm"],
                  baseline_metrics=metrics(baseline_r), sensitivities=sensitivity, experiments=experiments,
                  local_samples=rows,
                  limitations=["This is a local diagnostic with fixed other lines, TP, star and IP; fitted parameters are not a unique physical attribution.",
                               "Metadata shape bounds use finite-bin upper edges where available. n_air code3 has no finite upper bound; 20% is a chosen sensitivity scale, not a formal constraint.",
                               "The main line strength was already calibrated in the adopted model. A large reduction can compensate stellar blend mismatch.",
                               "Per-species isolated convolved transmissions do not multiply exactly to the total convolved transmission."],
                  references=["https://hitran.org/docs/uncertainties/", "https://hitran.org/docs/definitions-and-units/"])
    (OUT / "order7_4383_telluric_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT / "order7_4383_telluric_audit.npz", wavelength_um=wave,
                        residual_baseline=baseline_r, derivative=baseline_j,
                        **{f"transmission_{s}": t for s, t in isolated.items()})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == source_hash
    print(json.dumps(dict(baseline=metrics(baseline_r), sensitivities=sensitivity), indent=2))


if __name__ == "__main__":
    main()
