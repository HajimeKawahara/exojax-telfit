"""Test local stellar-profile flexibility without changing the current fit."""

from __future__ import annotations

from datetime import datetime, timezone
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from blase_stellar import build_clone
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c
from fit_blase_stellar import forward_functions
from fit_line_strength import build_context

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    source = OUT / "order7_stellar_fit.npz"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as saved:
        data = {key: saved[key] for key in saved.files}
    context = build_context("order7_physical_e10", "order7_physical_e",
                            ("H2O", "CH4", "N2O", "CO", "CO2"))
    block = context["blocks"][0]
    grid, velocity = block["grid"], block["velocity"]
    physical = context["physical_parameters"]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    relative_rv = (info.iloc[data["exposures"]]["RV [km/s]"].to_numpy()
                   - info.iloc[0]["RV [km/s]"])
    rv = jnp.asarray(physical["stellar_rv_reference_kms"] + relative_rv)
    clone = build_clone(grid, 5)
    raw = jnp.asarray(clone.corrected_raw(data["q"]))
    original = forward_functions(block, clone, physical, relative_rv)
    np.testing.assert_allclose(original["full"](data["q"], data["z"]),
                               data["component_refit"][0], rtol=0, atol=1e-11)
    upper, factors = block["upper"], block["line_factors"]
    tau = jnp.asarray(block["baseline_tau"]
                      + (factors * (upper * data["z"])[None, :]) @ block["line_tau"])
    selected = np.array([r["species"] == "CH4" and 4382.8 < r["nu"] < 4383.05
                         for r in block["records"]])
    ch4_tau = jnp.asarray((factors[:, selected]
                          * (1 + upper[selected] * data["z"][selected])[None, :])
                         @ block["line_tau"][selected])
    air = jnp.asarray(block["airmass"])
    widths, offsets = jnp.asarray(block["widths"]), jnp.asarray(block["offsets"])
    fraction, target = block["wing_fraction"], block["target"]
    # These are phenomenological changes to the complete local stellar blend,
    # not adjustments to identified atomic transitions or an adopted model.
    center = 4383.23
    distance = jnp.abs(grid-center)
    taper = .5 * (1 + jnp.cos(jnp.pi*jnp.clip((distance-.65)/.40, 0, 1)))

    def forward(parameters):
        depth, shift_kms, log_width, methane = parameters
        coordinate = center + (grid-center-center*shift_kms/c)/jnp.exp(log_width)
        warped = 1-jnp.exp(depth)*(1-jnp.interp(coordinate, grid, raw))
        changed = raw + taper*(warped-raw)
        rotated = convolve_rigid_rotation(changed, velocity, 4., .6, 0.)
        broadened = ipgauss(rotated, velocity, physical["stellar_extra_sigma_kms"])
        shifted = jax.vmap(lambda v: sampling(grid, grid, broadened, v))(rv)
        transmission = jnp.exp(-air[:, None]*(tau + (methane-1)*ch4_tau))

        def response(row, width, offset):
            core = sampling(target, grid, ipgauss(row, velocity, width), offset)
            wing = sampling(target, grid, ipgauss(row, velocity, 2*width), offset)
            return (1-fraction)*core + fraction*wing

        return jax.vmap(response)(shifted*transmission, widths, offsets)

    forward = jax.jit(forward)
    initial = np.array([0., 0., 0., 1.])
    np.testing.assert_allclose(forward(initial), data["component_refit"][0],
                               rtol=0, atol=1e-11)
    wave = data["wavelength_um"][0]
    nu = 1e4/wave
    local = (nu >= 4382.4) & (nu <= 4383.5)
    core = (nu >= 4382.7) & (nu <= 4383.1)
    wing = (nu >= 4383.18) & (nu <= 4383.4)
    basis = np.polynomial.chebyshev.chebvander(
        2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1, 5)
    inverse = np.linalg.pinv(basis)

    def residual(parameters):
        rows = data["y_over_c"][0] - np.asarray(forward(parameters))
        return rows - (rows @ inverse.T) @ basis.T

    def rms(values):
        return float(100*np.sqrt(np.mean(values**2)))

    cases = {"current": [], "stellar_depth": [0], "stellar_depth_center": [0, 1],
             "stellar_depth_width": [0, 2], "stellar_profile": [0, 1, 2],
             "methane_strength": [3], "stellar_profile_and_methane": [0, 1, 2, 3]}
    bounds = np.array([[-.30, -5., -.35, 0.], [.30, 5., .35, 3.]])
    records, arrays = {}, {}
    for name, free in cases.items():
        params = initial.copy()
        result = None
        if free:
            def objective(vector):
                params[free] = vector
                return residual(params)[:5, local].ravel()/.0045
            result = least_squares(objective, initial[free], jac="3-point",
                                   bounds=bounds[:, free], max_nfev=120,
                                   ftol=1e-10, xtol=1e-10, gtol=1e-7)
            params[free] = result.x
        rows = residual(params)
        record = dict(parameters=dict(depth_multiplier=float(np.exp(params[0])),
                      center_shift_kms=float(params[1]), width_multiplier=float(np.exp(params[2])),
                      methane_multiplier_relative_to_current=float(params[3])),
                      free_parameter_count=len(free),
                      fit=None if result is None else dict(success=bool(result.success),
                      nfev=result.nfev, optimality=float(result.optimality), message=result.message))
        for scope, mask in [("local", local), ("core", core), ("wing", wing),
                            ("whole_order", np.ones(len(nu), bool))]:
            record[scope] = dict(wavelength_pixels=int(mask.sum()),
                                training_rms_percent=rms(rows[:5, mask]),
                                evaluation_rms_percent=rms(rows[5:, mask]))
        records[name] = record
        arrays[f"residual_{name}"] = rows
        arrays[f"component_{name}"] = np.asarray(forward(params))
        print(name, json.dumps(record), flush=True)

    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), adopted=False,
                  source_sha256=digest, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  interval_cm1=[4382.4, 4383.5], training_frames=data["exposures"][:5].tolist(),
                  evaluation_frames=data["exposures"][5:].tolist(), cases=records,
                  stellar_transform_rest_center_cm1=center,
                  taper_full_radius_cm1=.65, taper_zero_radius_cm1=1.05,
                  parameters="Local stellar depth, center and width; optional multiplicative scaling of current selected CH4 optical depth in 4382.8-4383.05 cm-1",
                  limitations=["Post hoc explanatory experiment; not a blind validation or adopted fit.",
                               "Stellar transform changes the complete local blend, not individual atomic lines.",
                               "Five training frames determine shared parameters; four evaluation frames are only scored.",
                               "Full-order Q5 is independently projected in every frame, including evaluation.",
                               "Atmosphere, wavelength solution, IP and all other line corrections remain fixed.",
                               "No planet-injection test; lower RMS does not prove preservation of planetary lines."])
    (OUT / "order7_4383_profile_diagnostic.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT / "order7_4383_profile_diagnostic.npz", wavenumber_cm1=nu,
                        exposures=data["exposures"], local_mask=local, **arrays)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    frame = 6
    axes[0].plot(nu, data["y_over_c"][0, frame], "k.-", lw=.8, ms=3, label="Observed, frame 30")
    for name, color, label in [("current", "tab:blue", "Current fit"),
                               ("stellar_profile", "tab:orange", "Local stellar-profile test")]:
        axes[0].plot(nu, data["y_over_c"][0, frame]-arrays[f"residual_{name}"][frame],
                     color=color, label=label+" + Q5")
        axes[1].plot(nu, 100*arrays[f"residual_{name}"][5:].mean(axis=0),
                     color=color, label=label)
    axes[0].plot(nu, data["stellar_only_refit"][0, frame], "--", color="tab:green", label="Current star alone")
    axes[0].plot(nu, data["telluric_only_refit"][0, frame], ":", color="tab:red", label="Current telluric alone")
    for name, color, label in [("current", "tab:blue", "Current"),
                               ("methane_strength", "tab:purple", "CH4 strength only"),
                               ("stellar_profile_and_methane", "tab:brown", "Stellar profile + CH4 strength")]:
        axes[2].plot(nu, 100*arrays[f"residual_{name}"][5:].mean(axis=0), color=color, label=label)
    axes[0].set_ylabel("Flux / continuum")
    for ax in axes[1:]:
        ax.set_ylabel("Evaluation mean\nresidual (%)")
        ax.axhline(0, color=".6", lw=.7)
    for ax in axes:
        ax.set_xlim(4382.25, 4383.65)
        visible = (nu >= 4382.25) & (nu <= 4383.65)
        plotted = np.concatenate([np.asarray(line.get_ydata())[visible]
                                  for line in ax.lines if len(line.get_ydata()) == len(nu)])
        padding = .08*np.ptp(plotted)
        ax.set_ylim(plotted.min()-padding, plotted.max()+padding)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=.15)
    axes[2].set_xlabel(r"Wavenumber (cm$^{-1}$)")
    fig.suptitle("4383 cm$^{-1}$: overlapping stellar and methane absorption\nPost hoc profile tests; five training frames, four evaluation frames; current model unchanged")
    fig.tight_layout()
    fig.savefig(OUT / "order7_4383_profile_diagnostic.png", dpi=160)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


if __name__ == "__main__":
    main()
