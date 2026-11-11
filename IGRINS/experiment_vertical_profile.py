"""Test three atmospheric profile modes with exact nonlinear checks.

The optimization is local and training-only. The adopted model is never changed.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from atmosphere_profile import load_profile
from fit_line_strength import build_context, metrics, predict_components, WORKING_SIGMA
from hitran_cache import load_mdb
from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import Patm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
NAMES = ["water_pressure_exponent", "lower_temperature_K", "upper_temperature_K"]
SCALES = np.array([.5, 2., 3.])


def main():
    started = time.monotonic()
    context = build_context()
    reference = json.loads((OUT / "motion_clean_control.json").read_text())
    profile = load_profile(reference["layers"])
    with np.load(OUT / "adopted_telluric_fit.npz") as data:
        strength_z = data["parameters"][data["line_parameter_indices"]]
        adopted_component = data["component"]
        contrast = data["line_contrast"]
    np.testing.assert_allclose(predict_components(context, strength_z), adopted_component,
                               atol=1e-10, rtol=0)
    pressure = np.asarray(profile["pressure_bar"])
    columns = np.asarray(profile["number_column_cm2"])
    temperature = np.asarray(profile["temperature_K"])
    reference_water = np.asarray(profile["vmr"]["H2O"])
    reference_water_column = float(columns @ reference_water)
    lower_mode = .5 * (1 + np.tanh(np.log(pressure / .4) / .4))
    troposphere = .5 * (1 + np.tanh(np.log(pressure / .1) / .25))
    upper_mode = (1 - lower_mode) * troposphere

    def atmosphere(q):
        physical = np.asarray(q) * SCALES
        water = reference_water * np.exp(physical[0] * np.log(pressure / pressure[-1]))
        water *= reference_water_column / (columns @ water)
        temp = temperature + physical[1] * lower_mode + physical[2] * upper_mode
        return temp, water

    bases, line_counts = [], []
    for block in context["blocks"]:
        order_bases = []
        for species in ["H2O", "CO2"]:
            low = float(block["grid"][0]) - reference["line_margin_cm1"]
            high = float(block["grid"][-1]) + reference["line_margin_cm1"]
            db = load_mdb(species, low, high, reference["line_cutoff"])
            opa = OpaDirect(db, block["grid"])
            with np.load(ROOT / "cache" / f"{species}.npz") as raw:
                indices = np.flatnonzero((raw["nu"] >= low) & (raw["nu"] <= high)
                                        & (raw["sw"] >= reference["line_cutoff"]))
            selected = [(i, r) for i, r in enumerate(block["records"]) if r["species"] == species]
            local = np.array([i for i, r in selected], dtype=int)
            raw_selected = np.array([r["raw_index"] for i, r in selected], dtype=int)
            mapping = np.searchsorted(indices, raw_selected)
            np.testing.assert_array_equal(indices[mapping], raw_selected)

            def layer(t, p, ps, factors, db=db, offset=opa.opainfo):
                strength = line_strength(t, db.logsij0, db.nu_lines, db.elower,
                                         db.qr_interp_lines(t, 296.), 296.) * factors
                width = gamma_hitran(p, t, ps, db.n_air, db.gamma_air,
                                     db.gamma_self) + gamma_natural(db.A)
                sigma = doppler_sigma(db.nu_lines, t, db.molmass)
                # Keep the original adopted line shift while isolating profile and width.
                shift = db.delta_air * p / Patm
                return xsvector(offset - shift[:, None], sigma, width, strength)

            order_bases.append(dict(species=species, layer=jax.jit(layer),
                n_lines=len(indices), local=local, mapping=mapping,
                scales=block["line_factors"][:, local[0]]))
            line_counts.append(dict(order=block["order"], species=species, n_lines=len(indices)))
        bases.append(order_bases)

    evaluation_count = 0

    def evaluate(q, z=strength_z, self_width=True):
        nonlocal evaluation_count
        temp, water = atmosphere(q)
        output = []
        for block, order_bases in zip(context["blocks"], bases):
            total = np.zeros_like(block["baseline_tau"])
            local_z = z[block["parameter_slice"]]
            for basis in order_bases:
                factors = np.ones(basis["n_lines"])
                factors[basis["mapping"]] += block["upper"][basis["local"]] * local_z[basis["local"]]
                if factors.min() < -1e-10:
                    raise ValueError("Negative line strength")
                factors = jnp.asarray(np.maximum(factors, 0.))
                vmr = water if basis["species"] == "H2O" else np.asarray(profile["vmr"]["CO2"])
                # The same abundance enters column opacity and collider partial pressure.
                for scale in np.unique(basis["scales"]):
                    actual_vmr = scale * vmr
                    partial_pressure = pressure * actual_vmr if self_width else np.zeros_like(pressure)
                    tau = np.zeros(len(block["grid"]))
                    for t, p, ps, column in zip(temp, pressure, partial_pressure, columns * actual_vmr):
                        tau += column * np.asarray(basis["layer"](t, p, ps, factors))
                    total[basis["scales"] == scale] += tau
            intrinsic = jnp.asarray(block["shifted_star"]) * jnp.exp(
                -jnp.asarray(block["airmass"])[:, None] * total)

            def response(row, width, rv):
                core = sampling(block["target"], block["grid"], ipgauss(row, block["velocity"], width), rv)
                wing = sampling(block["target"], block["grid"], ipgauss(row, block["velocity"], 2 * width), rv)
                return (1 - block["wing_fraction"]) * core + block["wing_fraction"] * wing

            output.append(np.asarray(jax.vmap(response)(intrinsic, block["widths"], block["offsets"])))
        evaluation_count += 1
        print("Exact spectrum", evaluation_count, "q", np.asarray(q).tolist(),
              "elapsed", round(time.monotonic() - started, 1), flush=True)
        return np.array(output)

    reconstructed = evaluate(np.zeros(3), self_width=False)
    reconstruction_error = float(np.max(abs(reconstructed - adopted_component)))
    np.testing.assert_allclose(reconstructed, adopted_component, atol=1e-10, rtol=0)
    cases, arrays = {}, {}

    def record(tag, component, **extra):
        result, residual = metrics(context, component)
        cases[tag] = dict(metrics=result, **extra)
        arrays[f"component_{tag}"] = component
        print(tag, "train", result["training"]["rms_percent"],
              "eval", result["evaluation_score"]["rms_percent"], flush=True)
        return residual

    record("adopted", reconstructed)
    baseline = evaluate(np.zeros(3))
    residual = record("abundance_consistent_self", baseline, scaled_parameters=[0., 0., 0.])
    directions = []
    for i, name in enumerate(NAMES):
        q = np.eye(3)[i]
        pair = []
        for sign, suffix in [(-1, "minus"), (1, "plus")]:
            component = evaluate(sign * q)
            record(f"{name}_{suffix}", component,
                   scaled_parameters=(sign * q).tolist(), physical_parameters=(SCALES * sign * q).tolist())
            pair.append(component)
        directions.append(context["project"]((pair[1] - pair[0]) / 2))
    directions = np.stack(directions, axis=-1)
    keep = np.broadcast_to(context["allowed"][:, None], residual[:, :5].shape)
    matrix = directions[:, :5][keep]
    y = residual[:, :5][keep]
    sigma = WORKING_SIGMA
    fixed = lsq_linear(np.vstack([matrix / sigma, np.eye(3)]),
                       np.r_[y / sigma, np.zeros(3)], bounds=(-3, 3), tol=1e-12)
    fixed_component = evaluate(fixed.x)
    record("profile_fixed_strengths", fixed_component, scaled_parameters=fixed.x.tolist(),
           physical_parameters=(fixed.x * SCALES).tolist(), success=bool(fixed.success),
           linear_prediction_training_error=float(np.sqrt(np.mean((
               context["project"](fixed_component - baseline)[:, :5][keep] - matrix @ fixed.x)**2))))

    # Conditional intensity tangent at the adopted model; exact spectra below
    # include the changed profile, changed strengths, and abundance-consistent widths.
    strength_matrix = np.zeros((len(y), len(strength_z)))
    row_offset = 0
    for b, block in enumerate(context["blocks"]):
        rows = 5 * int(context["allowed"][b].sum())
        derivative = -sigma * np.asarray(block["data_jacobian"](
            strength_z[block["parameter_slice"]], jnp.asarray(context["y_over_c"][b, :5])))
        strength_matrix[row_offset:row_offset + rows, block["parameter_slice"]] = derivative
        row_offset += rows
    widths = 1 + 9 * contrast**2 / (contrast**2 + .005**2)
    upper_fraction = np.array([r["uncertainty_upper"] for r in context["records"]])
    lower = -1 / upper_fraction - strength_z
    strength_fit = lsq_linear(np.vstack([strength_matrix / sigma, np.diag(1 / widths)]),
        np.r_[y / sigma, -strength_z / widths], bounds=(lower, np.full(len(lower), np.inf)), tol=1e-11)
    strength_component = evaluate(np.zeros(3), strength_z + strength_fit.x)
    record("self_strength_profile", strength_component, success=bool(strength_fit.success),
           linear_prediction_training_error=float(np.sqrt(np.mean((
               context["project"](strength_component - baseline)[:, :5][keep]
               - strength_matrix @ strength_fit.x)**2))))
    full = np.column_stack([strength_matrix, matrix])
    joint = lsq_linear(np.vstack([full / sigma, np.diag(np.r_[1 / widths, np.ones(3)])]),
        np.r_[y / sigma, -strength_z / widths, np.zeros(3)],
        bounds=(np.r_[lower, [-3.] * 3], np.r_[np.full(len(lower), np.inf), [3.] * 3]),
        tol=1e-11, max_iter=200)
    joint_component = evaluate(joint.x[-3:], strength_z + joint.x[:-3])
    record("profile_joint_strengths", joint_component, scaled_parameters=joint.x[-3:].tolist(),
           physical_parameters=(joint.x[-3:] * SCALES).tolist(), success=bool(joint.success),
           linear_prediction_training_error=float(np.sqrt(np.mean((
               context["project"](joint_component - baseline)[:, :5][keep] - full @ joint.x)**2))))
    for tag, q in [("fixed", fixed.x), ("joint", joint.x[-3:])]:
        temp, water = atmosphere(q)
        arrays[f"temperature_{tag}_K"] = temp
        arrays[f"water_vmr_{tag}"] = water
        cases[f"profile_{'fixed_strengths' if tag == 'fixed' else 'joint_strengths'}"]["profile_summary"] = dict(
            water_column_ratio=float(columns @ water / reference_water_column),
            water_weighted_pressure_bar=float((columns * water) @ pressure / (columns @ water)),
            maximum_temperature_change_K=float(np.max(abs(temp - temperature))),
            maximum_reference_water_vmr=float(water.max()))
    norms = np.linalg.norm(matrix, axis=0)
    source_paths = [Path(__file__), ROOT / "fit_line_strength.py", ROOT / "atmosphere_profile.py",
        ROOT / "hitran_cache.py", ROOT / "cache/atmosphere_profile.json", ROOT / "cache/atmosphere_profile.npz",
        ROOT / "cache/H2O.par", ROOT / "cache/CO2.par", ROOT / "cache/H2O.npz", ROOT / "cache/CO2.npz",
        OUT / "adopted_telluric_model.json", OUT / "adopted_telluric_fit.npz"] + context["source_files"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        method="Training-only bounded local linear least squares, followed by exact 20-layer Voigt evaluation",
        parameter_names=NAMES, prior_scales=SCALES.tolist(), scaled_bounds=[-3, 3],
        profile_definition=dict(water="w0(P)*(P/Pbottom)^a, renormalized to exactly preserve reference water number column",
            temperature="T0 + lower_temperature_K*lower_mode + upper_temperature_K*upper_mode; saved mode arrays define smooth transitions",
            self_width="Each epoch uses Pself=P*adopted_abundance_scale*profile_vmr in gamma_air*(P-Pself)+gamma_self*Pself"),
        cases=cases, reconstruction_max_error=reconstruction_error,
        opacity_line_counts=line_counts, training_exposures=context["exposures"][:5].tolist(),
        evaluation_exposures=context["exposures"][5:].tolist(),
        training_response_cosine=(matrix.T @ matrix / (norms[:, None] * norms[None, :])).tolist(),
        reference_water_weighted_pressure_bar=float((columns * reference_water) @ pressure / reference_water_column),
        seconds=time.monotonic() - started,
        limitations=["This is a local sensitivity experiment, not a fully iterated nonlinear fit or calibrated posterior.",
            "The intensity nuisance tangent is evaluated at the adopted no-self model; exact checks update strengths and opacity together.",
            "Epoch water and CO2 columns, pressure layers, hydrostatic air columns, stellar model, line positions and original IP are fixed.",
            "Temperature-dependent line strengths, air/self widths and Doppler widths are recomputed for every perturbed layer.",
            "Self widths share n_air; line centers retain adopted total-pressure shifts.",
            "The smooth modes and prior scales are exploratory choices, not meteorological measurement errors.",
            "Evaluation spectra determine no coefficients or hyperparameters; results do not adopt a new model."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
    (OUT / "vertical_profile_experiment.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(OUT / "vertical_profile_experiment.npz", **arrays,
        pressure_bar=pressure, reference_temperature_K=temperature, reference_water_vmr=reference_water,
        reference_number_column_cm2=columns, lower_temperature_mode=lower_mode, upper_temperature_mode=upper_mode,
        wavelength_um=context["wave"], included_mask=context["allowed"], score_mask=context["score"],
        filtered_directions=directions, training_strength_tangent=strength_matrix,
        adopted_strength_z=strength_z, strength_prior_widths=widths,
        fixed_scaled_parameters=fixed.x, self_strength_delta=strength_fit.x, joint_delta=joint.x)
    lines = ["Atmospheric vertical-profile experiment (conditional local fits)", "",
             "Exact nonlinear spectrum RMS, in percent of original continuum:"]
    for name, case in cases.items():
        lines.append(f"{name}: training {case['metrics']['training']['rms_percent']:.8f}%, evaluation {case['metrics']['evaluation_score']['rms_percent']:.8f}%")
        if "physical_parameters" in case:
            lines.append(f"  {dict(zip(NAMES, case['physical_parameters']))}")
    lines += ["", "Limitations:"] + report["limitations"] + [""]
    (OUT / "VERTICAL_PROFILE_EXPERIMENT_REPORT.txt").write_text("\n".join(lines))
    print("Completed in", round(report["seconds"], 1), "seconds", flush=True)


if __name__ == "__main__":
    main()
