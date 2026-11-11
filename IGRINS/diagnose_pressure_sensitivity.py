"""Explore global pressure-width errors around the adopted weak-line fit.

This is a bounded local diagnostic, not a calibrated broadening retrieval.
Only training spectra determine the four trial broadening coefficients.
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

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import lsq_linear

from atmosphere_profile import load_profile
from fit_line_strength import build_context, predict_components, metrics, WORKING_SIGMA
from hitran_cache import load_mdb
from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.opacity.lpf.lpf import xsvector
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import Patm

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
NAMES = ["H2O_log_gamma_air", "H2O_delta_n_air", "CO2_log_gamma_air", "CO2_delta_n_air"]
STEPS = np.full(4, .1)


def main():
    started = time.monotonic()
    context = build_context()
    reference_meta = json.loads((OUT/"motion_clean_control.json").read_text())
    profile = load_profile(reference_meta["layers"])
    with np.load(OUT/"quadratic_strength_fit.npz") as saved:
        z = saved["theta_fixed_contrast10"][saved["line_parameter_indices"]]
        expected = saved["component_fixed_contrast10"]
        contrast = saved["line_contrast"]
    np.testing.assert_allclose(predict_components(context, z), expected, atol=1e-10, rtol=0)
    bases, counts = [], []
    for block in context["blocks"]:
        order_bases = []
        grid = block["grid"]
        for species in ["H2O", "CO2"]:
            low = float(grid[0])-reference_meta["line_margin_cm1"]
            high = float(grid[-1])+reference_meta["line_margin_cm1"]
            db = load_mdb(species, low, high, reference_meta["line_cutoff"])
            opa = OpaDirect(db, grid)
            with np.load(ROOT/"cache"/f"{species}.npz") as raw:
                indices = np.flatnonzero((raw["nu"] >= low) & (raw["nu"] <= high)
                                        & (raw["sw"] >= reference_meta["line_cutoff"]))
            selected = [(i, r) for i, r in enumerate(block["records"]) if r["species"] == species]
            local = np.array([i for i, r in selected], dtype=int)
            raw_selected = np.array([r["raw_index"] for i, r in selected], dtype=int)
            mapping = np.searchsorted(indices, raw_selected)
            np.testing.assert_array_equal(indices[mapping], raw_selected)
            # Bind this database in defaults; later loop iterations create new ones.
            def layer(t, p, ps, q, factors, db=db, offset=opa.opainfo):
                strength = line_strength(t, db.logsij0, db.nu_lines, db.elower,
                                         db.qr_interp_lines(t, 296.), 296.)*factors
                width = gamma_hitran(p, t, ps, db.n_air+q[1],
                                     db.gamma_air*jnp.exp(q[0]), db.gamma_self)+gamma_natural(db.A)
                sigma = doppler_sigma(db.nu_lines, t, db.molmass)
                # Hold the already adopted pressure shift unchanged to isolate width.
                shift = db.delta_air*p/Patm
                return xsvector(offset-shift[:, None], sigma, width, strength)
            order_bases.append(dict(species=species, layer=jax.jit(layer), n_lines=len(indices),
                local=local, mapping=mapping, scales=block["line_factors"][:, local[0]],
                columns=np.asarray(profile["number_column_cm2"])*np.asarray(profile["vmr"][species]),
                self_pressure=np.asarray(profile["pressure_bar"])*np.asarray(profile["vmr"][species])))
            counts.append(dict(order=block["order"], species=species, n_lines=len(indices)))
        bases.append(order_bases)

    def evaluate(q, strength_z=z, self_width=False):
        output = []
        for block, order_bases in zip(context["blocks"], bases):
            total = np.zeros_like(block["baseline_tau"])
            block_z = strength_z[block["parameter_slice"]]
            for s, basis in enumerate(order_bases):
                factors = np.ones(basis["n_lines"])
                factors[basis["mapping"]] += block["upper"][basis["local"]]*block_z[basis["local"]]
                if factors.min() < -1e-10:
                    raise ValueError("Negative line intensity in pressure diagnostic")
                factors = jnp.asarray(np.maximum(factors, 0.))
                tau = np.zeros(len(block["grid"]))
                self_pressure = basis["self_pressure"] if self_width else np.zeros(len(profile["pressure_bar"]))
                for t, pressure, ps, column in zip(profile["temperature_K"], profile["pressure_bar"],
                                                    self_pressure, basis["columns"]):
                    tau += column*np.asarray(basis["layer"](t, pressure, ps, jnp.asarray(q[2*s:2*s+2]), factors))
                total += basis["scales"][:, None]*tau
            intrinsic = jnp.asarray(block["shifted_star"])*jnp.exp(-jnp.asarray(block["airmass"])[:, None]*total)
            def response(row, width, rv):
                core = sampling(block["target"], block["grid"], ipgauss(row, block["velocity"], width), rv)
                wing = sampling(block["target"], block["grid"], ipgauss(row, block["velocity"], 2*width), rv)
                return (1-block["wing_fraction"])*core+block["wing_fraction"]*wing
            output.append(np.asarray(jax.vmap(response)(intrinsic, block["widths"], block["offsets"])))
        return np.array(output)

    baseline = evaluate(np.zeros(4))
    reconstruction_error = float(np.max(abs(baseline-expected)))
    np.testing.assert_allclose(baseline, expected, atol=1e-10, rtol=0)
    baseline_metrics, residual = metrics(context, baseline)
    print("Baseline reconstructed:", reconstruction_error, "elapsed", time.monotonic()-started, flush=True)
    cases, components, directions = {}, {}, []
    for i, name in enumerate(NAMES):
        q = np.zeros(4); q[i] = STEPS[i]
        pair = []
        for sign in [-1, 1]:
            component = evaluate(sign*q)
            tag = f"{name}_{'minus' if sign < 0 else 'plus'}"
            m, r = metrics(context, component)
            cases[tag] = dict(parameters=(sign*q).tolist(), metrics=m)
            components[f"component_{tag}"] = component
            pair.append(component)
            print(tag, "evaluation RMS", m["evaluation_score"]["rms_percent"], flush=True)
        # Unit coordinate here means one declared trial scale (0.1).
        directions.append(context["project"]((pair[1]-pair[0])/2))
    directions = np.stack(directions, axis=-1)
    mask = context["allowed"]
    train_keep = np.broadcast_to(mask[:, None], residual[:, :5].shape)
    matrix = directions[:, :5][train_keep]
    y = residual[:, :5][train_keep]
    sig = WORKING_SIGMA
    fit = lsq_linear(np.vstack([matrix/sig, np.eye(4)]), np.r_[y/sig, np.zeros(4)],
                     bounds=(-3., 3.), tol=1e-12)
    q_best = STEPS*fit.x
    exact = evaluate(q_best)
    m, r = metrics(context, exact)
    cases["global_width_fit_fixed_strengths"] = dict(parameters=q_best.tolist(),
        scaled_parameters=fit.x.tolist(), fit_success=bool(fit.success), metrics=m,
        linear_prediction_max_error=float(np.max(abs(context["project"](exact-baseline)-directions@fit.x))))
    components["component_global_width_fit_fixed_strengths"] = exact
    print("Global fit", q_best, "evaluation RMS", m["evaluation_score"]["rms_percent"], flush=True)

    # Profile the already fitted strengths locally with their adopted weak-line prior.
    strength_matrix = np.zeros((len(y), len(z)))
    row_offset = 0
    for b, block in enumerate(context["blocks"]):
        rows = 5*int(mask[b].sum())
        derivative = -sig*np.asarray(block["data_jacobian"](z[block["parameter_slice"]],
                                                  jnp.asarray(context["y_over_c"][b, :5])))
        strength_matrix[row_offset:row_offset+rows, block["parameter_slice"]] = derivative
        row_offset += rows
    widths = 1+9*contrast**2/(contrast**2+.005**2)
    full_matrix = np.column_stack([strength_matrix, matrix])
    penalties = np.diag(np.r_[1/widths, np.ones(4)])
    full_y = np.r_[y/sig, -z/widths, np.zeros(4)]
    upper_fraction = np.array([r["uncertainty_upper"] for r in context["records"]])
    lower = np.r_[-1/upper_fraction-z, np.full(4, -3.)]
    upper = np.r_[np.full(len(z), np.inf), np.full(4, 3.)]
    joint = lsq_linear(np.vstack([full_matrix/sig, penalties]), full_y,
                       bounds=(lower, upper), tol=1e-11, max_iter=200)
    q_joint = STEPS*joint.x[-4:]
    exact_joint = evaluate(q_joint, z+joint.x[:-4])
    jm, jr = metrics(context, exact_joint)
    cases["local_joint_width_strength_fit"] = dict(parameters=q_joint.tolist(),
        scaled_parameters=joint.x[-4:].tolist(), fit_success=bool(joint.success), metrics=jm,
        maximum_strength_z_change=float(np.max(abs(joint.x[:-4]))),
        linear_prediction_training_error=float(np.sqrt(np.mean((
            context["project"](exact_joint-baseline)[:, :5][train_keep]-full_matrix@joint.x)**2))))
    components["component_local_joint_width_strength_fit"] = exact_joint
    print("Local joint", q_joint, "evaluation RMS", jm["evaluation_score"]["rms_percent"], flush=True)
    self_component = evaluate(np.zeros(4), self_width=True)
    sm, sr = metrics(context, self_component)
    cases["reference_self_width_only"] = dict(metrics=sm,
        maximum_model_change=float(np.max(abs(self_component-baseline))),
        note="Reference VMR self pressure; epoch scale changes column only; line centers held fixed")
    components["component_reference_self_width_only"] = self_component
    norms = np.linalg.norm(matrix, axis=0)
    cosine = matrix.T@matrix/(norms[:, None]*norms[None, :])
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), baseline_case="fixed_contrast10",
        parameter_names=NAMES, trial_scales=STEPS.tolist(), scaled_parameter_bounds=[-3, 3],
        parameter_prior="Unit Gaussian penalty in the declared trial coordinates; exploratory modeling choice",
        baseline_metrics=baseline_metrics, reconstruction_max_error=reconstruction_error,
        opacity_line_counts=counts, cases=cases,
        training_response_cosine=cosine.tolist(),
        training_response_residual_cosine=(matrix.T@y/(norms*np.linalg.norm(y))).tolist(),
        seconds=time.monotonic()-started,
        limitations=["Global species-wide perturbations cannot diagnose line-specific broadening errors.",
                     "Coefficients are a one-step local linearized training fit, followed by exact nonlinear evaluation.",
                     "No prior widths or perturbations are selected using evaluation spectra.",
                     "The local strength profile uses the adopted strength priors and positivity bounds, but is not a full nonlinear refit.",
                     "TP, columns, IP, wavelength solution and polynomial masks remain fixed.",
                     "Self widths use reference VMR and the same n_air exponent; centers and column scaling stay unchanged.",
                     "These are conditional sensitivity tests, not measured HITRAN errors or an adopted new model."],
        source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [Path(__file__), ROOT/"fit_line_strength.py", ROOT/"hitran_cache.py",
                         ROOT/"cache/H2O.par", ROOT/"cache/CO2.par", OUT/"quadratic_strength_fit.json",
                         OUT/"quadratic_strength_fit.npz"]+context["source_files"]})
    (OUT/"pressure_sensitivity.json").write_text(json.dumps(result, indent=2)+"\n")
    np.savez_compressed(OUT/"pressure_sensitivity.npz", component_baseline=baseline,
        filtered_residual_baseline=residual, filtered_scaled_directions=directions,
        global_parameters=q_best, local_joint_parameters=q_joint,
        local_joint_strength_z=z+joint.x[:-4], wavelength_um=context["wave"],
        included_mask=context["allowed"], score_mask=context["score"], **components)
    lines=["Exploratory pressure broadening sensitivity at adopted fixed_contrast10", "",
           "All RMS values are percentages of the original continuum on the same evaluation mask.",
           "All fitting coefficients are selected from the five training exposures only.", "",
           f"Baseline evaluation RMS: {baseline_metrics['evaluation_score']['rms_percent']:.8f}%", ""]
    for name, case in cases.items():
        lines.append(f"{name}: training {case['metrics']['training']['rms_percent']:.8f}%, evaluation {case['metrics']['evaluation_score']['rms_percent']:.8f}%")
        if "parameters" in case:
            lines.append(f"  parameters {dict(zip(NAMES, case['parameters']))}")
    lines += ["", "Limitations:"]+result["limitations"]+[""]
    (OUT/"PRESSURE_SENSITIVITY_REPORT.txt").write_text("\n".join(lines))
    print("Completed in", result["seconds"], "s", flush=True)


if __name__ == "__main__":
    main()
