"""Compare matched-stellar numerical precision controls without refitting."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
import pandas as pd

from blase_stellar import build_clone
from fit_blase_stellar import forward_functions
from fit_line_strength import build_context

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def rms(values):
    return float(100 * np.sqrt(np.mean(np.asarray(values) ** 2)))


def main():
    paths = [OUT / "order7_polish_initial.npz", OUT / "order7_polish_fixed_q.npz"]
    controls = []
    for path in paths:
        with np.load(path) as saved:
            controls.append({key: saved[key] for key in saved.files})
    np.testing.assert_array_equal(controls[0]["q"], controls[1]["q"])
    context = build_context("order7_physical_e10", "order7_physical_e", ("H2O", "CH4", "N2O", "CO", "CO2"))
    assert len(context["blocks"]) == 1
    block = context["blocks"][0]
    clone = build_clone(np.asarray(block["grid"]), block["order"])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    relative_rv = info.iloc[context["exposures"]]["RV [km/s]"].to_numpy() - info.iloc[0]["RV [km/s]"]
    functions = forward_functions(block, clone, context["physical_parameters"], relative_rv)
    components = [np.asarray(functions["full"](saved["q"], saved["z"]))[None] for saved in controls]
    wave = context["wave"][0]
    x = 2 * (np.log(wave) - np.min(np.log(wave))) / np.ptp(np.log(wave)) - 1
    orthogonal, _ = np.linalg.qr(np.polynomial.chebyshev.chebvander(x, 5))

    def project(value):
        return value - (value @ orthogonal) @ orthogonal.T

    score = context["score"][0]
    delta = components[1] - components[0]
    filtered_delta = project(delta)
    residuals = [project(context["y_over_c"] - component) for component in components]
    rows = {}
    for name, residual in zip(["coarse", "tight"], residuals):
        rows[name] = dict(evaluation_score_rms_percent=rms(residual[:, 5:][:, :, score]),
                          evaluation_all_rms_percent=rms(residual[:, 5:]),
                          training_all_rms_percent=rms(residual[:, :5]))
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(), display_order=7, cube_index=5,
        comparison="Same stellar amplitudes and physical model; only accuracy of the telluric-strength solution changes",
        solver_settings=dict(outer=dict(ftol=1e-9, xtol=1e-9, gtol=1e-7),
            coarse=dict(atol=1e-8, btol=1e-8, maxiter=200, accepted_nfev=70,
                        status="Interrupted near-optimum checkpoint; convergence is not asserted"),
            tight=dict(atol=1e-10, btol=1e-10, maxiter=1000, nfev=7,
                       status="Converged by the unchanged outer ftol criterion")),
        final_null_refinement_result=False, stellar_parameters_bit_identical=True,
        max_absolute_z_change=float(np.max(np.abs(controls[1]["z"] - controls[0]["z"]))),
        max_absolute_component_difference_percent=float(100 * np.max(np.abs(delta))),
        max_absolute_q5_difference_percent=float(100 * np.max(np.abs(filtered_delta))),
        q5_evaluation_score_difference_rms_percent=rms(filtered_delta[:, 5:][:, :, score]),
        q5_evaluation_all_difference_rms_percent=rms(filtered_delta[:, 5:]),
        residual_metrics=rows,
        limitations=["These matched-stellar controls precede later null refinements and are not the final adopted numerical baseline.",
                     "Large changes in weak or blended line coefficients can coexist with much smaller changes in the spectrum."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths + [Path(__file__), ROOT / "fit_blase_stellar.py", ROOT / "fit_line_strength.py"]})
    np.savez_compressed(OUT / "order7_numerical_precision.npz", component_coarse=components[0],
                        component_tight=components[1], residual_coarse=residuals[0], residual_tight=residuals[1],
                        wavelength_um=context["wave"], exposures=context["exposures"], score_mask=context["score"])
    (OUT / "order7_numerical_precision.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
