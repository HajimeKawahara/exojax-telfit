"""Check the control's training objective at finite physical parameter steps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from fit_transformerpayne_labels import CONTEXT, OUT, Problem, STELLAR_NAMES


def main():
    result = json.loads((OUT / "transformerpayne_stellar_control.json").read_text())
    with np.load(CONTEXT) as saved:
        data = {key: saved[key] for key in saved.files}
    with np.load(OUT / "transformerpayne_stellar_control.npz") as saved:
        data["included_mask"] = saved["calibration_mask"].copy()
        theta = saved["parameters_calibrated"].copy()
    problem = Problem(data)
    base_residual = problem.residual(theta)
    objective = float(base_residual @ base_residual)
    np.testing.assert_allclose(objective, result["fits"]["stellar"]["objective"], atol=1e-7, rtol=1e-10)
    jacobian = problem.jacobian(theta)[:, :6]
    gradient = 2 * jacobian.T @ base_residual
    steps = [10, .01, .01, .02, .02, .02]
    rows = []
    for i, (name, step) in enumerate(zip(STELLAR_NAMES, steps)):
        for sign in [-1, 1]:
            trial = theta.copy()
            trial[i] += sign * step
            if not problem.lower[i] <= trial[i] <= problem.upper[i]:
                continue
            residual = problem.residual(trial)
            value = float(residual @ residual)
            row = dict(parameter=name, delta=sign*step, objective=value,
                       objective_change=value-objective, relative_change=(value-objective)/objective,
                       analytic_first_order_change=float(gradient[i]*sign*step))
            rows.append(row)
            print(json.dumps(row), flush=True)
    best = min(rows, key=lambda row: row["objective"])
    report = dict(training_only=True, no_refit=True, calibration_threshold=.95,
                  baseline_objective=objective, physical_steps=dict(zip(STELLAR_NAMES, steps)),
                  gradient=gradient.tolist(), trials=rows, best_trial=best,
                  explanation="Each coordinate is perturbed separately; this is a finite-step local check, not a proof of global optimality. No evaluation residual is used.",
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "transformerpayne_stellar_control_sensitivity.json").write_text(json.dumps(report, indent=2)+"\n")
    path = OUT / "TRANSFORMERPAYNE_STELLAR_CONTROL_REPORT.txt"
    lines = ["", "Training-only finite-step sensitivity check:",
             "Each free stellar parameter was perturbed separately, using 10 K / 0.01 dex / 0.01 dex / 0.02 km/s / 0.02 km/s / 0.02 km/s.",
             "Only feasible directions were tested at the bounds. Strengths remained fixed; evaluation data was not used."]
    for row in rows:
        lines.append(f"{row['parameter']} {row['delta']:+g}: objective change {row['objective_change']:+.8f} ({100*row['relative_change']:+.8f}%).")
    lines += [report["explanation"]]
    path.write_text(path.read_text()+"\n".join(lines)+"\n")
    print("Best finite-step trial", json.dumps(best), flush=True)


if __name__ == "__main__":
    main()
