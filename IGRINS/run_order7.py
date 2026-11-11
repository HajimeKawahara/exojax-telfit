"""Fit the full released wavelength range of requested Order 7."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TRAIN = [0, 19, 39, 59, 78]
EVALUATION = [10, 30, 50, 70]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(initial: Path | None = None) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    from prepare_order7_scope import main as prepare_scope
    prepare_scope()
    protected = [OUT / "adopted_telluric_model.json", OUT / "adopted_blase_telluric_fit.npz",
                 OUT / "adopted_telluric_fit.npz"]
    protected = [path for path in protected if path.exists()]
    before = {str(path.relative_to(ROOT)): digest(path) for path in protected}
    arguments = ["--orders", "5", "--pixels", "0", "1848", "--train",
                 *map(str, TRAIN), "--atmosphere", "profile", "--layers", "20",
                 "--dv", ".75", "--shift", "--ip", "mixture", "--ip-epoch",
                 "--species", "H2O", "CH4", "N2O", "CO", "CO2", "--fix-star",
                 "--max-nfev", "150"]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
               JAX_ENABLE_X64="true", MPLCONFIGDIR="/tmp/igrins-matplotlib")
    plan = dict(created_utc=datetime.now(timezone.utc).isoformat(), requested_order_label=7,
                released_cube_index=5, requested_full_range_um=[2.26448634, 2.29592734],
                stored_range_um=[2.266221670879659, 2.294583949845698],
                mapping="Wavelength match; edge pixels absent from the released cube are not reconstructed",
                pixels=[0, 1848], training_exposures=TRAIN, evaluation_exposures=EVALUATION,
                absorption_mask=False, physical_species=["H2O", "CH4", "N2O", "CO", "CO2"],
                commands=[], results=[], protected_sha256=before)
    initial = initial or ROOT.parent / "config/order7_initial.json"
    if not initial.is_file():
        raise FileNotFoundError(f"Missing physical initialization: {initial}")
    for exposure in EVALUATION:
        tag = f"order7_physical_e{exposure}"
        command = [sys.executable, str(ROOT / "fit_joint.py"), *arguments,
                   "--holdout", str(exposure), "--initial", str(initial), "--tag", tag]
        if exposure != EVALUATION[0]:
            command.append("--freeze-shared")
        plan["commands"].append(command)
        (OUT / "order7_physical_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(f"Starting Order 7 / cube index 5, evaluation frame {exposure}", flush=True)
        with (OUT / f"{tag}.log").open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((OUT / f"{tag}.json").read_text())
        assert result["training_fit"]["success"] and result["heldout_calibration_fit"]["success"]
        with np.load(OUT / f"{tag}.npz") as saved:
            assert saved["orders"].tolist() == [5]
            assert saved["included_mask"].shape == (1, 1848) and saved["included_mask"].all()
            assert saved["exposures"].tolist() == TRAIN + [exposure]
            assert np.all(np.isfinite(saved["model"])) and np.all(saved["continuum"] > 0)
        summary = dict(exposure=exposure, tag=tag, seconds=result["seconds"],
                       training_rms_percent=100 * result["training_metrics"]["rms"],
                       score_rms_percent=100 * result["heldout_score_metrics"]["rms"],
                       bounds_hit=result["bounds_hit"],
                       gradient_error=result["heldout_calibration_fit"]["jacobian_relative_error"])
        plan["results"].append(summary)
        assert before == {str(path.relative_to(ROOT)): digest(path) for path in protected}
        (OUT / "order7_physical_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        initial = OUT / "order7_physical_e10.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path,
                        help="Physical parameter seed; defaults to config/order7_initial.json")
    main(parser.parse_args().initial)
