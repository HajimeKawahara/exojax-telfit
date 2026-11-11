"""Export spectra from saved precision-test parameters without refitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

from check_line_shape_planet import water_signal
from fit_line_shapes import build_context, predict_components

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="line_shape_precision")
    args = parser.parse_args()
    source_path = OUT/f"{args.source}.json"
    result = json.loads(source_path.read_text())
    context = build_context()
    arrays = {key:context[key] for key in ["wave", "allowed", "score", "orders", "exposures",
                                          "y_over_c", "reference_component"]}
    keys = {}
    for name, setting in result["settings"].items():
        baseline = predict_components(context, np.asarray(setting["baseline_theta"]))
        baseline_residual = context["project"](context["y_over_c"]-baseline)
        null = predict_components(context, np.asarray(setting["null_history"][-1]["fitted_theta"]))
        arrays[f"baseline_{name}"] = baseline
        arrays[f"null_{name}"] = null
        arrays[f"null_response_{name}"] = context["project"](baseline-null)
        keys[name] = []
        for index, case in enumerate(setting["water"]["cases"]):
            label = f"{name}_case{index}"
            signal = water_signal(context, case)
            injected = predict_components(context, np.asarray(case["fitted_theta"]))
            arrays[f"signal_{label}"] = signal
            arrays[f"prediction_{label}"] = injected
            arrays[f"expected_{label}"] = context["project"](signal)
            arrays[f"response_{label}"] = context["project"](context["y_over_c"]+signal-injected)-baseline_residual
            keys[name].append(dict(label=label, case_index=index,
                                   **{k:case[k] for k in ["fwhm_kms", "kp_kms", "depth"]}))
    destination = OUT/f"{args.source}_arrays.npz"
    np.savez_compressed(destination, **arrays)
    paths = [Path(__file__), ROOT/"fit_line_shapes.py", ROOT/"check_line_shape_planet.py",
             source_path, OUT/"line_shape_fit.json", OUT/"line_shape_fit.npz"]+context["source_files"]
    meta = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source_path.relative_to(ROOT)),
                operation="Forward evaluation of saved baseline/null/injected parameters; no optimization",
                cases=keys, arrays_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    destination.with_suffix(".json").write_text(json.dumps(meta, indent=2)+"\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
