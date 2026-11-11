"""Exclude one independently diagnosed moving blend from a copy of the mask.

The original mask and frozen final comparison are preserved. The additional
window is based on the trusted Dec06/Dec14 motion diagnostic, not RMS tuning.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def main():
    parent = OUT / "telluric_mask.npz"
    source = OUT / "crossnight_physical.json"
    diagnostic = json.loads(source.read_text())
    fit = next(row for row in diagnostic["fits"]
               if row["target_wavelength_um"] == 1.501046034
               and row["label"] == "06_14" and row["hypothesis"] == "stellar"
               and row["other_night_temperature_offset_K"] == 0)
    if not fit["success"] or fit["bounds_hit_indices"]:
        raise ValueError("The motion-based exclusion requires a converged, unbounded local fit.")
    reference_rv = next(row["median_stellar_rv_kms"] for row in diagnostic["inventory"]
                        if row["night"] == "14")
    rv = pd.read_csv(ROOT / "data/20201214_info.csv")["RV [km/s]"].to_numpy()
    center = fit["target_wavelength_um"] * (1 + fit["parameters"]["reference_center_kms"] / C_KMS)
    half_width = 12.
    lower = center * (1 + (rv.min()-reference_rv-half_width)/C_KMS)
    upper = center * (1 + (rv.max()-reference_rv+half_width)/C_KMS)
    with np.load(parent) as saved:
        data = {key: saved[key] for key in saved.files}
    old = data["mask"].copy()
    b = int(np.flatnonzero(data["orders"] == 40)[0])
    excluded = (data["wavelength_um"][b] >= lower) & (data["wavelength_um"][b] <= upper)
    data["mask"][b, excluded] = False
    for key in ["H2O_mask", "CO2_mask", "telluric_line_mask", "continuum_anchor_mask"]:
        data[key] &= data["mask"]
    target = OUT / "telluric_mask_motion_clean.npz"
    np.savez_compressed(target, **data)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(), order=40,
                  observed_reference_center_um=center, half_width_kms=half_width,
                  wavelength_interval_um=[lower, upper],
                  original_pixels=data["original_pixel"][excluded].tolist(),
                  newly_excluded_included_pixels=int(np.sum(old & ~data["mask"])),
                  retained_pixels=int(data["mask"].sum()),
                  per_order_counts=data["mask"].sum(axis=1).tolist(),
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [parent, source, Path(__file__)]},
                  note="Separate exploratory follow-up mask. Historical frozen validation remains unchanged.",
                  limitations=["Window also overlaps a neighboring positive residual; RMS reduction cannot all be assigned to stellar exclusion.",
                               "The second feature near1.494855um is retained because its local width reaches a bound.",
                               "Compare physical models only on identical masks; do not compare absolute RMS to the original-mask fits as physical improvement."])
    target.with_suffix(".json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
