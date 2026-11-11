"""Derive Order 7 coverage and noise metadata from the downloaded observations."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TRAIN = [0, 19, 39, 59, 78]
EVALUATION = [10, 30, 50, 70]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "data/cube14v3.pic"
    info_path = ROOT / "data/20201214_info.csv"
    with source.open("rb") as stream:
        wavelength, cube = pickle.load(stream)
    info = pd.read_csv(info_path)
    if wavelength.shape != (44, 1848) or cube.shape != (44, 79, 1848):
        raise ValueError("Expected the released 2020-12-14 cube with 44 orders and 79 frames.")
    wave, flux = np.asarray(wavelength[5]), np.asarray(cube[5])
    np.testing.assert_allclose(wave[[0, -1]], [2.266221670879659, 2.294583949845698],
                               rtol=0, atol=1e-10)
    if not np.isfinite(wave).all() or not np.all(np.diff(wave) > 0):
        raise ValueError("Order 7 requires finite ascending stored wavelengths.")
    if not np.isfinite(flux).all() or np.any(flux <= 0):
        raise ValueError("This full-pixel workflow requires finite positive stored fluxes.")
    if len(info) != cube.shape[1]:
        raise ValueError("Observation metadata and cube frame counts differ.")
    snr = info.iloc[EVALUATION]["Med. SNR"].to_numpy(dtype=float)
    if not np.isfinite(snr).all() or np.any(snr <= 0):
        raise ValueError("Evaluation-frame median S/N must be finite and positive.")
    pixel = np.arange(len(wave))
    score = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    calibration = ((pixel // 80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        user_order_label=7, released_cube_index=5,
        user_requested_wavelength_um=[2.26448634, 2.29592734],
        actual_available_wavelength_um=[float(wave.min()), float(wave.max())],
        actual_available_wavenumber_cm1=[float(1e4/wave.max()), float(1e4/wave.min())],
        pixels=[0, len(wave)], stored_pixel_count=len(wave), stored_frame_count=cube.shape[1],
        stored_flux_sample_count=int(flux.size), invalid_wavelength_count=0,
        invalid_flux_count=0, nonpositive_flux_count=0,
        flux_range=[float(flux.min()), float(flux.max())],
        provided_quality_flags=False, provided_variance=False,
        selection="All stored pixels; no residual, stellar-depth, telluric-depth or extra edge exclusions.",
        training_exposures=TRAIN, evaluation_exposures=EVALUATION,
        fit_pixel_count=len(wave), training_sample_count=len(wave)*len(TRAIN),
        evaluation_all_included_sample_count=len(wave)*len(EVALUATION),
        spatial_score_pixel_count=int(score.sum()),
        evaluation_score_sample_count=int(score.sum())*len(EVALUATION),
        spatial_calibration_pixel_count=int(calibration.sum()),
        array_orientation="Fits reverse stored pixels to ascending wavenumber; scope uses stored wavelength order.",
        metadata_noise_proxy=dict(evaluation_snr=snr.tolist(),
                                  pooled_percent=float(np.sqrt(np.mean((100/snr)**2))),
                                  note="One median SNR per frame; not order-specific photon variance."),
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in [source, info_path]},
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT / "order7_scope_audit.json").write_text(json.dumps(result, indent=2)+"\n")
    print(f"Order 7: {len(wave)} pixels, {len(TRAIN)} training and {len(EVALUATION)} evaluation frames")


if __name__ == "__main__":
    main()
