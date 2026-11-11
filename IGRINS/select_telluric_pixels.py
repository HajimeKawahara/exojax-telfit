"""Freeze conservative stellar masks before comparing telluric models.

Selection uses the existing exposures 0, 78, and 39 only. The known residual
regions make this an exploratory selection; exposure 20 is reserved for a
subsequent conditional validation and is not used here.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np
from scipy.ndimage import binary_dilation

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
STAR_DEPTH_LIMIT = .02
TELLURIC_TRANSMISSION_FLOOR = .15
PAD_PIXELS = 2
LINE_DEPTH_LIMIT = .02
CONTINUUM_DEPTH_LIMIT = .01


def intervals(mask, wave, original_pixel):
    """List retained contiguous intervals in original detector coordinates."""
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    return [dict(pixel_min=int(original_pixel[g[-1]]),
                 pixel_max=int(original_pixel[g[0]]),
                 wavelength_min_um=float(wave[g[-1]]),
                 wavelength_max_um=float(wave[g[0]]), count=len(g))
            for g in groups if len(g)]


def main():
    paths = [OUT / "residual_components.npz", OUT / "residual_regions.csv",
             OUT / "joint_star_profile.json", Path(__file__)]
    with np.load(paths[0]) as source:
        components = {key: source[key] for key in source.files}
    with paths[1].open() as stream:
        regions = list(csv.DictReader(stream))
    fit = json.loads(paths[2].read_text())
    lo, hi = fit["pixels"]
    original_pixel = np.arange(lo, hi)[::-1]
    orders, wave = components["orders"], components["wavelength_um"]
    # Validate coordinate mapping without using any additional exposure flux.
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, _ = pickle.load(stream)
    if not np.array_equal(wave, wavelength[orders][:, original_pixel]):
        raise ValueError("Component wavelengths do not match original cube pixels.")
    if set(components["exposures"]) != {0, 78, 39}:
        raise ValueError("Selection must use the existing discovery exposures only.")

    stellar_depth = np.max(1 - components["stellar_only"], axis=1)
    telluric_depth = np.max(1 - components["telluric_only"], axis=1)
    known_bad = np.zeros_like(wave, dtype=bool)
    mask = np.empty_like(known_bad)
    for b, order in enumerate(orders):
        for region in regions:
            if (int(region["order"]) == order
                    and region["attribution"] != "Stellar/telluric blend"):
                known_bad[b] |= ((wave[b] >= float(region["wavelength_min_um"]))
                                & (wave[b] <= float(region["wavelength_max_um"])))
        excluded = binary_dilation(
            (stellar_depth[b] > STAR_DEPTH_LIMIT) | known_bad[b],
            iterations=PAD_PIXELS)
        mask[b] = (~excluded
                   & (1 - telluric_depth[b] > TELLURIC_TRANSMISSION_FLOOR))

    line_masks = {species: mask & (np.max(
        components[f"{species}_marginal_absorption"], axis=1) > LINE_DEPTH_LIMIT)
                  for species in ["H2O", "CO2"]}
    telluric_lines = mask & (telluric_depth > LINE_DEPTH_LIMIT)
    continuum_anchors = mask & (telluric_depth < CONTINUUM_DEPTH_LIMIT)
    pixel = np.arange(hi - lo)
    calibration = ((pixel // 80) % 2 == 0) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    scoring = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    np.savez_compressed(
        OUT / "telluric_mask.npz", orders=orders, wavelength_um=wave,
        mask=mask, original_pixel=original_pixel, H2O_mask=line_masks["H2O"],
        CO2_mask=line_masks["CO2"], telluric_line_mask=telluric_lines,
        continuum_anchor_mask=continuum_anchors,
        selection_exposures=components["exposures"],
        stellar_depth=stellar_depth, telluric_depth=telluric_depth)

    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        selection_exposures=components["exposures"].tolist(),
        reserved_holdout_exposure=20, pixels=[lo, hi],
        rules=dict(maximum_stellar_only_depth=STAR_DEPTH_LIMIT,
                   minimum_telluric_only_transmission=TELLURIC_TRANSMISSION_FLOOR,
                   exclusion_padding_pixels=PAD_PIXELS,
                   known_exclusions="All non-blend regions in residual_regions.csv",
                   informative_marginal_line_depth=LINE_DEPTH_LIMIT,
                   continuum_anchor_maximum_telluric_depth=CONTINUUM_DEPTH_LIMIT),
        caveats=[
            "Post-hoc discovery mask; selected-region discovery RMS is descriptive.",
            "Weak predicted stellar absorption does not prove a perfect stellar model.",
            "Line and continuum masks are informational subsets, not additional fit cuts.",
            "Order 13 has few pure continuum pixels; normalization remains model-dependent.",
            "A new exposure tests time transfer, not independence from wavelength selection."],
        source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in paths},
        wavelength_mapping_validated=True, per_order=[])
    for b, order in enumerate(orders):
        report["per_order"].append(dict(
            order=int(order), retained=int(mask[b].sum()),
            calibration=int((mask[b] & calibration).sum()),
            score=int((mask[b] & scoring).sum()),
            H2O=int(line_masks["H2O"][b].sum()),
            CO2=int(line_masks["CO2"][b].sum()),
            telluric_lines=int(telluric_lines[b].sum()),
            continuum_anchors=int(continuum_anchors[b].sum()),
            intervals=intervals(mask[b], wave[b], original_pixel)))
    (OUT / "telluric_mask.json").write_text(json.dumps(report, indent=2) + "\n")

    fig, axes = plt.subplots(len(orders), 1, figsize=(14, 8), constrained_layout=True)
    for b, (order, ax) in enumerate(zip(orders, axes)):
        ax.fill_between(wave[b], 0, 100, where=~mask[b], color=".9", step="mid",
                        label="Excluded pixels")
        ax.plot(wave[b], 100 * telluric_depth[b], color="tab:blue", lw=1,
                label="Maximum telluric-only depth")
        ax.plot(wave[b], 100 * stellar_depth[b], color="tab:red", lw=1,
                label="Maximum stellar-only depth")
        ax.plot(wave[b, mask[b]], np.full(mask[b].sum(), -3), "|", color="tab:green",
                ms=7, label="Retained fit pixels")
        ax.axhline(100 * STAR_DEPTH_LIMIT, color="tab:red", ls=":", lw=.7)
        ax.set(xlim=(wave[b, -1], wave[b, 0]), ylim=(-6, 100), ylabel="Absorption (%)",
               title=f"Order {order}: {mask[b].sum()} / {mask.shape[1]} pixels retained")
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Observed vacuum wavelength (micron; assumed wavelength convention)")
    fig.suptitle("Fixed telluric comparison mask | discovery exposures 0, 78, 39")
    fig.savefig(OUT / "telluric_mask.png", dpi=160)
    plt.close(fig)
    print(json.dumps([{k: v for k, v in item.items() if k != "intervals"}
                      for item in report["per_order"]], indent=2))


if __name__ == "__main__":
    main()
