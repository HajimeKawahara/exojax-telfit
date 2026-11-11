"""Inspect the supplied IGRINS cube without altering the input data."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d, percentile_filter


def main() -> None:
    root = Path(__file__).resolve().parent
    output = root / "results"
    output.mkdir(exist_ok=True)
    with (root / "data/cube14v3.pic").open("rb") as stream:
        wavelength, flux = pickle.load(stream)
    metadata = {}
    for path in sorted((root / "data").glob("*_info.csv")):
        frame = pd.read_csv(path)
        metadata[path.name] = {
            "rows": len(frame),
            "ranges": {
                column: [float(frame[column].min()), float(frame[column].max())]
                for column in frame.columns
            },
        }
    median_flux = np.median(flux, axis=1)
    envelope = gaussian_filter1d(
        percentile_filter(median_flux, percentile=95, size=(1, 101)),
        sigma=30, axis=-1,
    )
    orders = []
    for order in range(flux.shape[0]):
        useful = np.flatnonzero(envelope[order] > 0.3 * envelope[order].max())
        orders.append({
            "order_index": order,
            "wavelength_min_max_assumed_um": wavelength[order, [0, -1]].tolist(),
            "median_flux": float(np.median(flux[order])),
            "nonpositive_pixels_all_exposures": int((flux[order] <= 0).sum()),
            "envelope_above_30_percent_pixel_bounds": useful[[0, -1]].tolist(),
        })
    inventory = {
        "target": "WASP-77A b",
        "source_url": "https://zenodo.org/records/10382053",
        "paper_url": "https://arxiv.org/html/2312.13069v1",
        "source_md5": "3fd41560b7dbcc60851ef3a0972c852f",
        "local_md5": hashlib.md5((root / "data/cube14v3.pic").read_bytes()).hexdigest(),
        "source": "data/cube14v3.pic",
        "structure": "list [wavelength, flux]",
        "wavelength_shape": list(wavelength.shape),
        "flux_shape": list(flux.shape),
        "axes": {"wavelength": ["order", "pixel"], "flux": ["order", "exposure", "pixel"]},
        "finite_wavelength": bool(np.isfinite(wavelength).all()),
        "finite_flux": bool(np.isfinite(flux).all()),
        "wavelength_unit": "micron (inferred from numerical range; no stored unit)",
        "flux_unit": "unknown; unnormalized counts-like values",
        "matching_metadata": "20201214_info.csv (79 pre-eclipse frames; agrees with paper Table 2)",
        "paper_reduction_summary": "Section II reports common-grid alignment to the last frame, count normalization, order removal, and edge trimming before SVD. The supplied count-like spectra retain strong stellar/telluric features and are inferred to be before SVD, rather than planetary residual spectra.",
        "wavelength_frame_note": "Paper equation 1 adds the observatory-frame barycentric velocity when constructing model line-of-sight velocities. This supports an observatory-frame interpretation, but the pickle contains no explicit frame or air/vacuum label.",
        "missing_information": ["per-pixel uncertainties", "air/vacuum wavelength convention", "explicit wavelength velocity frame", "flux units", "exact processing stage of this cube"],
        "metadata": metadata,
        "orders": orders,
    }
    (output / "data_inventory.json").write_text(json.dumps(inventory, indent=2) + "\n")
    fig, axes = plt.subplots(6, 2, figsize=(15, 15))
    for row, order in enumerate([3, 10, 14, 25, 34, 40]):
        wave = wavelength[order]
        raw, relative = axes[row]
        raw.plot(wave, flux[order, 0], lw=0.7, label="Exposure 0")
        raw.plot(wave, median_flux[order], lw=0.8, alpha=0.8, label="Median of 79 exposures")
        raw.plot(wave, envelope[order], color="black", lw=0.8, alpha=0.6, label="Diagnostic upper envelope")
        raw.set_ylabel("Flux (input units)")
        raw.set_title(f"Order index {order}: input flux")
        scale = np.median(flux[order, 0, 600:1500] / median_flux[order, 600:1500])
        relative.plot(wave, flux[order, 0] / envelope[order] / scale, lw=0.6, label="Exposure 0 (rescaled)")
        relative.plot(wave, median_flux[order] / envelope[order], lw=0.7, alpha=0.8, label="Median / upper envelope")
        relative.set_ylim(-0.02, 1.15)
        relative.set_ylabel("Relative flux")
        relative.set_title("Diagnostic normalization (not a physical continuum)")
        for ax in (raw, relative):
            ax.set_xlabel("Wavelength (assumed micron)")
            ax.axvspan(wave[600], wave[1499], color="gray", alpha=0.09)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle("IGRINS cube14v3: representative orders; shading marks pixels 600:1500", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(output / "data_overview.png", dpi=150)
    plt.close(fig)
    print(output / "data_overview.png")
    print(output / "data_inventory.json")


if __name__ == "__main__":
    main()
