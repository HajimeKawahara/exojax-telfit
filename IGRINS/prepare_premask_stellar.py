"""Reconstruct the adopted frame-30 stellar spectrum without telluric opacity."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib.util import find_spec
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from line_strength_support import native_grid
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EXPOSURE = 30


def numpy_gaussian(flux, velocity, sigma):
    """Independent Gaussian convolution for the reconstruction check."""
    kernel = np.exp(-0.5 * (velocity / sigma) ** 2)
    return fftconvolve(flux, kernel / kernel.sum(), mode="same")


def main():
    source = OUT / f"nonstellar_control_e{EXPOSURE}.json"
    meta = json.loads(source.read_text())
    base = json.loads((OUT / "motion_clean_control.json").read_text())
    p = meta["parameters"]
    for key, value in base["parameters"].items():
        if key in p:
            assert p[key] == value
    with np.load(OUT / "adopted_telluric_fit.npz") as adopted:
        orders, wave = adopted["orders"], adopted["wavelength_um"]
        assert EXPOSURE in adopted["exposures"]
    with np.load(OUT / f"nonstellar_control_e{EXPOSURE}.npz") as original:
        np.testing.assert_array_equal(wave, original["wavelength_um"])
        np.testing.assert_array_equal(orders, original["orders"])
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    rv = p["stellar_rv_reference_kms"] + float(
        info.iloc[EXPOSURE]["RV [km/s]"] - info.iloc[0]["RV [km/s]"])
    fraction = p["ip_wing_fraction"]
    spectra, records = [], []
    for order, wavelength in zip(orders, wave):
        grid = native_grid(wavelength, meta["dv_kms"])
        dv = float(c * np.log(grid[1] / grid[0]))
        velocity = jnp.arange(-int(60 / dv), int(60 / dv) + 1) * dv
        raw = load_stellar(grid)
        star = convolve_rigid_rotation(jnp.asarray(raw), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, p["stellar_extra_sigma_kms"])
        shifted = sampling(grid, grid, star, rv)
        width = p[f"ip_sigma_o{order}_kms"] * np.exp(p[f"log_ip_width_e{EXPOSURE}"])
        offset = p[f"calibration_rv_o{order}_kms"] + p[f"calibration_rv_e{EXPOSURE}_kms"]
        target = (1e4 / wavelength) * (
            1 + p[f"stretch_o{order}_kms"] * np.linspace(-1, 1, len(wavelength)) / c)
        result = np.asarray(
            (1 - fraction) * sampling(target, grid, ipgauss(shifted, velocity, width), offset)
            + fraction * sampling(target, grid, ipgauss(shifted, velocity, 2 * width), offset))

        # Rebuild all broadening and shifts without ExoJAX/JAX response functions.
        v = np.asarray(velocity)
        z = np.maximum(1 - (v / 4.) ** 2, 0.)
        rotation = 2 * (1 - .6) * np.sqrt(z) + .5 * np.pi * .6 * z
        independent = fftconvolve(raw, rotation / rotation.sum(), mode="same")
        independent = numpy_gaussian(independent, v, p["stellar_extra_sigma_kms"])
        independent = np.interp(grid * (1 + rv / c), grid, independent)
        independent = (
            (1 - fraction) * np.interp(target * (1 + offset / c), grid,
                                      numpy_gaussian(independent, v, width))
            + fraction * np.interp(target * (1 + offset / c), grid,
                                   numpy_gaussian(independent, v, 2 * width)))
        np.testing.assert_allclose(result, independent, rtol=0, atol=2e-12)
        assert np.all(np.isfinite(result)) and np.all(result > 0)
        spectra.append(result)
        records.append(dict(order=int(order), ip_sigma_kms=float(width),
                            calibration_rv_kms=float(offset),
                            minimum_flux=float(result.min()),
                            maximum_flux=float(result.max()),
                            numpy_max_absolute_difference=float(np.max(np.abs(result - independent)))))

    output = OUT / "premask_stellar_component.npz"
    np.savez_compressed(output, orders=orders, wavelength_um=wave,
                        exposure=np.array(EXPOSURE), stellar_only=np.array(spectra))
    sources = [Path(__file__), source, OUT / "motion_clean_control.json",
               OUT / "adopted_telluric_fit.npz", OUT / f"nonstellar_control_e{EXPOSURE}.npz",
               ROOT / "data/20201214_info.csv", ROOT / "cache/stellar_phoenix.npz",
               ROOT / "stellar_template.py", ROOT / "line_strength_support.py",
               ROOT / "fit_line_strength.py"]
    source_hashes = {str(path.relative_to(ROOT.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    for module in ["exojax.postproc.response", "exojax.postproc.spin_rotation",
                   "exojax.signal.convolve"]:
        specification = find_spec(module)
        if specification is None or specification.origin is None:
            raise ImportError(f"Cannot locate installed source for {module}")
        source_hashes[module.replace(".", "/") + ".py"] = hashlib.sha256(
            Path(specification.origin).read_bytes()).hexdigest()
    report = dict(
        created_utc=datetime.now(timezone.utc).isoformat(), exposure=EXPOSURE,
        adopted_case="fixed_contrast10", stellar_rv_kms=float(rv),
        definition="PHOENIX stellar-only counterfactual with telluric transmission set to unity; adopted rotation, extra broadening, frame velocity, original two-Gaussian IP, wavelength stretch and sampling",
        normalization="Original normalized PHOENIX pseudo-continuum; dimensionless continuum near one. No degree-five smooth correction or extra continuum is applied.",
        limitation="The stellar-only curve is not an additive component of the convolved stellar-times-telluric spectrum.",
        per_order=records, refit_performed=False,
        source_sha256=source_hashes,
        product_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(output), "checks": records}, indent=2))


if __name__ == "__main__":
    main()
