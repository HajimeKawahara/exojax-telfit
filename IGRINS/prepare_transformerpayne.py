"""Prepare a disk-integrated TransformerPayne trial on supported IGRINS orders."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("HF_HOME", str(CACHE / "huggingface"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import numpy as np


def download():
    """Pin the public checkpoint revision and keep every cache under IGRINS."""
    from huggingface_hub import HfApi, hf_hub_download

    repository = "RozanskiT/transformer_payne"
    filename = "TransformerPayneIntensities_v2.pkl"
    info = HfApi(token=False).model_info(repository, files_metadata=True)
    entry = next(item for item in info.siblings if item.rfilename == filename)
    print(f"Downloading {filename}: {entry.size} bytes at {info.sha}", flush=True)
    path = Path(hf_hub_download(repository, filename, revision=info.sha,
                               cache_dir=CACHE / "huggingface", token=False))
    metadata = dict(repository=repository, filename=filename, revision=info.sha,
                    path=str(path), size_bytes=path.stat().st_size,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    downloaded_utc=datetime.now(timezone.utc).isoformat())
    (CACHE / "transformerpayne_checkpoint.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata), flush=True)


def grids():
    """Run in the existing ExoJAX environment to preserve its exact grids."""
    from line_strength_support import native_grid

    meta = json.loads((ROOT / "results/motion_clean_control.json").read_text())
    output = dict(orders=np.array([34, 40]))
    with np.load(ROOT / "results/adopted_telluric_fit.npz") as adopted:
        for order in output["orders"]:
            index = int(np.flatnonzero(adopted["orders"] == order)[0])
            grid = native_grid(adopted["wavelength_um"][index], meta["dv_kms"])
            wave = 1e8 / grid
            if wave.min() < 1500 or wave.max() > 20000:
                raise ValueError("Requested grid exceeds advertised checkpoint coverage.")
            output[f"grid_o{order}"] = grid
    np.savez_compressed(CACHE / "transformerpayne_input_grids.npz", **output)
    print({key: value.shape for key, value in output.items()})


def prepare():
    """Integrate intensity and continuum separately; apply no broadening here."""
    import jax
    from transformer_payne import TransformerPayne

    checkpoint = json.loads((CACHE / "transformerpayne_checkpoint.json").read_text())
    path = Path(checkpoint["path"])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == checkpoint["sha256"]
    emulator = TransformerPayne.from_file(str(path))
    parameters = emulator.to_parameters(dict(logteff=np.log10(5600.), logg=4.5, vmic=1.))
    if not bool(emulator.is_in_bounds(parameters)):
        raise ValueError("Stellar parameters exceed checkpoint bounds.")
    print("Model loaded", emulator.model_definition.architecture_parameters, flush=True)
    start = time.monotonic()
    with np.load(CACHE / "transformerpayne_input_grids.npz") as saved:
        output = {key: saved[key] for key in saved.files}
    records = []
    batch_size = 256

    def flux(logwave, nmu):
        mu, weight = np.polynomial.legendre.leggauss(nmu)
        mu, weight = (mu + 1) / 2, weight / 2
        if mu.min() < float(emulator.min_parameters[-1]) or mu.max() > float(emulator.max_parameters[-1]):
            raise ValueError("Angular quadrature exceeds checkpoint bounds.")
        total = np.zeros((len(logwave), 2))
        for angle, quadrature_weight in zip(mu, weight):
            for left in range(0, len(logwave), batch_size):
                piece = logwave[left:left + batch_size]
                padded = np.pad(piece, (0, batch_size - len(piece)), mode="edge")
                intensity = np.asarray(emulator.intensity(padded, angle, parameters))[:len(piece)]
                if not np.all(np.isfinite(intensity)) or np.any(intensity[:, 1] <= 0):
                    raise ValueError("Non-finite intensity or nonpositive continuum.")
                total[left:left + len(piece)] += 2 * np.pi * angle * quadrature_weight * intensity
        return total

    for order in output["orders"]:
        grid = output[f"grid_o{order}"]
        wavelength = 1e8 / grid
        if wavelength.min() < 1500 or wavelength.max() > 20000:
            raise ValueError("Do not extrapolate the public model to the K band.")
        logwave = np.log10(wavelength)
        integrated = flux(logwave, 8)
        normalized = integrated[:, 0] / integrated[:, 1]
        if np.any(normalized <= 0):
            raise ValueError("Nonpositive normalized stellar flux.")
        output[f"flux_o{order}"] = normalized
        output[f"continuum_o{order}"] = integrated[:, 1]
        # Higher angular order on a fixed sparse subset bounds integration sensitivity.
        subset = np.arange(0, len(grid), 20)
        finer = flux(logwave[subset], 12)
        difference = finer[:, 0] / finer[:, 1] - normalized[subset]
        record = dict(order=int(order), n_wavelengths=len(grid),
                      vacuum_wavelength_range_angstrom=[float(wavelength.min()), float(wavelength.max())],
                      normalized_flux_range=[float(normalized.min()), float(normalized.max())],
                      quadrature_8_vs_12_max_absolute=float(np.max(np.abs(difference))),
                      quadrature_8_vs_12_rms=float(np.sqrt(np.mean(difference ** 2))))
        records.append(record)
        print(json.dumps(record), "elapsed_s", time.monotonic() - start, flush=True)
    target = CACHE / "stellar_transformerpayne.npz"
    np.savez_compressed(target, **output)
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                  parameters=dict(teff_K=5600., logg=4.5, vmic_kms=1., abundances="checkpoint solar defaults"),
                  parameter_names=emulator.stellar_parameter_names,
                  parameter_values=np.asarray(parameters).tolist(), checkpoint=checkpoint,
                  wavelength_convention="log10(vacuum Angstrom) input; ascending cm^-1 output grids",
                  normalization="2*pi*integral(mu*I_lambda dmu) / [2*pi*integral(mu*I_continuum dmu)]",
                  angular_quadrature=8, batch_size=batch_size,
                  broadening="None; comparison applies the same stellar rotation, extra width, and IP as PHOENIX",
                  unsupported_orders={"13": "Near 2.07 micron, outside advertised 0.15-2.0 micron support; retain PHOENIX"},
                  limitations=["Experimental MARCS/Korg/Kurucz emulator; not a validated NIR stellar retrieval",
                               "Neural layers use float32 internally after float64 wavelength encoding",
                               "Disk-integrated flux followed by the same simple rotation kernel as the PHOENIX control"],
                  versions={name: version(name) for name in ["transformer-payne", "jax", "jaxlib", "flax", "numpy"]},
                  devices=[str(device) for device in jax.devices()], per_order=records,
                  elapsed_seconds=time.monotonic() - start,
                  source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__), CACHE / "transformerpayne_input_grids.npz", CACHE / "transformerpayne_checkpoint.json"]},
                  product_sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    target.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print("Saved", target, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--grids", action="store_true")
    args = parser.parse_args()
    if args.download:
        download()
    elif args.grids:
        grids()
    else:
        prepare()
