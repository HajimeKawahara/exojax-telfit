"""Download and normalize one PHOENIX HiRes template for WASP-77A."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import numpy as np
from astropy.io import fits
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import gaussian_filter1d

CACHE = Path(__file__).resolve().parent / "cache"
BASE = "https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/"
FILES = {
    "stellar_wave.fits": "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits",
    "stellar_flux.fits": "PHOENIX-ACES-AGSS-COND-2011/Z-0.0/"
                         "lte05600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits",
}


def prepare():
    """Retain raw FITS and native samples; only estimate a smooth continuum."""
    CACHE.mkdir(exist_ok=True)
    provenance = {}
    for filename, suffix in FILES.items():
        path = CACHE / filename
        url = BASE + suffix
        if not path.exists():
            temporary = path.with_suffix(".part")
            with urlopen(url, timeout=120) as response, temporary.open("wb") as stream:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    stream.write(chunk)
            # Reject an HTML error response before keeping the asset.
            fits.getheader(temporary)
            temporary.replace(path)
        provenance[filename] = {
            "url": url, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_modified_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
        }
        print(f"Ready: {filename} ({path.stat().st_size} bytes)", flush=True)
    wavelength = np.asarray(fits.getdata(CACHE / "stellar_wave.fits"), dtype=float)
    flux, header = fits.getdata(CACHE / "stellar_flux.fits", header=True)
    flux = np.asarray(flux, dtype=float)
    if wavelength.shape != flux.shape or np.any(np.diff(wavelength) <= 0):
        raise ValueError("PHOENIX wavelength and flux arrays do not match.")
    # The HiRes wavelength file is in vacuum Angstroms, not air or log wavelength.
    keep = (wavelength >= 14000.) & (wavelength <= 25000.)
    wavelength, flux = wavelength[keep], flux[keep]
    if np.any(~np.isfinite(flux)) or np.any(flux <= 0):
        raise ValueError("Template contains invalid flux.")
    logwave = np.log(wavelength)
    bin_width = 500. / 299792.458
    edges = np.arange(logwave[0], logwave[-1] + bin_width, bin_width)
    nodes, envelope = [], []
    for left, right in zip(edges[:-1], edges[1:]):
        selected = (logwave >= left) & (logwave < right)
        if selected.any():
            nodes.append(float(np.mean(logwave[selected])))
            envelope.append(float(np.percentile(flux[selected], 99.5)))
    # Smooth the continuum nodes, never the line spectrum itself.
    smooth_log_envelope = gaussian_filter1d(np.log(envelope), 1., mode="nearest")
    continuum = np.exp(PchipInterpolator(nodes, smooth_log_envelope, extrapolate=True)(logwave))
    normalized = flux / continuum
    np.savez_compressed(CACHE / "stellar_phoenix.npz",
                        nu_cm1=(1e8 / wavelength)[::-1],
                        flux_normalized=normalized[::-1],
                        flux_lambda=flux[::-1], continuum_lambda=continuum[::-1])
    metadata = {
        "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "source": "https://phoenix.astro.physik.uni-goettingen.de/spectra/",
        "paper": "https://arxiv.org/abs/1303.5632",
        "files": provenance,
        "teff_K": float(header["PHXTEFF"]), "logg": float(header["PHXLOGG"]),
        "metallicity": float(header.get("PHXM_H", 0.)),
        "phoenix_version": str(header.get("PHXVER", "unknown")),
        "wavelength_convention": "vacuum Angstrom in raw FITS; cm^-1 in ascending cached grid",
        "wavelength_range_um": [float(wavelength[0] / 1e4), float(wavelength[-1] / 1e4)],
        "native_sampling_R": float(1 / np.median(np.diff(logwave))),
        "normalization": "Pseudo-continuum: 99.5th percentile in 500 km/s log-wavelength bins; Gaussian smooth log node flux with sigma=1 bin; PCHIP interpolation. Not a supplied physical continuum.",
        "broadening": "None. Native PHOENIX line samples retained. Apply stellar broadening before telluric multiplication and instrumental broadening after.",
        "normalized_flux_percentiles": np.percentile(normalized, [0, 1, 50, 99, 100]).tolist(),
    }
    (CACHE / "stellar_phoenix.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    prepare()
