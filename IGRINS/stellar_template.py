"""Read the intrinsic PHOENIX stellar template with strict coverage checks."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

CACHE = Path(__file__).resolve().parent / "cache/stellar_phoenix.npz"


@lru_cache(maxsize=1)
def _template():
    with np.load(CACHE) as data:
        return data["nu_cm1"], data["flux_normalized"]


def load_stellar(nu_grid):
    """Return normalized flux without Doppler shift or additional broadening."""
    nu = np.asarray(nu_grid, dtype=float)
    native_nu, flux = _template()
    if nu.ndim != 1 or not nu.size or np.any(~np.isfinite(nu)):
        raise ValueError("Requested wavenumbers must be a finite nonempty vector.")
    if np.min(nu) < native_nu[0] or np.max(nu) > native_nu[-1]:
        raise ValueError("Requested wavelengths exceed the PHOENIX cache coverage.")
    return np.interp(nu, native_nu, flux)
