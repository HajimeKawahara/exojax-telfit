"""Read HITRAN intensity uncertainty bins and cache individual line opacity.

The quoted finite upper bin edges are scales for a sensitivity model, not
independent Gaussian standard deviations supplied by HITRAN.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity.lpf.lpf import vvoigt, xsvector
from exojax.utils.constants import Patm, c

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
FINITE_UPPER = {4: .20, 5: .10, 6: .05, 7: .02, 8: .01, 9: .001}
UNCERTAINTY_LABELS = {0: "unreported/unavailable", 1: "default/constant",
                      2: "average/estimate", 3: ">=20%", 4: "10-20%",
                      5: "5-10%", 6: "2-5%", 7: "1-2%", 8: "0.1-1%",
                      9: "<0.1%"}


def native_grid(wavelength_um, dv_kms=.75, padding_kms=220.):
    """Reproduce the wavenumber grid in the current joint fitting code."""
    nu = 1e4 / np.asarray(wavelength_um)
    if not np.all(np.diff(nu) > 0):
        raise ValueError("Expected the ascending-wavenumber order used by fit_joint.")
    minimum = nu[0] * np.exp(-padding_kms/c)
    maximum = nu[-1] * np.exp(padding_kms/c)
    n = int(np.ceil(np.log(maximum/minimum)*c/dv_kms))+1
    n += n % 2
    return np.geomspace(minimum, maximum, n)


@lru_cache(maxsize=None)
def parse_lines(species):
    """Return line metadata preserving the raw-file index and all error codes."""
    path = CACHE / f"{species}.par"
    rows = path.read_text().splitlines()
    result = []
    for index, row in enumerate(rows):
        if len(row) != 160:
            raise ValueError(f"Invalid 160-column HITRAN row {index} in {path}")
        isotope = (int(row[2]) if row[2].isdigit() and row[2] != "0"
                   else 10 if row[2] == "0" else ord(row[2])-ord("A")+11)
        code = int(row[128])
        result.append(dict(species=species, raw_index=index, molecule=int(row[:2]),
                           iso=isotope, nu=float(row[3:15]), sw=float(row[15:25]),
                           ierr=row[127:133], iref=row[133:145], strength_code=code,
                           strength_reference=row[135:137],
                           uncertainty_label=UNCERTAINTY_LABELS[code],
                           uncertainty_upper=FINITE_UPPER.get(code)))
    with np.load(CACHE / f"{species}.npz") as saved:
        np.testing.assert_array_equal(saved["nu"], [record["nu"] for record in result])
        np.testing.assert_array_equal(saved["sw"], [record["sw"] for record in result])
        np.testing.assert_array_equal(saved["iso"], [record["iso"] for record in result])
    return result


def select_lines(species, wavelength_um, included_mask, strength_min=1e-25,
                 velocity_margin_kms=10.):
    """Select relevant finite-uncertainty lines without looking at residuals."""
    nu = 1e4/np.asarray(wavelength_um)
    allowed = nu[np.asarray(included_mask, dtype=bool)]
    if allowed.size == 0:
        raise ValueError("No included wavelength pixels.")
    candidates, selected = [], []
    for source in parse_lines(species):
        if source["sw"] < strength_min:
            continue
        if not nu.min()*np.exp(-velocity_margin_kms/c) <= source["nu"] <= nu.max()*np.exp(velocity_margin_kms/c):
            continue
        separation = float(np.min(np.abs(c*np.log(allowed/source["nu"]))))
        if separation > velocity_margin_kms:
            continue
        record = dict(source, nearest_included_velocity_kms=separation)
        candidates.append(record)
        if source["uncertainty_upper"] is not None:
            selected.append(record)
    codes, counts = np.unique([r["strength_code"] for r in candidates], return_counts=True)
    metadata = dict(species=species, strength_min=float(strength_min),
                    velocity_margin_kms=float(velocity_margin_kms),
                    observed_nu_range=[float(nu.min()), float(nu.max())],
                    candidate_count=len(candidates), selected_count=len(selected),
                    candidate_uncertainty_counts=dict(zip(map(str, codes), map(int, counts))),
                    excluded_unbounded_uncertainty_count=len(candidates)-len(selected),
                    selection_uses="line metadata and the fixed included mask only",
                    uncertainty_scale_note="Finite bin upper edges; not measured Gaussian standard deviations")
    return selected, metadata


def prepare_line_tau(species, records, grid, profile, shifted=True,
                     self_broadening=False):
    """Return individual vertical optical depths on a supplied wavenumber grid."""
    grid = np.asarray(grid, dtype=float)
    indices = np.asarray([record["raw_index"] for record in records], dtype=int)
    if len(indices) == 0:
        return np.zeros((0, len(grid))), dict(species=species, line_count=0)
    raw_sha = hashlib.sha256((CACHE / f"{species}.par").read_bytes()).hexdigest()
    pressure = np.asarray(profile["pressure_bar"])
    temperature = np.asarray(profile["temperature_K"])
    vmr = np.asarray(profile["vmr"][species])
    columns = np.asarray(profile["number_column_cm2"])*vmr
    self_pressure = pressure*vmr if self_broadening else np.zeros_like(pressure)
    with np.load(CACHE / f"{species}.npz") as saved:
        arrays = {key: np.asarray(saved[key][indices]) for key in
                  ["nu", "sw", "a", "gamma_air", "gamma_self", "n_air", "elower", "delta_air", "line_mass", "iso"]}
        temperatures = np.asarray(saved["temperatures"])
        partition_isotopes = np.asarray(saved["partition_isotopes"])
        partition_sums = np.asarray(saved["partition_sums"])
    np.testing.assert_array_equal(arrays["nu"], [record["nu"] for record in records])
    np.testing.assert_array_equal(arrays["sw"], [record["sw"] for record in records])
    digest = hashlib.sha256(raw_sha.encode())
    for array in [indices, grid, pressure, temperature, columns, self_pressure,
                  temperatures, partition_sums]:
        digest.update(array.tobytes())
    for key in sorted(arrays):
        digest.update(key.encode()); digest.update(arrays[key].tobytes())
    digest.update(str(shifted).encode())
    digest.update(hashlib.sha256(Path(__file__).read_bytes()).digest())
    path = CACHE / f"line_strength_{species}_{digest.hexdigest()[:20]}.npz"
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        with np.load(path) as saved:
            np.testing.assert_array_equal(saved["grid"], grid)
            np.testing.assert_array_equal(saved["raw_indices"], indices)
            tau = np.asarray(saved["tau"])
        return tau, json.loads(metadata_path.read_text())
    a = {key: jnp.asarray(value) for key, value in arrays.items()}
    isotope_index = jnp.asarray(np.searchsorted(partition_isotopes, arrays["iso"]))
    t_grid = jnp.asarray(temperatures)
    q_table = jnp.asarray(partition_sums)
    offset = jnp.asarray(grid)[None, :]-a["nu"][:, None]

    @jax.jit
    def layer_values(t, p, ps):
        qr = jax.vmap(lambda q: jnp.interp(t, t_grid, q)/jnp.interp(296., t_grid, q))(q_table)[isotope_index]
        strength = line_strength(t, jnp.log(a["sw"]), a["nu"], a["elower"], qr, 296.)
        width = gamma_hitran(p, t, ps, a["n_air"], a["gamma_air"], a["gamma_self"])+gamma_natural(a["a"])
        sigma = doppler_sigma(a["nu"], t, a["line_mass"])
        delta = a["delta_air"]*(p-ps)/Patm if shifted else jnp.zeros_like(a["nu"])
        shifted_offset = offset-delta[:, None]
        per_line = vvoigt(shifted_offset, sigma, width)*strength[:, None]
        total = xsvector(shifted_offset, sigma, width, strength)
        return per_line, total

    tau = np.zeros((len(indices), len(grid)))
    summed_reference = np.zeros(len(grid))
    for t, p, ps, column in zip(temperature, pressure, self_pressure, columns):
        per_line, total = layer_values(t, p, ps)
        tau += column*np.asarray(per_line)
        summed_reference += column*np.asarray(total)
    error = float(np.max(np.abs(tau.sum(axis=0)-summed_reference)))
    relative = error/max(float(np.max(summed_reference)), 1e-300)
    np.testing.assert_allclose(tau.sum(axis=0), summed_reference, rtol=1e-12, atol=1e-14)
    if np.any(~np.isfinite(tau)) or np.any(tau < 0):
        raise ValueError("Invalid per-line optical depth.")
    metadata = dict(species=species, line_count=len(indices), grid_count=len(grid),
                    path=str(path.relative_to(ROOT)), raw_sha256=raw_sha,
                    shifted=bool(shifted), self_broadening=bool(self_broadening),
                    layers=len(pressure), sum_vs_xsvector_max_absolute_error=error,
                    sum_vs_xsvector_relative_linf=relative,
                    identity="tau_total(delta)=tau_baseline+sum(delta_i*tau_i); S_i=S_i,reference*(1+delta_i)")
    np.savez_compressed(path, tau=tau, grid=grid, raw_indices=indices,
                        nu=arrays["nu"], reference_sum=summed_reference)
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    return tau, metadata
