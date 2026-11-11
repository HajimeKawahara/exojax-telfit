"""Differentiable fixed-pressure opacity tables for flexible terrestrial profiles.

Temperature changes affect exact ExoJAX cross sections tabulated per layer.
Species layer-column weights can redistribute a fixed total molecular column.
Pressure boundaries and the reference background mean molecular mass stay fixed;
the latter is an explicit trace-gas approximation when redistributing water.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import os
from pathlib import Path
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

CACHE = Path(__file__).resolve().parent / "cache"
TABLE_VERSION = "fixed-pressure-logtau-hermite-v1"


def pressure_node_profile(pressure, node_pressures, node_values):
    """Interpolate node values linearly in log pressure, constant outside nodes."""
    return jnp.interp(jnp.log(jnp.asarray(pressure)),
                      jnp.log(jnp.asarray(node_pressures)), jnp.asarray(node_values))


def redistribution_weights(reference_column, log10_shape):
    """Return positive layer multipliers preserving the total molecular column."""
    column = jnp.asarray(reference_column)
    shape = jnp.broadcast_to(jnp.asarray(log10_shape), column.shape)
    weights = 10. ** (shape - jnp.max(shape))
    return weights * jnp.sum(column) / jnp.sum(column * weights)


@dataclass
class ProfileOpacity:
    """Temperature-interpolated optical depths and an exact reference evaluator."""

    offsets_K: jax.Array
    log_layer_tau: jax.Array
    log_tau_slopes: jax.Array
    reference_column: jax.Array
    pressure_bar: jax.Array
    reference_temperature_K: jax.Array
    metadata: dict
    _xs: Callable
    _self_pressure: np.ndarray

    def layer_tau(self, delta_temperature_K):
        """Return per-layer optical depth; out-of-table temperatures produce NaN."""
        delta = jnp.broadcast_to(jnp.asarray(delta_temperature_K), self.pressure_bar.shape)
        step = self.offsets_K[1] - self.offsets_K[0]
        index = jnp.clip(jnp.floor((delta-self.offsets_K[0])/step).astype(int),
                         0, len(self.offsets_K)-2)
        fraction = (delta-self.offsets_K[index])/step
        row = jnp.arange(len(delta))
        y0 = self.log_layer_tau[row, index]
        y1 = self.log_layer_tau[row, index+1]
        m0 = self.log_tau_slopes[row, index]
        m1 = self.log_tau_slopes[row, index+1]
        u = fraction[:, None]
        value = ((2*u**3-3*u**2+1)*y0 + (u**3-2*u**2+u)*step*m0
                 + (-2*u**3+3*u**2)*y1 + (u**3-u**2)*step*m1)
        valid = (delta >= self.offsets_K[0]) & (delta <= self.offsets_K[-1])
        return jnp.where(valid[:, None], jnp.exp(value), jnp.nan)

    def tau(self, delta_temperature_K, layer_weights=None):
        """Sum layer optical depths after optional molecular-column redistribution."""
        layers = self.layer_tau(delta_temperature_K)
        if layer_weights is not None:
            layers = layers * jnp.asarray(layer_weights)[:, None]
        return jnp.sum(layers, axis=0)

    def direct_tau(self, delta_temperature_K, layer_weights=None):
        """Evaluate exact cross sections without temperature interpolation, on CPU."""
        delta = np.broadcast_to(np.asarray(delta_temperature_K), self.pressure_bar.shape)
        temperature = np.asarray(self.reference_temperature_K) + delta
        if np.any((temperature < 150) | (temperature > 400)):
            raise ValueError("Temperatures must lie within cached partition sums (150-400 K).")
        weights = np.ones(len(temperature)) if layer_weights is None else np.asarray(layer_weights)
        if weights.shape != temperature.shape or np.any(weights < 0):
            raise ValueError("Layer weights must be nonnegative and match the pressure grid.")
        result = np.zeros(self.log_layer_tau.shape[-1])
        for t, p, ps, column, weight in zip(temperature, np.asarray(self.pressure_bar),
                                           self._self_pressure, np.asarray(self.reference_column), weights):
            result += column * weight * np.asarray(self._xs(t, p, ps))
        return result


def prepare_profile_opacity(opa, grid, profile, species, cross_section,
                            shifted=False, self_broadening=False,
                            temperature_range_K=30., temperature_step_K=2.):
    """Build or load per-layer temperature tables using the supplied exact opacity."""
    started = time.monotonic()
    temperature = np.asarray(profile["temperature_K"], dtype=float)
    pressure = np.asarray(profile["pressure_bar"], dtype=float)
    vmr = np.asarray(profile["vmr"][species], dtype=float)
    column = np.asarray(profile["number_column_cm2"], dtype=float) * vmr
    if temperature_range_K <= 0 or temperature_step_K <= 0:
        raise ValueError("Temperature range and step must be positive.")
    nstep = int(round(2*temperature_range_K/temperature_step_K))
    if nstep < 2 or not np.isclose(nstep*temperature_step_K, 2*temperature_range_K):
        raise ValueError("Temperature step must divide the full table range.")
    offsets = np.linspace(-temperature_range_K, temperature_range_K, nstep+1)
    if np.min(temperature)+offsets[0] < 150 or np.max(temperature)+offsets[-1] > 400:
        raise ValueError("Temperature table extends outside cached partition sums.")
    if np.any(column <= 0):
        raise ValueError("All reference molecular layer columns must be positive.")
    self_pressure = pressure * vmr if self_broadening else np.zeros_like(pressure)
    source = inspect.getsource(cross_section)
    digest = hashlib.sha256((TABLE_VERSION + source + species + str(shifted)).encode())
    for value in (grid, pressure, temperature, column, self_pressure, offsets,
                  opa.mdb.nu_lines, opa.mdb.logsij0, opa.mdb.A, opa.mdb.elower,
                  opa.mdb.gamma_air, opa.mdb.gamma_self, opa.mdb.n_air,
                  opa.mdb.delta_air, opa.mdb.molmass):
        digest.update(np.asarray(value).tobytes())
    # Include the per-isotope partition sums captured by the local cache adapter.
    for cell in opa.mdb.qr_interp_lines.__closure__ or ():
        value = np.asarray(cell.cell_contents)
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    cache_key = digest.hexdigest()
    path = CACHE / f"profile_{species}_{cache_key[:20]}.npz"
    xs = jax.jit(lambda t, p, ps: cross_section(opa, t, p, ps, shifted))
    if path.exists():
        with np.load(path) as saved:
            log_tau = saved["log_layer_tau"]
        print(f"Profile opacity {species}: loaded {path.name}", flush=True)
    else:
        log_tau = np.empty((len(pressure), len(offsets), len(grid)), dtype=np.float64)
        for layer, (t, p, ps, n) in enumerate(zip(temperature, pressure, self_pressure, column)):
            for k, dt in enumerate(offsets):
                value = n * np.asarray(xs(t+dt, p, ps))
                if not np.all(np.isfinite(value)) or np.any(value < 0):
                    raise ValueError("Invalid exact layer optical depth.")
                log_tau[layer, k] = np.log(np.maximum(value, 1e-300))
            if layer % 5 == 0 or layer == len(pressure)-1:
                print(f"Profile opacity {species}: layer {layer+1}/{len(pressure)}, "
                      f"{time.monotonic()-started:.1f} s", flush=True)
        CACHE.mkdir(exist_ok=True)
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, log_layer_tau=log_tau, offsets_K=offsets,
                                pressure_bar=pressure, reference_temperature_K=temperature,
                                reference_column=column)
        temporary.replace(path)
    slopes = np.gradient(log_tau, temperature_step_K, axis=1, edge_order=2)
    metadata = dict(species=species, cache_path=str(path), cache_key=cache_key,
                    table_version=TABLE_VERSION, pressure_shift=bool(shifted),
                    reference_self_broadening=bool(self_broadening),
                    temperature_offset_range_K=offsets[[0, -1]].tolist(),
                    temperature_step_K=float(temperature_step_K),
                    interpolation="Cubic Hermite of log(layer optical depth), finite-difference nodal slopes",
                    layer_count=len(pressure), grid_count=len(grid),
                    reference_total_column_cm2=float(column.sum()),
                    approximation="Fixed pressure boundaries and background mean molecular mass; reference self widths if enabled",
                    preparation_seconds=time.monotonic()-started)
    return ProfileOpacity(jnp.asarray(offsets), jnp.asarray(log_tau), jnp.asarray(slopes),
                          jnp.asarray(column), jnp.asarray(pressure), jnp.asarray(temperature),
                          metadata, xs, self_pressure)
