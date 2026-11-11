"""Evaluate supported TransformerPayne stellar labels and their exact derivatives."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np
from transformer_payne import TransformerPayne

ROOT = Path(__file__).resolve().parent


class StellarEmulator:
    """Integrate intensity and continuum before normalizing the disk spectrum."""

    lower = np.array([4000., 2., -2.5, 0.])
    upper = np.array([8000., 5., 1., 5.])
    label_names = ["teff_K", "logg", "metallicity_dex", "vmic_kms"]

    def __init__(self, grids: dict[int, np.ndarray]):
        checkpoint = json.loads((ROOT / "cache/transformerpayne_checkpoint.json").read_text())
        path = Path(checkpoint["path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != checkpoint["sha256"]:
            raise ValueError("Checkpoint checksum mismatch.")
        self.emulator = TransformerPayne.from_file(str(path))
        self.grids = {int(order): np.asarray(grid) for order, grid in grids.items()}
        self.logwave = {}
        for order, grid in self.grids.items():
            wavelength = 1e8 / grid
            if wavelength.min() < 1500 or wavelength.max() > 20000:
                raise ValueError("Do not extrapolate outside the public wavelength range.")
            self.logwave[order] = np.log10(wavelength)
        self.solar = np.asarray(self.emulator.solar_parameters)
        self.names = self.emulator.stellar_parameter_names
        metals = set(self.emulator.metals)
        self.metal_mask = jnp.asarray([name in metals for name in self.names], dtype=float)
        self.indices = [self.names.index(name) for name in ["logteff", "logg", "vmic"]]
        self.mu, self.weights = np.polynomial.legendre.leggauss(8)
        self.mu, self.weights = (self.mu + 1) / 2, self.weights / 2
        self.batch_size = 256
        self._last_labels = None
        self._last_result = None

        def intensity(labels, wavelength, mu):
            return self.emulator.intensity(wavelength, mu, self.parameters(labels))

        derivative = jax.jacfwd(intensity, argnums=0)
        self.intensity = jax.jit(intensity)
        self.intensity_with_jacobian = jax.jit(
            lambda labels, wavelength, mu: (intensity(labels, wavelength, mu),
                                           derivative(labels, wavelength, mu)))

    def parameters(self, labels):
        """Shift Li through U together; retain the checkpoint helium abundance."""
        p = jnp.asarray(self.solar) + labels[2] * self.metal_mask
        return (p.at[self.indices[0]].set(jnp.log10(labels[0]))
                .at[self.indices[1]].set(labels[1]).at[self.indices[2]].set(labels[3]))

    def evaluate(self, labels, with_jacobian=True):
        """Return flux and optional (wavelength, four-label) Jacobian by order."""
        labels = np.asarray(labels, dtype=float)
        if labels.shape != (4,) or np.any(labels < self.lower) or np.any(labels > self.upper):
            raise ValueError("Labels exceed the advertised model domain.")
        parameters = np.asarray(self.parameters(jnp.asarray(labels)))
        if (np.any(parameters < np.asarray(self.emulator.min_stellar_parameters))
                or np.any(parameters > np.asarray(self.emulator.max_stellar_parameters))):
            raise ValueError("An elemental abundance exceeds the checkpoint bounds.")
        if (self._last_labels is not None and np.array_equal(labels, self._last_labels)
                and (not with_jacobian or self._last_result[1] is not None)):
            return self._last_result
        fluxes, jacobians = {}, {} if with_jacobian else None
        for order, logwave in self.logwave.items():
            total = np.zeros((len(logwave), 2))
            gradient = np.zeros((len(logwave), 2, 4)) if with_jacobian else None
            for mu, weight in zip(self.mu, self.weights):
                for left in range(0, len(logwave), self.batch_size):
                    piece = logwave[left:left + self.batch_size]
                    padded = np.pad(piece, (0, self.batch_size - len(piece)), mode="edge")
                    if with_jacobian:
                        value, jac = self.intensity_with_jacobian(labels, padded, mu)
                        gradient[left:left + len(piece)] += (
                            2 * np.pi * mu * weight * np.asarray(jac)[:len(piece)])
                    else:
                        value = self.intensity(labels, padded, mu)
                    total[left:left + len(piece)] += (
                        2 * np.pi * mu * weight * np.asarray(value)[:len(piece)])
            if np.any(~np.isfinite(total)) or np.any(total <= 0):
                raise ValueError("Invalid integrated intensity or continuum.")
            fluxes[order] = total[:, 0] / total[:, 1]
            if with_jacobian:
                jacobians[order] = (gradient[:, 0] - fluxes[order][:, None] * gradient[:, 1]) / total[:, 1, None]
                if np.any(~np.isfinite(jacobians[order])):
                    raise ValueError("Nonfinite stellar derivative.")
        self._last_labels = labels.copy()
        self._last_result = (fluxes, jacobians)
        return self._last_result
