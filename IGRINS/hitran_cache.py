"""Adapt the local HITRAN cache to the existing ExoJAX opacity interface."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np


CACHE = Path(__file__).resolve().parent / "cache"


def load_mdb(name: str, numin: float, numax: float,
             crit: float = 1e-28) -> SimpleNamespace:
    """Select lines by their natural-abundance 296 K reference strength."""
    with np.load(CACHE / f"{name}.npz") as data:
        mask = ((data["nu"] >= numin) & (data["nu"] <= numax)
                & (data["sw"] >= crit))
        nu = data["nu"][mask]
        isotope_index = jnp.asarray(np.searchsorted(data["partition_isotopes"],
                                                   data["iso"][mask]))
        temperatures = jnp.asarray(data["temperatures"])
        partition_sums = jnp.asarray(data["partition_sums"])
        names = {"a": "A", "gamma_air": "gamma_air", "gamma_self": "gamma_self",
                 "n_air": "n_air", "elower": "elower", "delta_air": "delta_air",
                 "line_mass": "molmass", "sw": "Sij0", "iso": "isoid"}
        attributes = {attribute: jnp.asarray(data[key][mask])
                      for key, attribute in names.items()}

    def qr_interp_lines(temperature, reference=296.0):
        # One interpolation per isotope, then gather by line.
        from jax import vmap

        ratio = vmap(lambda q: jnp.interp(temperature, temperatures, q)
                     / jnp.interp(reference, temperatures, q))(partition_sums)
        return ratio[isotope_index]

    return SimpleNamespace(
        dbtype="hitran", gpu_transfer=True, nu_lines=nu,
        dev_nu_lines=jnp.asarray(nu), logsij0=jnp.log(attributes["Sij0"]),
        Tref=296.0, qr_interp_lines=qr_interp_lines,
        simple_molecule_name=name, **attributes,
    )
