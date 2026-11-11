"""Prepare flexible-profile tables and compare interpolation with exact ExoJAX."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import cross_section
from hitran_cache import load_mdb
from profile_opacity import (prepare_profile_opacity, pressure_node_profile,
                             redistribution_weights)

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[13, 34, 40])
    parser.add_argument("--species", nargs="+", default=["H2O", "CO2"])
    parser.add_argument("--step", type=float, default=2.)
    parser.add_argument("--tag", default="profile_opacity_checks")
    args = parser.parse_args()
    profile = load_profile(20)
    p = profile["pressure_bar"]
    node_p = np.array([.01, .1, .35, .7])
    cases = {
        "zero": np.zeros(20),
        "offgrid_uniform": np.full(20, 1.3),
        "offgrid_structure": np.asarray(pressure_node_profile(p, node_p, [0., -13.7, 9.1, -7.3])),
        "cool_limit": np.full(20, -19.7),
        "warm_limit": np.full(20, 19.7),
    }
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wave, _ = pickle.load(stream)
    result = {"orders": args.orders, "species": args.species,
              "temperature_step_K": args.step, "checks": [],
              "validation": "Exact ExoJAX at five temperature profiles and a normalized column redistribution; 20 pressure layers, airmass1.8155, Gaussian IP sigma2.3 km/s",
              "profile_module_sha256": hashlib.sha256((ROOT / "profile_opacity.py").read_bytes()).hexdigest()}
    started = time.monotonic()
    for order in args.orders:
        nu = 1e4 / wave[order, 600:1500][::-1]
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/.75))+1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = c*np.log(grid[1]/grid[0])
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        interior = (grid >= nu[0]) & (grid <= nu[-1])
        for species in args.species:
            db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
            opa = OpaDirect(db, grid)
            table = prepare_profile_opacity(opa, grid, profile, species, cross_section,
                                             shifted=True, temperature_step_K=args.step)
            shape = pressure_node_profile(p, [.1, .35, .7], [0., .4, -.3])
            weights = redistribution_weights(table.reference_column, shape)
            column_error = abs(float(jnp.sum(table.reference_column*weights)/jnp.sum(table.reference_column))-1)
            assert column_error < 1e-12
            amplitude = .7 if species == "H2O" else 1.
            smooth = jax.jit(lambda tau: ipgauss(jnp.exp(-1.8155*amplitude*tau), velocity, 2.3))
            item = dict(order=order, species=species, table=table.metadata,
                        redistribution_relative_column_error=column_error, cases={})
            for label, delta in cases.items():
                # Applying a representative redistribution to all species also
                # exercises upper layers that contribute little at reference humidity.
                actual_weights = np.ones(20) if label == "zero" else weights
                approximate = np.asarray(table.tau(delta, actual_weights))
                exact = table.direct_tau(delta, actual_weights)
                difference = np.asarray(smooth(approximate)-smooth(exact))[interior]
                rms = float(np.sqrt(np.mean(difference**2)))
                maximum_error = float(np.max(np.abs(difference)))
                item["cases"][label] = dict(rms_flux_difference=rms,
                                             maximum_flux_difference=maximum_error,
                                             relative_tau_l2_error=float(np.linalg.norm(approximate-exact)/np.linalg.norm(exact)))
                assert rms < 5e-4 and maximum_error < 1e-3
            direction = jnp.asarray(pressure_node_profile(p, node_p, [.2, -1., .7, .5]))
            delta = jnp.asarray(cases["offgrid_structure"])
            approximate_fun = jax.jit(lambda dt: smooth(table.tau(dt, weights)))
            derivative = np.asarray(jax.jvp(approximate_fun, (delta,), (direction,))[1])[interior]
            exact_plus = np.asarray(smooth(table.direct_tau(delta+.01*direction, weights)))
            exact_minus = np.asarray(smooth(table.direct_tau(delta-.01*direction, weights)))
            exact_derivative = (exact_plus-exact_minus)[interior]/.02
            item["derivative_vs_exact"] = dict(
                relative_l2_error=float(np.linalg.norm(derivative-exact_derivative)/np.linalg.norm(exact_derivative)),
                maximum_absolute_error_per_K=float(np.max(np.abs(derivative-exact_derivative))))
            # Verify both away from interpolation nodes and at the initial zero offset.
            for label in ["zero", "offgrid_structure"]:
                delta = jnp.asarray(cases[label])
                value = np.asarray(jax.jvp(approximate_fun, (delta,), (direction,))[1])[interior]
                finite = np.asarray((approximate_fun(delta+1e-4*direction)
                                     - approximate_fun(delta-1e-4*direction))/2e-4)[interior]
                error = float(np.linalg.norm(value-finite)/max(np.linalg.norm(value), 1e-30))
                item[f"autodiff_finite_difference_error_{label}"] = error
                assert error < 1e-5
            result["checks"].append(item)
            print(json.dumps({"order": order, "species": species, "cases": item["cases"],
                              "derivative_vs_exact": item["derivative_vs_exact"]}), flush=True)
            result["seconds"] = time.monotonic()-started
            (ROOT / "results" / f"{args.tag}.json").write_text(json.dumps(result, indent=2)+"\n")
    print("All profile opacity checks passed.", flush=True)


if __name__ == "__main__":
    main()
