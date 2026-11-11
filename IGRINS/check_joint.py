"""Check physical invariants of the joint stellar/telluric implementation."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np

from atmosphere_profile import load_profile
from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.utils.constants import c
from fit_joint import cross_section, profile_tau
from hitran_cache import load_mdb
from stellar_template import load_stellar


def main():
    root = Path(__file__).resolve().parent
    checks = {}
    for nlayer in [10, 20, 40]:
        p = load_profile(nlayer)
        boundaries = p["pressure_boundaries_bar"]
        np.testing.assert_allclose(np.sum(p["delta_pressure_bar"]),
                                   boundaries[-1] - boundaries[0], rtol=1e-14)
        assert np.all(p["pressure_bar"] > boundaries[:-1])
        assert np.all(p["pressure_bar"] < boundaries[1:])
        assert np.all(p["number_column_cm2"] > 0)
        assert np.all((p["temperature_K"] > 150) & (p["temperature_K"] < 400))
        checks[f"profile_{nlayer}_layers"] = {
            "reference_pwv_mm": p["metadata"]["reference_pwv_mm"],
            "reference_CO2_column_cm2": p["metadata"]["reference_vertical_columns_cm2"]["CO2"],
        }
    # A compact real-data band avoids constructing an unnecessary full spectrum.
    grid = np.geomspace(6350., 6370., 2048)
    velocity = jnp.arange(-80, 81) * float(c * np.log(grid[1] / grid[0]))
    interior = slice(300, -300)
    star = load_stellar(grid)
    assert np.all(np.isfinite(star)) and np.all(star > 0)
    db = load_mdb("CO2", grid[0] - 3, grid[-1] + 3, crit=1e-27)
    opa = OpaDirect(db, grid)
    profile = load_profile(20)
    tau, column = profile_tau(opa, grid, profile, "CO2")
    expected_column = np.sum(profile["number_column_cm2"] * profile["vmr"]["CO2"])
    np.testing.assert_allclose(column, expected_column, rtol=1e-12)
    transmission = np.exp(-tau)
    telluric_only = np.asarray(ipgauss(transmission, velocity, 2.5))
    stellar_only = np.asarray(ipgauss(star, velocity, 2.5))
    # Demonstrate the measurable effect of multiplication before instrumental IP.
    correct = np.asarray(ipgauss(star * transmission, velocity, 2.5))
    separate = stellar_only * telluric_only
    checks["product_before_IP_vs_separate"] = {
        "rms_difference_continuum": float(np.sqrt(np.mean((correct[interior] - separate[interior])**2))),
        "maximum_difference_continuum": float(np.max(np.abs(correct[interior] - separate[interior]))),
    }
    # Positive observer-frame RV must move a line toward lower wavenumber.
    center = float(grid[len(grid)//2])
    test_line = 1 - .5 * np.exp(-.5 * ((grid-center)/.02)**2)
    shifted = np.asarray(sampling(grid, grid, test_line, 25.))
    measured = float(grid[np.argmin(shifted)])
    expected = center/(1 + 25./c)
    assert abs(measured-expected) <= np.max(np.diff(grid))
    checks["positive_rv_shift"] = {"expected_center_cm1": expected, "measured_center_cm1": measured}
    # Existing per-line pressure-shift branch is differentiated at nonzero self pressure.
    def signal(pressure):
        xs = cross_section(opa, 265., pressure, .001, True)
        return jnp.sum(xs) * 1e22
    value = .5
    step = 1e-5
    exact = float(jax.jit(jax.grad(signal))(value))
    finite = float((signal(value+step)-signal(value-step))/(2*step))
    np.testing.assert_allclose(exact, finite, rtol=1e-3, atol=1e-8)
    checks["pressure_shift_derivative"] = {
        "autodiff": exact, "finite_difference": finite,
        "relative_error": abs(exact-finite)/max(abs(exact), 1e-30),
    }
    checks["status"] = "All physical invariant checks passed"
    (root/"results/joint_checks.json").write_text(json.dumps(checks, indent=2)+"\n")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
