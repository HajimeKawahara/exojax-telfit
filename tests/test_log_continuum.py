"""Check positive continuum fits and the nonzero-residual template response."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IGRINS"))
from log_continuum import fit_log_continuum, filter_log_continuum_template
from smooth_continuum import cubic_spline_basis


def test_noisy_blaze_retains_saturated_and_negative_samples():
    rng = np.random.default_rng(23)
    basis, _ = cubic_spline_basis(512, spacing=128.)
    pixel = np.arange(len(basis))
    x = np.linspace(-1., 1., len(basis))
    continuum = np.exp(-.2 - .8 * x ** 2 - .3 * x)
    model = 1 - .9999 * np.exp(-.5 * ((pixel - 220) / 6) ** 2)
    model[216:224] = 0.
    observed = continuum * model + rng.normal(0., .0005, len(pixel))
    observed[216:224] = -.002
    observed[224] = -.001
    fit = fit_log_continuum(observed, model, basis, continuum * 1.1)
    assert fit["diagnostics"]["success"]
    assert fit["diagnostics"]["fitted_pixels"] == len(pixel)
    assert fit["diagnostics"]["fitted_nonpositive_observations"] == 9
    assert fit["diagnostics"]["stationary"]
    assert not fit["active_mask"].any()
    np.testing.assert_array_equal(fit["residual"][model == 0], observed[model == 0])
    assert fit["residual"][224] < observed[224]
    np.testing.assert_allclose(fit["continuum"], continuum, atol=.002)


def test_exact_response_matches_refits_with_nonzero_model_residual():
    basis, _ = cubic_spline_basis(400, spacing=80.)
    pixel = np.arange(len(basis))
    model = 1 - .7 * np.exp(-.5 * ((pixel - 120) / 2) ** 2)
    coefficients = np.linspace(-.7, .2, basis.shape[1])
    continuum = np.exp(basis @ coefficients)
    predicted = model * continuum
    jacobian = predicted[:, None] * basis
    # A model discrepancy orthogonal to the tangent leaves a known stationary
    # optimum while giving a nonzero second-order term in the Hessian.
    discrepancy = .12 * np.sin(pixel / 13.)
    discrepancy -= jacobian @ np.linalg.lstsq(jacobian, discrepancy, rcond=None)[0]
    observed = predicted + discrepancy
    fit = fit_log_continuum(observed, model, basis, continuum)
    template = -np.exp(-.5 * ((pixel - 108) / 1.4) ** 2)
    epsilon = 1e-4
    plus = fit_log_continuum(observed + epsilon * template, model, basis, fit["continuum"])
    minus = fit_log_continuum(observed - epsilon * template, model, basis, fit["continuum"])
    expected = (plus["residual"] - minus["residual"]) / (2 * epsilon)
    actual = filter_log_continuum_template(template, observed, model, basis, fit["continuum"])
    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=2e-6)
    gauss_newton = template - jacobian @ np.linalg.lstsq(jacobian, template, rcond=None)[0]
    assert np.max(np.abs(gauss_newton - actual)) > 1e-4
    batch = filter_log_continuum_template(np.stack([template, 2 * template]), observed,
                                          model, basis, fit["continuum"])
    np.testing.assert_allclose(batch, np.stack([actual, 2 * actual]), atol=1e-14)


def test_rejects_nonidentifiable_and_nonminimum_response():
    basis, _ = cubic_spline_basis(100, spacing=32.)
    ones = np.ones(len(basis))
    with pytest.raises(ValueError, match="rank deficient"):
        fit_log_continuum(ones, np.zeros(len(basis)), basis, ones)
    with pytest.raises(ValueError, match="strictly positive"):
        fit_log_continuum(ones, ones, basis, np.zeros(len(basis)))
    with pytest.raises(ValueError, match="positive definite"):
        filter_log_continuum_template(ones, 3 * ones, ones, basis, ones)
    active = np.ones(basis.shape[1], dtype=bool)
    np.testing.assert_array_equal(
        filter_log_continuum_template(ones, ones, ones, basis, ones, active), ones)
