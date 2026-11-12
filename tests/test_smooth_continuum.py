"""Check continuum fits without division by deep absorption or data clipping."""

from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IGRINS"))
from smooth_continuum import cubic_spline_basis, fit_positive_continuum


def synthetic_spectrum(n_pixels=512):
    pixel = np.arange(n_pixels)
    x = np.linspace(-1., 1., n_pixels)
    continuum = 1.2 + .2 * x - .7 * x ** 2 + .1 * x ** 3
    model = np.ones(n_pixels)
    for center, depth, width in [(70, .4, 3), (220, .999, 6), (460, .8, 4)]:
        model *= 1 - depth * np.exp(-.5 * ((pixel - center) / width) ** 2)
    model[216:225] = 0.
    return continuum, model


def test_recovers_broad_blaze_with_saturated_absorption_and_negative_data():
    continuum, model = synthetic_spectrum()
    observed = continuum * model
    observed[model == 0] = -.003
    basis, _ = cubic_spline_basis(len(model), spacing=64)
    fitted = fit_positive_continuum(observed, model, basis)
    np.testing.assert_allclose(fitted["continuum"], continuum, rtol=0, atol=2e-13)
    np.testing.assert_array_equal(fitted["residual"][model == 0], observed[model == 0])
    assert fitted["diagnostics"]["fitted_pixels"] == len(model)
    assert fitted["diagnostics"]["fitted_nonpositive_observations"] == 9
    assert fitted["diagnostics"]["fitted_zero_model_pixels"] == 9
    assert fitted["diagnostics"]["active_lower_bounds"] == 0


def test_negative_unsaturated_samples_remain_in_least_squares():
    continuum, model = synthetic_spectrum()
    observed = continuum * model
    observed[80] = -.02
    basis, _ = cubic_spline_basis(len(model), spacing=128)
    fitted = fit_positive_continuum(observed, model, basis)
    expected, _, _, _ = np.linalg.lstsq(model[:, None] * basis, observed, rcond=None)
    assert expected.min() > 0
    np.testing.assert_allclose(fitted["coefficients"], expected, rtol=0, atol=2e-13)
    assert fitted["residual"][80] < observed[80]


@pytest.mark.parametrize("explicit_breakpoints", [False, True])
def test_explicit_calibration_mask_predicts_withheld_right_edge(explicit_breakpoints):
    continuum, model = synthetic_spectrum()
    kwargs = (dict(breakpoints=[0, 128, 256, 384, 511]) if explicit_breakpoints
              else dict(spacing=96))
    basis, knots = cubic_spline_basis(len(model), **kwargs)
    mask = (np.arange(len(model)) // 8) % 2 == 0
    mask[-13:] = False
    fitted = fit_positive_continuum(continuum * model, model, basis, mask)
    np.testing.assert_allclose(fitted["continuum"][-13:], continuum[-13:], atol=2e-13)
    np.testing.assert_allclose(fitted["prediction"], continuum * model, atol=2e-13)
    assert fitted["diagnostics"]["fitted_pixels"] == int(mask.sum())
    assert knots[0] == 0 and knots[-1] == 511


def test_active_positive_floor_is_explicit_even_for_negative_observations():
    basis, _ = cubic_spline_basis(100, spacing=32)
    fitted = fit_positive_continuum(-np.ones(100), np.ones(100), basis,
                                     coefficient_floor=1e-7)
    assert np.all(fitted["continuum"] > 0)
    np.testing.assert_allclose(fitted["continuum"], 1e-7, rtol=0, atol=1e-22)
    assert fitted["diagnostics"]["active_lower_bounds"] == basis.shape[1]
    assert fitted["diagnostics"]["fitted_nonpositive_observations"] == 100


def test_rejects_unidentifiable_or_nonfinite_continuum_inputs():
    basis, _ = cubic_spline_basis(100, spacing=32)
    with pytest.raises(ValueError, match="rank deficient"):
        fit_positive_continuum(np.ones(100), np.zeros(100), basis)
    with pytest.raises(ValueError, match="finite"):
        fit_positive_continuum(np.full(100, np.nan), np.ones(100), basis)
    with pytest.raises(ValueError, match="boolean"):
        fit_positive_continuum(np.ones(100), np.ones(100), basis, np.ones(100))
    with pytest.raises(ValueError, match="Breakpoints"):
        cubic_spline_basis(100, breakpoints=[0, 50, 50, 99])
