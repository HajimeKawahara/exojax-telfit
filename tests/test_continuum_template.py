"""Check the continuum response against independent constrained fits."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize import nnls

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "IGRINS"))
from smooth_continuum import cubic_spline_basis, filter_continuum_template


@pytest.mark.parametrize("strength", [0., 10.])
@pytest.mark.parametrize("use_mask", [False, True])
def test_template_response_matches_constrained_fit(strength, use_mask):
    basis, _ = cubic_spline_basis(300, spacing=60.)
    pixel = np.arange(len(basis))
    model = 1 - .5 * np.exp(-.5 * ((pixel - 125) / 2.) ** 2)
    design = model[:, None] * basis
    penalty = np.diff(np.eye(basis.shape[1]), n=2, axis=0)
    mask = (pixel % 3 != 0) if use_mask else np.ones(len(pixel), dtype=bool)
    selected = np.vstack((design[mask], np.sqrt(strength) * penalty))
    # Negative flux at one edge makes at least one constraint active.
    observed = model * (.3 + .002 * pixel) - .7 * np.exp(-pixel / 25.)
    floor = 1e-8

    def fit(values):
        target = np.r_[values[mask], np.zeros(len(penalty))]
        excess = nnls(selected, target - selected @ np.full(basis.shape[1], floor))[0]
        return values - design @ (excess + floor), excess == 0

    baseline, active = fit(observed)
    assert active.any()
    template = -.02 * np.exp(-.5 * ((pixel - 9) / 1.4) ** 2)
    epsilon = 1e-5
    perturbed, new_active = fit(observed + epsilon * template)
    np.testing.assert_array_equal(active, new_active)
    expected = (perturbed - baseline) / epsilon
    actual = filter_continuum_template(template, model, basis, active,
                                       regularization_strength=strength, fit_mask=mask)
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-7)


def test_batch_and_all_active_templates():
    basis, _ = cubic_spline_basis(100, spacing=30.)
    model = np.ones(len(basis))
    templates = np.eye(2, len(basis), k=10)
    active = np.ones(basis.shape[1], dtype=bool)
    np.testing.assert_array_equal(filter_continuum_template(templates, model, basis, active),
                                  templates)
    batch = filter_continuum_template(templates, model, basis)
    singles = np.stack([filter_continuum_template(row, model, basis) for row in templates])
    np.testing.assert_allclose(batch, singles, atol=1e-14)
