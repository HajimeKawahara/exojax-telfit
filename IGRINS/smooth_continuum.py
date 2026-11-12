"""Positive, smooth continua fitted directly in observed flux units."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import BSpline
from scipy.optimize import nnls


def cubic_spline_basis(n_pixels: int, spacing: float = 128.,
                       breakpoints=None) -> tuple[np.ndarray, np.ndarray]:
    """Return a clamped cubic basis and knots on pixel coordinates 0..N-1.

    Explicit breakpoints must include both endpoints. Otherwise, uniform
    intervals no wider than ``spacing`` pixels span the complete order,
    avoiding a short final interval. Basis functions sum to one.
    """
    if not isinstance(n_pixels, (int, np.integer)) or n_pixels < 4:
        raise ValueError("At least four pixels are required")
    if breakpoints is None:
        if not np.isfinite(spacing) or spacing <= 0:
            raise ValueError("Spline spacing must be finite and positive")
        intervals = int(np.ceil((n_pixels - 1) / spacing))
        breaks = np.linspace(0., n_pixels - 1., intervals + 1)
    else:
        breaks = np.asarray(breakpoints, dtype=float)
    if (breaks.ndim != 1 or len(breaks) < 2
            or not np.isfinite(breaks).all() or np.any(np.diff(breaks) <= 0)
            or breaks[0] != 0 or breaks[-1] != n_pixels - 1):
        raise ValueError("Breakpoints must increase from zero to n_pixels - 1")
    knots = np.r_[np.repeat(breaks[0], 3), breaks, np.repeat(breaks[-1], 3)]
    basis = BSpline.design_matrix(np.arange(n_pixels), knots, 3).toarray()
    return basis, knots


def fit_positive_continuum(observed, model, basis, fit_mask=None,
                           coefficient_floor: float = 1e-8,
                           regularization: float = 0.) -> dict:
    """Fit ``observed = continuum * model`` with positive spline coefficients.

    ``observed`` and the returned continuum share the input flux scale. No
    division by the model, absorption selection, or rejection of negative
    measured flux is performed. An explicit mask restricts calibration only;
    predictions and residuals always cover every input pixel. Active lower
    bounds are reported because positivity alone does not establish a good fit.
    An optional squared second-coefficient-difference penalty limits curvature.
    """
    observed, model, basis = (np.asarray(value, dtype=float)
                              for value in (observed, model, basis))
    if (observed.ndim != 1 or model.shape != observed.shape or basis.ndim != 2
            or basis.shape[0] != observed.size or basis.shape[1] == 0):
        raise ValueError("Use matching one-dimensional spectra and a pixel-by-basis matrix")
    if not all(np.isfinite(value).all() for value in (observed, model, basis)):
        raise ValueError("All input samples must be finite; no implicit masking")
    if (np.any(model < 0) or np.any(basis < 0)
            or not np.allclose(basis.sum(axis=1), 1., rtol=0, atol=1e-12)):
        raise ValueError("Model and basis must be nonnegative; basis rows must sum to one")
    if not np.isfinite(coefficient_floor) or coefficient_floor <= 0:
        raise ValueError("The coefficient floor must be finite and strictly positive")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("The curvature penalty must be finite and nonnegative")
    if fit_mask is None:
        fit_mask = np.ones(observed.size, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask)
        if fit_mask.shape != observed.shape or fit_mask.dtype != np.bool_:
            raise ValueError("The explicit fitting mask must be a matching boolean array")
    design = model[:, None] * basis
    selected = design[fit_mask]
    singular = np.linalg.svd(selected, compute_uv=False)
    tolerance = (singular[0] * max(selected.shape) * np.finfo(float).eps
                 if singular.size else 0.)
    rank = int(np.count_nonzero(singular > tolerance))
    if rank != basis.shape[1]:
        raise ValueError("The selected continuum design is rank deficient")
    offset = selected @ np.full(basis.shape[1], coefficient_floor)
    response = observed[fit_mask] - offset
    if regularization:
        penalty = np.sqrt(regularization) * np.diff(np.eye(basis.shape[1]), n=2, axis=0)
        selected = np.vstack([selected, penalty])
        response = np.r_[response, np.zeros(penalty.shape[0])]
    excess, _ = nnls(selected, response,
                     maxiter=10 * basis.shape[1])
    coefficients = excess + coefficient_floor
    continuum = basis @ coefficients
    prediction = model * continuum
    residual = observed - prediction
    return dict(coefficients=coefficients, continuum=continuum,
                prediction=prediction, residual=residual,
                diagnostics=dict(
                    solver="scipy.optimize.nnls", rank=rank,
                    curvature_regularization=float(regularization),
                    design_condition=float(singular[0] / singular[-1]),
                    fitted_pixels=int(fit_mask.sum()),
                    fitted_nonpositive_observations=int(np.count_nonzero(observed[fit_mask] <= 0)),
                    fitted_zero_model_pixels=int(np.count_nonzero(model[fit_mask] == 0)),
                    coefficient_floor=float(coefficient_floor),
                    active_lower_bounds=int(np.count_nonzero(excess == 0)),
                    minimum_continuum=float(continuum.min()),
                    fitted_rms=float(np.sqrt(np.mean(residual[fit_mask] ** 2))),
                    all_pixel_rms=float(np.sqrt(np.mean(residual ** 2)))))


def filter_continuum_template(template, model, basis, active_lower_mask=None,
                              regularization_strength: float = 0.,
                              fit_mask=None):
    """Apply the local continuum-fit response to flux-space templates.

    The last template axis is the observed pixel axis. The spectral model,
    fit mask, and active coefficient bounds are held fixed. ``True`` entries
    in ``active_lower_mask`` identify coefficients fixed at the lower bound.
    Regularization penalizes second differences of spline coefficients, using
    the same strength as the continuum fit. Apply the residual Q5 projection
    separately after this function, in the same flux units as the data.

    This is a local derivative, not an injection-recovery validation. If a
    perturbation changes the active bounds, refit the perturbed data instead.
    Broad modes and narrow lines near order boundaries can lose appreciable
    amplitude; use the filtered templates when evaluating CCF amplitudes.
    """
    template, model, basis = (np.asarray(value, dtype=float)
                              for value in (template, model, basis))
    if (model.ndim != 1 or basis.ndim != 2 or basis.shape[0] != model.size
            or template.ndim < 1 or template.shape[-1] != model.size):
        raise ValueError("Templates must end in the model's pixel axis")
    if not all(np.isfinite(value).all() for value in (template, model, basis)):
        raise ValueError("Template, model and basis must be finite")
    if not np.isfinite(regularization_strength) or regularization_strength < 0:
        raise ValueError("Regularization strength must be finite and nonnegative")
    if active_lower_mask is None:
        active_lower_mask = np.zeros(basis.shape[1], dtype=bool)
    else:
        active_lower_mask = np.asarray(active_lower_mask)
        if (active_lower_mask.shape != (basis.shape[1],)
                or active_lower_mask.dtype != np.bool_):
            raise ValueError("Active bounds must be a boolean array per coefficient")
    if fit_mask is None:
        fit_mask = np.ones(model.size, dtype=bool)
    else:
        fit_mask = np.asarray(fit_mask)
        if fit_mask.shape != model.shape or fit_mask.dtype != np.bool_:
            raise ValueError("The fit mask must be a boolean array per pixel")
    free = ~active_lower_mask
    if not free.any():
        return template.copy()
    design = model[:, None] * basis[:, free]
    selected = design[fit_mask]
    flattened = template.reshape(-1, model.size)
    target = flattened[:, fit_mask].T
    if regularization_strength:
        penalty = np.diff(np.eye(basis.shape[1]), n=2, axis=0)[:, free]
        selected = np.vstack((selected, np.sqrt(regularization_strength) * penalty))
        target = np.vstack((target, np.zeros((penalty.shape[0], target.shape[1]))))
    response, _, rank, _ = np.linalg.lstsq(selected, target, rcond=None)
    if rank != int(free.sum()):
        raise ValueError("The free continuum response is rank deficient")
    return (flattened - (design @ response).T).reshape(template.shape)
