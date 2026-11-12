"""Positive log-spline continua and their exact local template response."""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def _inputs(observed, model, basis):
    observed, model, basis = map(lambda x: np.asarray(x, dtype=float),
                                 (observed, model, basis))
    if (observed.ndim != 1 or model.shape != observed.shape or basis.ndim != 2
            or basis.shape[0] != observed.size or basis.shape[1] == 0):
        raise ValueError("Use matching spectra and a pixel-by-coefficient basis")
    if not all(np.isfinite(x).all() for x in (observed, model, basis)):
        raise ValueError("All samples must be finite; no implicit masking")
    if np.any(model < 0):
        raise ValueError("The fixed spectral model must be nonnegative")
    return observed, model, basis


def _positive_solve(matrix, target):
    eigenvalues = np.linalg.eigvalsh(matrix)
    tolerance = np.max(np.abs(eigenvalues)) * len(matrix) * np.finfo(float).eps
    if eigenvalues[0] <= tolerance:
        raise ValueError("The local Hessian is not positive definite or is numerically rank deficient")
    return np.linalg.solve(matrix, target)


def fit_log_continuum(observed, model, basis, initial_continuum,
                      max_nfev: int = 200) -> dict:
    """Fit ``observed = model * exp(basis @ coefficients)`` in flux units.

    All original samples, including negative measured flux and saturated
    absorption, enter the native-flux least squares. Wide log-coefficient
    bounds only guard exponentiation; any active bounds are reported.
    """
    observed, model, basis = _inputs(observed, model, basis)
    initial_continuum = np.asarray(initial_continuum, dtype=float)
    if (initial_continuum.shape != observed.shape
            or not np.isfinite(initial_continuum).all() or np.any(initial_continuum <= 0)):
        raise ValueError("The initial continuum must be finite and strictly positive")
    if np.linalg.matrix_rank(model[:, None] * basis) != basis.shape[1]:
        raise ValueError("The continuum design is rank deficient")
    initial = np.linalg.lstsq(basis, np.log(initial_continuum), rcond=None)[0]
    initial = np.clip(initial, -40. + 1e-8, 20. - 1e-8)

    def prediction(coefficients):
        return model * np.exp(basis @ coefficients)

    fit = least_squares(lambda a: prediction(a) - observed, initial,
                        jac=lambda a: prediction(a)[:, None] * basis,
                        bounds=(-40., 20.), method="trf", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=max_nfev)
    coefficients = fit.x.copy()
    # Cost tolerances can stop before derivatives are accurate enough for weak
    # template response. Polish the same objective using its exact Hessian.
    polish_steps = 0
    if not fit.active_mask.any():
        for _ in range(8):
            predicted = prediction(coefficients)
            residual = predicted - observed
            jacobian = predicted[:, None] * basis
            gradient = jacobian.T @ residual
            hessian = basis.T @ ((predicted * (2 * predicted - observed))[:, None] * basis)
            try:
                step = _positive_solve(hessian, -gradient)
            except ValueError:
                break
            if np.max(np.abs(step)) < 1e-13:
                break
            accepted = False
            for scale in 2. ** -np.arange(16):
                candidate = coefficients + scale * step
                if np.any(candidate <= -40.) or np.any(candidate >= 20.):
                    continue
                candidate_prediction = prediction(candidate)
                difference = candidate_prediction - predicted
                cost_change = 2 * residual @ difference + difference @ difference
                candidate_gradient = basis.T @ (candidate_prediction * (candidate_prediction - observed))
                roundoff = 32 * np.finfo(float).eps * (np.abs(residual) @ np.abs(predicted))
                stationary_improvement = (
                    np.max(np.abs(scale * step)) < 1e-5
                    and np.max(np.abs(candidate_gradient)) < .5 * np.max(np.abs(gradient))
                    and cost_change <= roundoff)
                if cost_change <= 0 or stationary_improvement:
                    coefficients = candidate
                    polish_steps += 1
                    accepted = True
                    break
            if not accepted:
                break
    continuum = np.exp(basis @ coefficients)
    predicted = model * continuum
    jacobian = predicted[:, None] * basis
    gradient = jacobian.T @ (predicted - observed)
    active = np.where(coefficients <= -40. + 1e-8, -1,
                      np.where(coefficients >= 20. - 1e-8, 1, 0))
    free_gradient = gradient.copy()
    free_gradient[(active == -1) & (gradient > 0)] = 0
    free_gradient[(active == 1) & (gradient < 0)] = 0
    optimality = float(np.max(np.abs(free_gradient)))
    return dict(coefficients=coefficients, continuum=continuum,
                prediction=predicted, residual=observed - predicted, active_mask=active,
                diagnostics=dict(success=bool(fit.success), nfev=int(fit.nfev),
                    message=str(fit.message), solver="scipy.optimize.least_squares + exact Newton polish",
                    polish_steps=polish_steps, optimality=optimality,
                    stationary=optimality <= 1e-9, design_condition=float(np.linalg.cond(jacobian)),
                    active_bounds=int(np.count_nonzero(active)),
                    fitted_pixels=len(observed), fitted_nonpositive_observations=int(np.sum(observed <= 0)),
                    fitted_zero_model_pixels=int(np.sum(model == 0)),
                    minimum_continuum=float(continuum.min())))


def filter_log_continuum_template(template, observed, model, basis, continuum,
                                  active_mask=None):
    """Apply the exact local refit response with fixed physical model/bounds.

    The last template axis is the pixel axis. This uses the full objective
    Hessian, including its nonzero-residual term, rather than Gauss-Newton.
    Apply the subsequent Q5 response separately in the same flux units.
    """
    observed, model, basis = _inputs(observed, model, basis)
    template, continuum = np.asarray(template, float), np.asarray(continuum, float)
    if (template.ndim < 1 or template.shape[-1] != len(observed)
            or continuum.shape != observed.shape or np.any(continuum <= 0)
            or not np.isfinite(template).all() or not np.isfinite(continuum).all()):
        raise ValueError("Use finite templates and a matching strictly positive continuum")
    if active_mask is None:
        active_mask = np.zeros(basis.shape[1], dtype=bool)
    active_mask = np.asarray(active_mask)
    if (active_mask.shape != (basis.shape[1],)
            or not np.isin(active_mask, [-1, 0, 1]).all()):
        raise ValueError("Active mask must give a boolean or SciPy bound flag per coefficient")
    free = active_mask == 0
    if not free.any():
        return template.copy()
    selected = basis[:, free]
    predicted = model * continuum
    jacobian = predicted[:, None] * selected
    hessian = selected.T @ ((predicted * (2 * predicted - observed))[:, None] * selected)
    flattened = template.reshape(-1, len(observed))
    response = _positive_solve(hessian, jacobian.T @ flattened.T)
    return (flattened - (jacobian @ response).T).reshape(template.shape)
