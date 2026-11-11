"""Verify SVD recovery preserves the finite least-squares problem."""

import importlib
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import svd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"IGRINS"))
from solver_support import install_trf_svd_fallback


def test_trf_recovers_failed_driver_without_changing_matrix(monkeypatch):
    module = importlib.import_module("scipy.optimize._lsq.trf")
    matrix = np.random.default_rng(2).normal(size=(13, 4))
    before = matrix.copy()

    def fail_default(values, **kwargs):
        if kwargs.get("lapack_driver") != "gesvd":
            raise np.linalg.LinAlgError("SVD did not converge")
        return svd(values, **kwargs)

    monkeypatch.setattr(module, "svd", fail_default)
    events = install_trf_svd_fallback()
    u, singular, vh = module.svd(matrix, full_matrices=False)
    np.testing.assert_array_equal(matrix, before)
    np.testing.assert_allclose((u*singular)@vh, matrix, atol=1e-13)
    assert len(events) == 1
    assert events[0]["relative_reconstruction_error"] < 1e-13


def test_trf_does_not_accept_invalid_reconstruction(monkeypatch):
    module = importlib.import_module("scipy.optimize._lsq.trf")

    def broken(values, **kwargs):
        if kwargs.get("lapack_driver") != "gesvd":
            raise np.linalg.LinAlgError("SVD did not converge")
        return np.eye(3), np.zeros(3), np.eye(3)

    monkeypatch.setattr(module, "svd", broken)
    install_trf_svd_fallback()
    with pytest.raises(np.linalg.LinAlgError, match="reconstruction"):
        module.svd(np.eye(3), full_matrices=False)
