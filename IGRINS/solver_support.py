"""Recover an occasional LAPACK divide-and-conquer SVD failure in TRF."""

from __future__ import annotations

import importlib

import numpy as np


def install_trf_svd_fallback():
    """Keep TRF unchanged, retrying only failed finite-matrix SVDs with GESVD."""
    module = importlib.import_module("scipy.optimize._lsq.trf")
    original = module.svd
    events = []

    def svd(matrix, *args, **kwargs):
        try:
            return original(matrix, *args, **kwargs)
        except np.linalg.LinAlgError:
            if not np.isfinite(matrix).all():
                raise
            options = dict(kwargs, lapack_driver="gesvd")
            u, singular, vh = original(matrix, *args, **options)
            rank = len(singular)
            reconstructed = (u[:, :rank]*singular) @ vh[:rank]
            error = float(np.linalg.norm(reconstructed-matrix)
                          / max(np.linalg.norm(matrix), np.finfo(float).tiny))
            if not np.isfinite(error) or error > 1e-10:
                raise np.linalg.LinAlgError("Fallback SVD failed reconstruction")
            event = dict(driver="gesvd", matrix_shape=list(matrix.shape),
                         relative_reconstruction_error=error)
            events.append(event)
            print("TRF SVD fallback", event, flush=True)
            return u, singular, vh

    module.svd = svd
    return events
