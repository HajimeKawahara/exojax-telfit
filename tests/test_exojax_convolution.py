"""FFT convolution preserves signal length and centered alignment."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.signal import convolve

from exojax.signal.convolve import convolve_same


@pytest.mark.parametrize("signal_length", [5, 6])
@pytest.mark.parametrize("kernel_length", [1, 3, 4, 9])
def test_convolve_same_matches_scipy(signal_length, kernel_length):
    signal = np.linspace(-1.0, 2.0, signal_length, dtype=np.float32)
    kernel = np.arange(1, kernel_length + 1, dtype=np.float32)
    kernel /= kernel.sum()

    actual = jax.jit(convolve_same)(jnp.asarray(signal), jnp.asarray(kernel))
    expected = convolve(signal, kernel, mode="same", method="direct")

    assert actual.shape == signal.shape
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-7)
