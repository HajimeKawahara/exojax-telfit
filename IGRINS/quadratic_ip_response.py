"""Source-normalized Gaussian transport with quadratic detector-pixel width.

The kernel is attached to the input wavelength: an isolated input line has a
unit-area response. A slowly changing width can redistribute a flat spectrum
slightly; the operator is deliberately not divided by its flat-field response.
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import jax
import jax.numpy as jnp
import numpy as np

from exojax.postproc.response import sampling
from exojax.utils.constants import c

IP_LOWER = np.array([-np.log(1.5), -.3, -.3])
IP_UPPER = -IP_LOWER


def _interp_extrapolate(values, nodes, samples):
    """Continue the endpoint slopes through the padded model domain."""
    result = np.interp(values, nodes, samples)
    left = samples[0]+(values-nodes[0])*(samples[1]-samples[0])/(nodes[1]-nodes[0])
    right = samples[-1]+(values-nodes[-1])*(samples[-1]-samples[-2])/(nodes[-1]-nodes[-2])
    return np.where(values < nodes[0], left, np.where(values > nodes[-1], right, result))


def pixel_geometry(block, wavelength_um):
    """Map native input wavelengths to pixels, including each frozen RV shift."""
    wave = np.asarray(wavelength_um)
    grid, target = np.asarray(block["grid"]), np.asarray(block["target"])
    if wave.ndim != 1 or len(wave) != len(target) or np.any(np.diff(target) <= 0):
        raise ValueError("Use the original pixel array in increasing wavenumber")
    pixel = np.arange(len(wave), dtype=float)
    coordinate = 2*pixel/(len(pixel)-1)-1
    # The target already includes the frozen wavelength stretch. Its finite
    # difference gives the local velocity span of one original detector pixel.
    pixel_velocity = np.gradient(c*np.log(target), pixel, edge_order=2)
    central_velocity = float(np.interp((len(pixel)-1)/2, pixel, pixel_velocity))
    if np.any(pixel_velocity <= 0):
        raise ValueError("Detector pixel velocity must be positive")
    shifted_targets = target[None, :]*(1+np.asarray(block["offsets"])[:, None]/c)
    x = np.array([_interp_extrapolate(grid, shifted, coordinate) for shifted in shifted_targets])
    ratio = np.array([_interp_extrapolate(grid, shifted, pixel_velocity/central_velocity)
                      for shifted in shifted_targets])
    if np.min(1-.3*np.abs(x)-.3*x*x) <= 0 or np.min(ratio) <= 0:
        raise ValueError("Padded geometry cannot guarantee positive widths at all IP bounds")
    return dict(x=x, velocity_ratio=ratio, pixel_velocity_kms=pixel_velocity,
                central_pixel_velocity_kms=central_velocity,
                coordinate="Array pixel index mapped to [-1,1], increasing wavenumber",
                padding="Endpoint slopes extrapolated through native padding; all allowed IP widths remain positive")


def validate_ip_parameters(parameters):
    """Reject unsupported widths before passing a parameter vector to JAX."""
    q = np.asarray(parameters)
    if q.shape != (3,) or not np.all(np.isfinite(q)):
        raise ValueError("Use three finite IP parameters")
    if np.any(q < IP_LOWER-1e-12) or np.any(q > IP_UPPER+1e-12):
        raise ValueError("IP parameters exceed the validated width range")


def transport(spectrum, sigma, velocity, wing_fraction=0.):
    """Scatter each source sample through its separately normalized kernel.

    The fixed native-grid support and zero padding match the legacy FFT
    convolution. The surrounding spectral padding keeps its edges away from
    the fitted region. The loop avoids an input-size-squared response matrix.
    """
    radius, size = len(velocity)//2, spectrum.shape[-1]
    core_norm = jnp.sum(jnp.exp(-.5*(velocity[:, None]/sigma[None, :])**2), axis=0)
    if wing_fraction:
        wing_norm = jnp.sum(jnp.exp(-.5*(velocity[:, None]/(2*sigma[None, :]))**2), axis=0)

    def add_offset(index, output):
        weight = (1-wing_fraction)*jnp.exp(-.5*(velocity[index]/sigma)**2)/core_norm
        if wing_fraction:
            weight += wing_fraction*jnp.exp(-.5*(velocity[index]/(2*sigma))**2)/wing_norm
        padding = [(0, 0)]*(spectrum.ndim-1)+[(radius, radius)]
        source = jnp.pad(spectrum*weight, padding)
        shifted = jax.lax.dynamic_slice_in_dim(source, 2*radius-index, size, axis=-1)
        return output+shifted

    return jax.lax.fori_loop(0, len(velocity), add_offset, jnp.zeros_like(spectrum))


def prepare_block(block, wavelength_um, mode="quadratic_pixel", wing_fraction=0.):
    """Return train/full model callbacks for parameters [line corrections, q3].

    In quadratic_pixel mode the Gaussian FWHM in detector pixels is proportional
    to exp(q0)*(1+q1*x+q2*x**2). Existing relative epoch widths remain frozen.
    constant_velocity mode has only the exp(q0) amplitude; q1/q2 are ignored.
    """
    if mode not in {"constant_velocity", "quadratic_pixel"}:
        raise ValueError("Unknown instrumental response mode")
    fraction = float(wing_fraction)
    if not 0 <= fraction < 1:
        raise ValueError("Wing fraction must be in [0,1)")
    geometry = pixel_geometry(block, wavelength_um)
    x, ratio = jnp.asarray(geometry["x"]), jnp.asarray(geometry["velocity_ratio"])
    grid, velocity, target = [jnp.asarray(block[key]) for key in ["grid", "velocity", "target"]]
    if len(velocity) % 2 != 1:
        raise ValueError("Instrumental kernel support must be odd")
    tau, upper = jnp.asarray(block["line_tau"]), jnp.asarray(block["upper"])
    factors, baseline = jnp.asarray(block["line_factors"]), jnp.asarray(block["baseline_tau"])
    star, airmass = jnp.asarray(block["shifted_star"]), jnp.asarray(block["airmass"])
    old_widths, offsets = jnp.asarray(block["widths"]), jnp.asarray(block["offsets"])
    nline = len(upper)

    def widths(q):
        if mode == "constant_velocity":
            shape = jnp.ones_like(x)
        else:
            shape = (1+q[1]*x+q[2]*x*x)*ratio
        return old_widths[:, None]*jnp.exp(q[0])*shape

    def respond(intrinsic, q, n_exposures):
        sigma = widths(q)[:n_exposures]
        broadened = jax.vmap(lambda row, width: transport(row, width, velocity, fraction))(
            intrinsic, sigma)
        return jax.vmap(lambda row, rv: sampling(target, grid, row, rv))(
            broadened, offsets[:n_exposures])

    def forward(parameters, n_exposures):
        z, q = parameters[:nline], parameters[nline:]
        delta_tau = (factors[:n_exposures]*(upper*z)[None, :]) @ tau
        intrinsic = star[:n_exposures]*jnp.exp(
            -airmass[:n_exposures, None]*(baseline[:n_exposures]+delta_tau))
        return respond(intrinsic, q, n_exposures)

    def jacobian(parameters, n_exposures):
        # Differentiate line opacity analytically, applying one shared linear
        # response to all line derivatives. Only the three IP derivatives need
        # autodiff through kernel widths and normalization.
        z, q = parameters[:nline], parameters[nline:]
        delta_tau = (factors[:n_exposures]*(upper*z)[None, :]) @ tau
        intrinsic = star[:n_exposures]*jnp.exp(
            -airmass[:n_exposures, None]*(baseline[:n_exposures]+delta_tau))
        derivative = (-airmass[:n_exposures, None, None]*intrinsic[:, None, :]
                      * (factors[:n_exposures]*upper)[..., None]*tau[None, :, :])
        broadened = jax.vmap(lambda rows, width: transport(rows, width, velocity, fraction))(
            derivative, widths(q)[:n_exposures])
        line_jac = jax.vmap(lambda rows, rv: jax.vmap(
            lambda row: sampling(target, grid, row, rv))(rows))(
                broadened, offsets[:n_exposures])
        ip_jac = jax.jacfwd(lambda ip: respond(intrinsic, ip, n_exposures))(q)
        return jnp.concatenate([jnp.swapaxes(line_jac, 1, 2), ip_jac], axis=-1)

    full_count = len(old_widths)
    return dict(
        train_forward=jax.jit(lambda parameters: forward(parameters, 5)),
        forward=jax.jit(lambda parameters: forward(parameters, full_count)),
        train_jacobian=jax.jit(lambda parameters: jacobian(parameters, 5)),
        jacobian=jax.jit(lambda parameters: jacobian(parameters, full_count)),
        widths=jax.jit(widths),
        response=jax.jit(lambda intrinsic, q: respond(intrinsic, q, full_count)),
        geometry=geometry, mode=mode, wing_fraction=fraction, n_line_parameters=nline,
        parameterization="sigma_kms=old_epoch_sigma*exp(q0)*(1+q1*x+q2*x^2)*local_pixel_velocity/central_pixel_velocity",
        normalization="Each input sample has a separately unit-normalized kernel; no output flat-field normalization",
        support_kms=[float(velocity[0]), float(velocity[-1])],
    )
