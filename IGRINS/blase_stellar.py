"""Anchor a pretrained, real blase JAX clone to the existing PHOENIX template.

``build_clone(grid_nu, order)`` returns a ``StellarClone``. Its
``corrected_raw(q)`` accepts one nonnegative amplitude ratio per fitted feature
and returns a differentiable spectrum on the original ascending wavenumber grid.
At ``q == 1`` it is exactly the original PHOENIX spectrum. Apply stellar
broadening/RV after this correction, and instrumental convolution after the
stellar spectrum is multiplied by telluric transmission.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import least_squares

from stellar_template import CACHE as PHOENIX_PATH, load_stellar

ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor/blase"
SOURCE = VENDOR / "blase/jaxemulator.py"
COMMIT = "6f6817b0c4095c104a10b2940375fc1e16cb80ca"
SETTINGS = dict(version=1, padding_angstrom=65., prominence=.003,
                fit_margin_angstrom=5., wing_cut_pixels=1000, max_nfev=240, ftol=1e-6,
                maximum_clone_depth=.98, sigma_bounds_angstrom=[.002, 2.],
                gamma_bounds_angstrom=[.0001, 2.])


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_upstream():
    spec = importlib.util.spec_from_file_location("igrins_blase_jax", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SparseLinearEmulator


def _fix_sparse_indices(model, wing_pixels):
    """Use consecutive integer support; leave the vendored source unchanged."""
    count = min(wing_pixels, model.n_pix)
    centers = np.searchsorted(np.asarray(model.wl_native), np.asarray(model.lam_centers))
    starts = np.clip(centers-count//2, 0, model.n_pix-count)
    model.indices_2D = jnp.asarray(starts[:, None]+np.arange(count), dtype=jnp.int32)
    model.indices_1D = model.indices_2D.reshape(-1)
    model.indices = np.expand_dims(np.asarray(model.indices_1D), axis=0)
    model.wl_2D = model.wl_native[model.indices_2D]
    model.wl_1D = model.wl_2D.reshape(-1)


def _peak_profile(model, log_sigma, log_gamma):
    """Evaluate the exact upstream pseudo-Voigt peak in inverse Angstroms."""
    sigma, gamma = jnp.exp(log_sigma), jnp.exp(log_gamma)
    fwhm = model._compute_fwhm(2*gamma, 2.3548*sigma)
    eta = model._compute_eta(2*gamma, fwhm)
    return (eta*model._lorentzian_line(0., gamma, 0.)
            + (1-eta)*model._gaussian_line(0., sigma, 0.))


@dataclass
class StellarClone:
    """Fixed-center/width clone with selected observational amplitude ratios."""

    raw_reference: np.ndarray
    centers_angstrom: np.ndarray
    base_log_amplitudes: np.ndarray
    base_log_sigma_widths: np.ndarray
    base_log_gamma_widths: np.ndarray
    amplitude_ratio_upper: np.ndarray
    metadata: dict
    _model: object
    _fit_indices: jax.Array
    _wavelength_native: jax.Array
    _reference_clone: jax.Array

    @property
    def n_lines(self):
        return len(self.centers_angstrom)

    def corrected_raw(self, q):
        """Return PHOENIX times the shared rest-frame clone amplitude ratio.

        Bounds should be 0 <= q <= amplitude_ratio_upper. The tiny positive
        floor permits q=0 without log/gradient NaNs; upstream already imposes
        a 1e-6 absorption floor on each sparse line profile.
        """
        q = jnp.asarray(q)
        if q.shape != (self.n_lines,):
            raise ValueError(f"Expected {self.n_lines} amplitude ratios, got {q.shape}.")
        log_amp = self._model.amplitudes.at[self._fit_indices].add(
            jnp.log(jnp.maximum(q, 1e-12)))
        changed = self._model.forward(log_amp, self._model.sigma_widths,
                                      self._model.gamma_widths)
        ratio = changed/self._reference_clone
        native_ratio = jnp.interp(self._wavelength_native, self._model.wl_active, ratio)
        return jnp.asarray(self.raw_reference)*native_ratio[::-1]


def build_clone(grid_nu, order):
    """Pretrain/cache the clone on noiseless PHOENIX, without observed data."""
    started = time.monotonic()
    grid = np.asarray(grid_nu, dtype=np.float64)
    if grid.ndim != 1 or len(grid) < 3 or np.any(np.diff(grid) <= 0):
        raise ValueError("grid_nu must be an ascending, finite wavenumber vector.")
    if not np.isfinite(grid).all():
        raise ValueError("grid_nu must be finite.")
    wavelength = 1e8/grid[::-1]
    log_step = float(np.median(np.diff(np.log(wavelength))))
    if not np.allclose(np.diff(np.log(wavelength)), log_step, rtol=1e-6, atol=1e-12):
        raise ValueError("The clone adapter requires a constant log-wavelength step.")
    left = int(np.ceil(np.log(wavelength[0]/(wavelength[0]-65.))/log_step))
    right = int(np.ceil(np.log((wavelength[-1]+65.)/wavelength[-1])/log_step))
    padded_wave = wavelength[0]*np.exp(np.arange(-left, len(wavelength)+right)*log_step)
    # load_stellar validates coverage but accepts either requested grid ordering.
    padded_flux = load_stellar(1e8/padded_wave)
    raw_reference = load_stellar(grid)
    hashes = dict(source_sha256=_sha(SOURCE), helper_sha256=_sha(__file__),
                  phoenix_sha256=_sha(PHOENIX_PATH),
                  native_grid_sha256=hashlib.sha256(grid.tobytes()).hexdigest())
    signature = dict(order=int(order), upstream_commit=COMMIT, settings=SETTINGS, **hashes)
    token = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
    cache = ROOT / "cache" / f"blase_clone_o{order}_{token}.npz"
    metadata_path = cache.with_suffix(".json")
    provenance = dict(repository="https://github.com/gully/blase", commit=COMMIT,
                      source_url=f"https://github.com/gully/blase/blob/{COMMIT}/src/blase/jaxemulator.py",
                      source_sha256=hashes["source_sha256"], license="MIT",
                      license_sha256=_sha(VENDOR / "LICENSE"),
                      installation="Unmodified upstream JAX module imported directly; no PyTorch installed",
                      adapter_change="Consecutive sparse integer indices replace upstream endpoint-inclusive linspace; vendor unchanged")
    (VENDOR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2)+"\n")
    emulator = _load_upstream()
    metadata = None
    load_path = cache if cache.exists() and metadata_path.exists() else None
    if load_path is None:
        # A completed earlier run can warm-start a changed optimizer/helper;
        # source, spectrum, grid, and feature-selection settings must agree.
        for candidate in sorted((ROOT / "cache").glob(f"blase_clone_o{order}_*.json")):
            old = json.loads(candidate.read_text())
            same_inputs = all(old.get(key) == hashes[key] for key in
                              ["source_sha256", "phoenix_sha256", "native_grid_sha256"])
            same_features = all(old["settings"].get(key) == SETTINGS[key] for key in
                                ["padding_angstrom", "prominence", "fit_margin_angstrom",
                                 "wing_cut_pixels", "maximum_clone_depth",
                                 "sigma_bounds_angstrom", "gamma_bounds_angstrom"])
            if same_inputs and same_features and candidate.with_suffix(".npz").exists():
                load_path = candidate.with_suffix(".npz")
    if load_path is not None:
        with np.load(load_path) as saved:
            state = {key: saved[key] for key in
                     ["lam_centers", "amplitudes", "sigma_widths", "gamma_widths"]}
            fit_indices = saved["fit_indices"]
        if load_path == cache:
            metadata = json.loads(metadata_path.read_text())
        model = emulator(padded_wave, flux_native=padded_flux, init_state_dict=state,
                         wing_cut_pixels=SETTINGS["wing_cut_pixels"])
        _fix_sparse_indices(model, SETTINGS["wing_cut_pixels"])
    else:
        model = emulator(padded_wave, flux_native=padded_flux,
                         prominence=SETTINGS["prominence"],
                         wing_cut_pixels=SETTINGS["wing_cut_pixels"])
        _fix_sparse_indices(model, SETTINGS["wing_cut_pixels"])
        fit_indices = np.flatnonzero(
            (np.asarray(model.lam_centers) >= wavelength[0]-5.)
            & (np.asarray(model.lam_centers) <= wavelength[-1]+5.))
    if metadata is None or not metadata["optimization"]["success"]:
        if not len(fit_indices):
            raise ValueError("No PHOENIX features passed the fixed prominence threshold.")
        take = jnp.asarray(fit_indices)
        native_mask = ((np.asarray(model.wl_active) >= wavelength[0])
                       & (np.asarray(model.wl_active) <= wavelength[-1]))
        fit_pixels = jnp.asarray(np.flatnonzero(native_mask))
        count = len(fit_indices)
        initial_sigma = np.clip(np.asarray(model.sigma_widths)[fit_indices],
                                np.log(.00201), np.log(1.999))
        initial_gamma = np.clip(np.asarray(model.gamma_widths)[fit_indices],
                                np.log(.000101), np.log(1.999))
        initial_depth = np.clip(np.exp(np.asarray(model.amplitudes)[fit_indices])
                                * np.asarray(_peak_profile(model, initial_sigma, initial_gamma)),
                                1.01e-5, .979)
        initial = np.r_[np.log(initial_depth), initial_sigma, initial_gamma]

        def unpack(parameters):
            log_depth, log_sigma, log_gamma = jnp.split(parameters, 3)
            log_amplitude = log_depth-jnp.log(_peak_profile(model, log_sigma, log_gamma))
            return (model.amplitudes.at[take].set(log_amplitude),
                    model.sigma_widths.at[take].set(log_sigma),
                    model.gamma_widths.at[take].set(log_gamma))

        def residual(parameters):
            return (model.forward(*unpack(parameters))-model.target)[fit_pixels]

        fun, jac = jax.jit(residual), jax.jit(jax.jacfwd(residual))
        bounds = (np.r_[np.full(count, np.log(1e-5)), np.full(count, np.log(.002)),
                        np.full(count, np.log(.0001))],
                  np.r_[np.full(count, np.log(.98)), np.full(count, np.log(2.)),
                        np.full(count, np.log(2.))])
        print(f"Cloning PHOENIX order {order}: {count} fitted / {model.n_lines} total blase features", flush=True)
        result = least_squares(lambda p: np.asarray(fun(p)), initial,
                               jac=lambda p: np.asarray(jac(p)), bounds=bounds,
                               max_nfev=SETTINGS["max_nfev"], x_scale="jac",
                               ftol=SETTINGS["ftol"], xtol=1e-7, gtol=1e-7)
        arrays = [np.asarray(value) for value in unpack(result.x)]
        model.amplitudes, model.sigma_widths, model.gamma_widths = map(jnp.asarray, arrays)
        error = np.asarray(model.forward(*arrays)-model.target)[native_mask]
        metadata = dict(**signature, created_utc=datetime.now(timezone.utc).isoformat(),
                        fitted_features=count, total_detected_features=int(model.n_lines),
                        fit_pixels=int(native_mask.sum()), clone_rms_percent=float(100*np.sqrt(np.mean(error**2))),
                        clone_max_absolute_percent=float(100*np.max(np.abs(error))),
                        optimization=dict(success=bool(result.success), status=int(result.status),
                                          message=result.message, nfev=int(result.nfev),
                                          cost=float(result.cost), optimality=float(result.optimality),
                                          warm_start_file=None if load_path is None else str(load_path)),
                        pretraining="Bounded least squares on noiseless PHOENIX; fixed centers; amplitudes and widths trained only within native range plus 5 Angstrom",
                        transfer="Only selected amplitudes vary; corrected raw spectrum is PHOENIX times clone(q)/clone(ones)",
                        caveat="Clone features are phenomenological blends, not individual atomic transitions; PHOENIX normalization above unity and clone mismatch are retained by ratio anchoring",
                        seconds=time.monotonic()-started)
        np.savez_compressed(cache, lam_centers=np.asarray(model.lam_centers),
                            amplitudes=arrays[0], sigma_widths=arrays[1], gamma_widths=arrays[2],
                            fit_indices=fit_indices, grid_nu=grid)
        metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
        print(f"Order {order} clone RMS {metadata['clone_rms_percent']:.4f}% ({result.nfev} evaluations)", flush=True)
    reference_clone = model.forward(model.amplitudes, model.sigma_widths, model.gamma_widths)
    if np.min(np.asarray(reference_clone)) < 1e-8:
        raise ValueError("Clone flux is too close to zero for stable ratio anchoring.")
    indices = jnp.asarray(fit_indices)
    depths = np.exp(np.asarray(model.amplitudes)[fit_indices])*np.asarray(
        _peak_profile(model, model.sigma_widths[indices], model.gamma_widths[indices]))
    all_depths = np.exp(np.asarray(model.amplitudes))*np.asarray(
        _peak_profile(model, model.sigma_widths, model.gamma_widths))
    untrained = np.ones(model.n_lines, dtype=bool)
    untrained[fit_indices] = False
    metadata["stability"] = dict(minimum_active_clone_flux=float(np.min(np.asarray(reference_clone))),
                                  maximum_untrained_line_peak_depth=float(np.max(all_depths[untrained])),
                                  untrained_saturated_features=int(np.sum(all_depths[untrained] >= 1.)))
    metadata_path.write_text(json.dumps(metadata, indent=2)+"\n")
    clone = StellarClone(raw_reference=raw_reference,
                         centers_angstrom=np.asarray(model.lam_centers)[fit_indices],
                         base_log_amplitudes=np.asarray(model.amplitudes)[fit_indices],
                         base_log_sigma_widths=np.asarray(model.sigma_widths)[fit_indices],
                         base_log_gamma_widths=np.asarray(model.gamma_widths)[fit_indices],
                         amplitude_ratio_upper=np.minimum(3., (1-1e-5)/depths),
                         metadata=dict(metadata, cache_file=str(cache)), _model=model,
                         _fit_indices=indices, _wavelength_native=jnp.asarray(wavelength),
                         _reference_clone=reference_clone)
    np.testing.assert_array_equal(np.asarray(clone.corrected_raw(np.ones(clone.n_lines))), raw_reference)
    return clone
