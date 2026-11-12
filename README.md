# exojax-telfit

Experimental telluric fitting with ExoJAX, developed on the WASP-77Ab IGRINS
time series. The reference workflow fits **Order 7**, including strong absorption,
with a layered terrestrial atmosphere, PHOENIX plus a blasé stellar correction,
and regularized HITRAN line-strength corrections.
The same fitting procedure can be applied independently to all 44 released
orders, followed by a separate 79-frame export for CCF analysis.

This repository contains the implementation and earlier diagnostic experiments.
Observations, line lists, model checkpoints, numerical results, and **enuma data
are not included**. The original ExoJAX repository and its local IGRINS products
are unchanged by this export. A continuum comparison figure is included below.

## Setup

The reference environment used Python 3.12 on Linux with CPU JAX in 64-bit mode.
Install the recorded ExoJAX source revision and the small convolution fix used by
the fitting runs:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
mkdir -p _deps
git clone https://github.com/HajimeKawahara/exojax.git _deps/exojax
git -C _deps/exojax checkout 19fe37b5509d931e6fb85bcd731be60877751544
git -C _deps/exojax apply ../../patches/exojax-convolution.patch
python -m pip install -r requirements.txt -e ./_deps/exojax
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=4
export OMP_NUM_THREADS=4
export MPLCONFIGDIR=/tmp/igrins-matplotlib
```

The patch preserves the requested FFT length and the exact output length of
`convolve_same`, including odd FFT lengths and one-element kernels. It is
distributed separately rather than copying the ExoJAX source tree.

## Current Order 7 workflow

Run commands from this repository's root. Inputs and outputs stay under
`IGRINS/data`, `IGRINS/cache`, and `IGRINS/results`, all ignored by Git.

```bash
python IGRINS/prepare_data.py
python IGRINS/prepare_hitran.py
python IGRINS/prepare_stellar.py
python IGRINS/prepare_atmosphere.py
python IGRINS/prepare_order7_scope.py

python IGRINS/run_order7.py
python IGRINS/fit_order7_stellar.py --strength-solver lbfgsb --null-refits 8
python IGRINS/diagnose_order7_species.py
python IGRINS/check_order7.py
python IGRINS/report_order7.py
python IGRINS/activate_order7.py
```

See [data preparation](docs/DATA.md) for source URLs, checksums, reference
atmospheric assumptions, and download limitations. The HITRAN endpoint is live:
preserve the downloaded files and their metadata when comparing exact results.
The dense opacity and line-response calculations can require substantial RAM.

The first stellar-clone preparation has a bounded evaluation budget. If it stops
with a message that the PHOENIX clone has not converged, rerun the same stellar-fit
command: the clone checkpoint is retained and used as the starting point.
For an interrupted observational fit with a saved checkpoint, add `--resume`.
`--polish` also requires `--resume` and an existing stellar/telluric checkpoint.

The initializer in `config/order7_initial.json` preserves the earlier physical
starting values, including the fixed stellar velocity and extra broadening.
Historical fitted arrays are not required to start this workflow. Earlier
coarse/tight solver-control tables appear in the report only when their local
control files exist; they are not fabricated for a new run.

## All orders and the observed CCF series

After preparing the same inputs, run:

```bash
python IGRINS/run_all_orders.py --jobs 8 --cpus-per-job 8 \
  --degree-four-continuum-orders $(seq 19 35) \
  --degree-five-continuum-orders $(seq 36 42) \
  --all-pixel-continuum-orders 43
python IGRINS/summarize_all_orders.py --combine --png
python IGRINS/run_ccf_series.py --jobs 2 --cpus-per-job 8
python IGRINS/summarize_ccf_series.py
python IGRINS/refine_continuum.py
python IGRINS/report_continuum_refinement.py --normalized
```

Choose CPU allocations that fit the machine. The shared fit uses the five
training frames and four comparison frames listed below. Existing complete
products are reused, including the frozen Order 7 reference when available.
The all-order report preserves missing or failed orders explicitly. It combines
the nine-frame products only when all 44 pass its checks.

Some full-order blaze shapes require explicit continuum exceptions. Cube
indices 19–35 use degree four, and 36–42 use degree five. Index 43 retains a
cubic but profiles it on all pixels in its comparison frames. The original
cubic profiles crossed zero at faint edges in these orders. The fifth-degree
choice for indices 36–38 also improves independent score-block residuals over
the positive but inadequate fourth-degree profiles. The report records
the actual degree, scope and minimum continuum. These smooth-component changes
do not repair the low throughput or guarantee small normalized residuals.

The CCF application preserves the original fiducial stage ordering. It fits
each new frame's water scaling, wavelength offset, IP width and continuum
against the frozen, uncorrected physical model. It then applies the frozen
stellar and line-strength corrections and the degree-five residual projection,
keeping that continuum fixed. New frames follow the original held-out
calibration rules and each order's adopted continuum degree/scope. The known
nine frames reuse their saved nuisance parameters and continuum, with an exact
conversion between physical and exported flux normalization. Their model and
residual arrays are checked against the saved benchmark in the intermediate
`allorders_ccf_no_injection.npz` product.

The final continuum revision then freezes these physical, stellar and
line-strength parameters and fits a positive cubic B-spline continuum to each
observed frame. Uniform intervals span the complete order, with a nominal
maximum spacing of 128 pixels (18 coefficients per frame). The fit works
directly in observed flux units and retains deep absorption and negative
measured flux. It recomputes the degree-five residual subtraction after this
continuum fit. Thus the final continua and residuals, including the original
nine frames, differ from the intermediate benchmark.

Stored cube index 39 uses a spline of **log continuum**, `C = exp(B @ a)`,
with the same 18 parameters and native-flux objective. The earlier linear
spline reached its numerical floor at the last pixel in 65 frames; division
by this tiny continuum let one sample contaminate the whole-order Q5 fit.
The log parameterization removes that endpoint collapse without masking pixels
or raising a continuum floor. It converges for all 79 frames, with minimum
continuum 0.0006403 in frame-p95 units and no active numerical guards. This
repair changes only index 39; all other orders, including user Order 7, retain
their previous continuum and residual arrays.

![Stored index 39 after the endpoint repair](docs/figures/order39_log_continuum.png)

The whole-order normalization artifact is removed. The faint-edge noise and
individual spectral-line mismatches remain visible; the repair does not
establish photon-limited residuals or planet-signal recovery.

This local flexibility addresses the coherent high-wavenumber edge shape that
the original whole-order polynomials could not follow. Short-block validation
supports the change. Simultaneously freeing multiplicative splines and additive
polynomials made their decomposition unstable; a global curvature penalty also
degraded edge predictions. Neither option is adopted.

The CCF product contains the **original observations with no artificial planet
injection**, residuals before and after degree-five smooth subtraction,
stellar/telluric/full model components, original BJD/phase metadata and validity
flags. Fitted wavelength sampling offsets, stretch and per-frame IP parameters
are included for downstream template evaluation; they do not constitute an
independently verified absolute or barycentric wavelength solution.
No temporal mean or common residual template is subtracted, and no
telluric division is applied. Flux residuals are stored as fractions, not
percentages. Wavelength decreases along the last axis; wavenumber increases.
Order coordinates are zero-based cube indices; only cube index 5 has the
confirmed user label Order 7.

`IGRINS/results/allorders_ccf_continuum_refined.npz` is the final combined array file.
Its generated README describes the fields and quality flags. Very faint
continua can amplify continuum-normalized noise; use
`residual_fixed_scale_q5_independent` to apply the smooth subtraction directly
in units of each frame's p95 observed flux without continuum division.
Active coefficient bounds and earlier fit flags remain recorded. A small
positive continuum can still make normalized residuals unusable at faint edges.
In this run, 153 order-frame fits in other orders have at least one active
coefficient bound; four continuum samples reach the numerical floor. The
endpoint normalization failure in stored order 39 is repaired. Its remaining
fixed-scale residual RMS is about 1.92%; stellar and telluric line mismatches
have not been eliminated by changing the continuum.
The fixed-scale product remains finite at every original sample.
A complete export is not an injection-recovery test.
Optimizer termination also differs from strict repeated-refit stabilization:
the recorded all-order run passed the optimizer checks but did not meet that
stricter threshold. Per-order flags are in `allorders_summary.json`.

For all 44 orders and 79 frames, fixed-scale Q5 RMS changes from **1.7181% to
1.5216%**; the last 128 pixels change from **1.3897% to 0.6184%**. These are
percentages of each frame's p95 flux, not percentages of fitted continuum.
The remaining narrow spectral mismatches are not eliminated by this revision.

![Continuum revision at stored cube index 8](docs/figures/allorders_continuum_o08.png)

All panels above use the same observational flux scale before and after the
change. The lower panels enlarge the rightmost 128 pixels. The local report
contains all 44 orders and the corresponding per-order metrics.
The separate `allorders_continuum_refined_normalized_spectral_fits.pdf`
shows only the current fit in the original three-panel continuum-normalized
format and explicitly shows these faint-edge limitations.

Continuum fitting attenuates narrow signals, especially at boundaries. Apply
the local continuum-plus-Q5 response to CCF templates already convolved and
sampled into frame-p95 flux units:

```python
import sys
import numpy as np
sys.path.insert(0, "IGRINS")
from refine_continuum import filter_ccf_template

with np.load("IGRINS/results/allorders_ccf_continuum_refined.npz") as result:
    processed_template = filter_ccf_template(template_flux, result, order=8, frame=30)
```

The response keeps the physical model and active coefficient bounds fixed.
For the log continuum it uses the exact objective Hessian, including the
nonzero-residual term, rather than a Gauss-Newton approximation.
It has been checked against small finite-difference perturbations, but is not
an end-to-end planet injection-recovery test. It does not restore information
removed by the smooth fit. Original observations and the earlier continuum
product remain available for alternative CCF analyses.

`run_ccf_series.py --wait-for-fits 21600` can consume shared products as they
become available. Allocate separate CPU sets with `taskset` when running it
alongside the shared-fit batch. Intermediate products and all input data remain
Git-ignored under `IGRINS/`.

## Model and evaluation

- **Band:** 2.26622167–2.29458395 microns in the released archive, all 1,848 pixels.
  User Order 7 corresponds to released cube row 5, identified by wavelength.
- **Shared training frames:** 0, 19, 39, 59, 78. Comparison frames: 10, 30, 50, 70.
  Frame indices are zero-based. Evaluation-frame nuisance parameters and smooth
  residual terms are fitted conditionally; this is not a fully untouched holdout.
- **Atmosphere:** dated 20-layer T-P/H2O profile with H2O, CH4, N2O, CO, and CO2,
  pressure shifts, and a symmetric two-Gaussian instrumental profile.
- **Star:** PHOENIX anchored to a real blasé clone. Selected feature amplitudes
  vary; stellar centers, widths, and reference kinematics remain fixed.
- **Telluric strengths:** nonnegative, shared corrections with HITRAN uncertainty
  categories and a prior that suppresses unnecessary changes to weak lines.
- **Continuum:** positive cubic B-splines, with a log-continuum parameterization
  for stored index 39, in the final observed-series revision;
  the original polynomial physical fits remain the upstream calibration.
- **Smooth residual:** a degree-five polynomial per frame on the full order.
  No time-common residual template is subtracted.
- **Optimization:** bounded TRF least squares for physical and stellar parameters;
  L-BFGS-B for the final large telluric-strength fit, using JAX derivatives.
  The continuum and polynomial terms use linear least squares.

The reference all-pixel RMS was **0.820934%** of the original fitted continuum.
See [reference results and limitations](docs/REFERENCE_RESULTS.md). The strict
repeated-refit threshold was not met in that reference run, and planet-injection
recovery has not been validated for this expanded Order 7 region. This export
does not establish a validated telluric-removal pipeline for planetary spectroscopy.

## Diagnostics and optional comparison

```bash
python IGRINS/diagnose_order7_4383_profile.py
python IGRINS/diagnose_order7_4383_telluric.py
```

These local experiments diagnose overlapping stellar and CH4 absorption; they
do not replace the active fit. Earlier scripts for pressure broadening, line
shifts, T-P freedom, IP alternatives, weak-line priors, spectral masks,
signal-injection checks, and TransformerPayne are retained alongside the current
workflow. Most historical scripts expect the corresponding earlier products and
are not independent fresh-checkout entry points. Their old-order validation
results must not be transferred to Order 7.

`overlay_enuma.py` remains an optional reader for a separately supplied local
`IGRINS/series_model.npz`. Its keys are `frame`, `wave`, `flux_model`, and
`flux_telluric`. The report adds that overlay only when the file exists. The
reader documents its inferred flux normalization and optional smooth blaze
adjustment. **Neither the enuma input nor its derived comparison arrays or plots
are distributed here.**

## Checks and provenance

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
python -m compileall -q IGRINS
```

The lightweight regression checks exercise the required ExoJAX convolution
behavior, the guarded TRF SVD fallback, positive continua and their template
response. `check_order7.py` independently
verifies fitted products after a real fit. Initial-export source hashes and
the environment are recorded in
`docs/implementation_manifest.json`.

The implementation is MIT licensed. The vendored blasé JAX module retains its
own [MIT license](IGRINS/vendor/blase/LICENSE) and
[upstream provenance](IGRINS/vendor/blase/PROVENANCE.json). External observations,
line lists, and stellar models retain their respective source terms.
