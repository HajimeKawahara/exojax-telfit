# External data and caches

Run the commands below from the repository root, after installing the main
environment described in [README.md](../README.md). Observational data, generated
results, line lists, stellar templates, and external model checkpoints stay under
the ignored `IGRINS/data/`, `IGRINS/cache/`, and `IGRINS/results/` directories.
The supplied enuma data are not distributed with this repository.

## Observations

The current analysis uses the 2020-12-14 WASP-77A observations from
[Zenodo record 10382053](https://zenodo.org/records/10382053)
(DOI: [10.5281/zenodo.10382053](https://doi.org/10.5281/zenodo.10382053)).
Only these two files are needed for the current Order 7 analysis:

| File | Bytes | Published MD5 |
| --- | ---: | --- |
| `cube14v3.pic` | 77,591,894 | `3fd41560b7dbcc60851ef3a0972c852f` |
| `20201214_info.csv` | 7,261 | `7c861b1baf8ae87801f95d7af8b2e9c6` |

```bash
python IGRINS/prepare_data.py
python IGRINS/prepare_data.py --verify-only
python IGRINS/inspect_data.py
```

The downloader streams the public files, verifies their recorded size and
SHA256, and replaces a destination only after verification. Existing valid files
are skipped. It can also verify a separately downloaded copy:

```bash
python IGRINS/prepare_data.py --data-dir /path/to/data --verify-only
```

The fitting scripts themselves read `IGRINS/data/`. Direct upstream downloads
are available for
[the cube](https://zenodo.org/records/10382053/files/cube14v3.pic?download=1) and
[the metadata](https://zenodo.org/records/10382053/files/20201214_info.csv?download=1).
The SHA256 checksums used by the downloader are:

```text
999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26  cube14v3.pic
68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e  20201214_info.csv
```

The pickle contains `[wavelength, flux]` with shapes `(44, 1848)` and
`(44, 79, 1848)`, respectively. Wavelength is interpreted in microns; the stored
flux has no explicit physical unit. Load only the verified source file, because
pickle is an executable serialization format.

User Order 7 maps by wavelength to released-cube **zero-based index 5**. The
stored interval is **2.2662216709–2.2945839498 microns**, with all 1,848 pixels
retained. The requested original range extends beyond these trimmed data;
missing edge pixels are not reconstructed. Shared fits use frames
`0, 19, 39, 59, 78`, and the four evaluation frames are `10, 30, 50, 70`.
These are zero-based indices, not the first five consecutive exposures.

The archive does not explicitly label every wavelength/time convention used by
the analysis. In particular, `prepare_atmosphere.py` assumes the CSV's `Time (BJD)`
column is BJD_TDB. Its output records this assumption and the resulting UTC times.

## HITRAN line lists

```bash
python IGRINS/prepare_hitran.py
```

This downloads H2O, CO2, CH4, N2O and CO from the
[HITRANonline API](https://hitran.org/lbl/api), covering **4050–6900 cm⁻¹**.
It preserves the raw 160-character records as `.par` files, parses the numerical
arrays into `.npz` files, and writes JSON provenance with retrieval URLs and
hashes. The strength uncertainty fields are read from the raw records by the
fitting helpers.

The original run used HITRAN2024 from the live endpoint and HAPI **1.2.2.3**, whose
default partition sums are TIPS-2021. The endpoint has **no version pin**: a future
download can differ from the recorded calculation. Preserve the local raw files
and provenance when exact numerical reproducibility matters, and compare their
hashes before interpreting changes in a fit. See the
[HITRAN references](https://hitran.org/docs/hitran-papers/) for the data editions.

## PHOENIX stellar template

```bash
python IGRINS/prepare_stellar.py
```

The fixed reference is PHOENIX-ACES at **5600 K**, **log g = 4.5**, and solar
metallicity. The downloader uses the upstream
[wavelength file](https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits)
and
[flux file](https://ftp.gwdg.de/pub/misc/phoenix/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/Z-0.0/lte05600-4.50-0.0.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits).
The raw wavelengths are vacuum Angstroms. The preparation retains 1.4–2.5 microns,
estimates a smooth pseudo-continuum, and preserves the native line samples.
Stellar rotation and instrumental convolution are applied later by the forward
model, not during this preparation. The cache records the source hashes and
normalization method. The library is described by
[Husser et al. (2013)](https://arxiv.org/abs/1303.5632).

The blasé clone is subsequently trained on this noiseless PHOENIX template.
That cloning step creates its own ignored cache; observational stellar-amplitude
calibration is a separate step.

## Dated atmospheric profile

```bash
python IGRINS/prepare_atmosphere.py
```

This obtains a small subset of the
[NOAA PSL NCEP/NCAR Reanalysis 1](https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html):
temperature (`air`), geopotential height (`hgt`) and specific humidity (`shum`).
It interpolates between 2020-12-14 00:00 and 06:00 UTC and the surrounding
2.5-degree grid cells to the inferred median observation time at Gemini South.
Files are read through xarray's SciPy NetCDF3 engine, so the main workflow does
not require the `netCDF4` package.

The atmosphere is a coarse reanalysis profile, not a local sounding. Explicit
upper-atmosphere temperature and water extensions, reference dry-species
abundances, site coordinates, time assumptions, and download hashes are written
to `IGRINS/cache/atmosphere_profile.json`.

## Optional historical experiments

`prepare_crossnight_data.py` retrieves the public December 6 and December 21
cubes for earlier cross-night diagnostics. Those nights and their metadata are
not required by the current Order 7 fit. Historical advanced line-shape scripts
retrieve additional live HITRAN parameters and depend on their corresponding
historical fit products.

TransformerPayne experiments used a separate environment because their JAX and
NumPy versions differed from the current ExoJAX environment. The recorded
environment used Python 3.12, `transformer-payne==0.10`, `jax==0.6.2`,
`jaxlib==0.6.2`, `flax==0.10.6`, `numpy==1.26.4`, and
`huggingface-hub==0.36.0`. For those historical experiments only:

```bash
python3.12 -m venv .venv-transformerpayne
.venv-transformerpayne/bin/python -m pip install \
  'transformer-payne==0.10' 'jax[cpu]==0.6.2' 'flax==0.10.6' \
  'numpy==1.26.4' 'huggingface-hub==0.36.0'
.venv-transformerpayne/bin/python IGRINS/prepare_transformerpayne.py --download
```

The downloader records the selected revision of
[`RozanskiT/transformer_payne`](https://huggingface.co/RozanskiT/transformer_payne).
The original trial used `TransformerPayneIntensities_v2.pkl` at revision
`b5b971f4f6cf003987b75c53fcbc8b009fd48195`. Its tested wavelength coverage stops
at 2.0 microns, so it is **not used for the current Order 7 band**. Grid preparation
and fitting scripts for the historical orders also require their earlier result
files; downloading the checkpoint alone does not reconstruct those experiments.
The environment and model checkpoint are not committed.

## Attribution and reproducibility limits

The data providers retain their respective data terms; this repository's code
license does not relicense the retrieved observations, line lists, weather data,
or stellar models. These assets are fetched from their upstream sources rather
than redistributed here.

The included upstream blasé JAX module is MIT licensed, copyright (c) 2020 gully,
at commit `6f6817b0c4095c104a10b2940375fc1e16cb80ca`. Preserve its
[LICENSE](../IGRINS/vendor/blase/LICENSE) and
[PROVENANCE.json](../IGRINS/vendor/blase/PROVENANCE.json). The module is loaded
directly; PyTorch is not needed for the current adapter.

Exact fit reproduction also depends on the ExoJAX commit and local patch, solver
settings, starting parameters, and generated numerical caches. Follow the main
README for the exported pipeline. Live upstream downloads can change, and the
published numerical result is a conditional fit rather than a demonstrated
planet-signal recovery result for this order.
