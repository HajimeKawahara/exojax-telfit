"""Cache a small dated NOAA atmosphere subset and interpret IGRINS times."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
import xarray as xr
from astropy import units as u
from astropy.coordinates import EarthLocation, SkyCoord
from astropy.time import Time
from astropy.utils import iers

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "cache"
RESULTS = ROOT / "results"
SITE = {"latitude_deg": -30.2407056, "longitude_deg": -70.7367028,
        "altitude_m": 2722.0}
SITE_REFERENCE = "https://www.ctio.noirlab.edu/noao/sites/default/files/telescopes/Mamajek12_positions.pdf"
TARGET_REFERENCE = "https://exoplanetarchive.ipac.caltech.edu/overview/WASP-77%20A"
BASE = "https://psl.noaa.gov/thredds/ncss/grid/Datasets/ncep.reanalysis/pressure/"


def time_metadata():
    """Invert assumed BJD_TDB, retaining the uncertainty in that convention."""
    iers.conf.auto_download = False
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    location = EarthLocation.from_geodetic(
        SITE["longitude_deg"] * u.deg, SITE["latitude_deg"] * u.deg,
        SITE["altitude_m"] * u.m,
    )
    target = SkyCoord("02h28m37.33s", "-07d03m38.40s")
    bjd = Time(info["Time (BJD)"].to_numpy(), format="jd", scale="tdb", location=location)
    observed = bjd.copy()
    for _ in range(4):
        observed = bjd - observed.light_travel_time(target, kind="barycentric")
    correction = target.radial_velocity_correction(obstime=observed).to_value(u.km/u.s)
    light_time = observed.light_travel_time(target, kind="barycentric").sec
    table = info.copy()
    table.insert(0, "exposure_index", np.arange(len(table)))
    table["UTC_assuming_BJD_TDB"] = observed.utc.isot
    table["barycentric_light_time_s"] = light_time
    table["astropy_barycentric_correction_kms"] = correction
    table["CSV_RV_plus_barycentric_correction_kms"] = info["RV [km/s]"] + correction
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / "time_metadata.csv", index=False)
    metadata = {
        "input_time_label": "Time (BJD)",
        "assumption": "BJD_TDB; timescale and exposure-time convention absent from CSV",
        "conversion": "Four iterations subtracting Astropy barycentric light-travel time; built-in ephemeris and bundled IERS data",
        "utc_start_assumed": observed.utc.isot[0],
        "utc_end_assumed": observed.utc.isot[-1],
        "utc_median_assumed": observed.utc.isot[len(info) // 2],
        "site": SITE,
        "site_reference": SITE_REFERENCE,
        "target_icrs": {"ra_deg": target.ra.deg, "dec_deg": target.dec.deg},
        "target_reference": TARGET_REFERENCE,
        "CSV_RV_plus_correction_range_kms": [float(np.min(info["RV [km/s]"] + correction)),
                                               float(np.max(info["RV [km/s]"] + correction))],
        "RV_interpretation": "CSV RV is consistent with observer-frame stellar velocity: systemic (~1.6845 km/s) minus Astropy barycentric correction. Use positive CSV relative changes for stellar Doppler shifts.",
        "time_precision_note": "Displayed milliseconds are computational precision, not established timing accuracy. BJD_UTC instead of BJD_TDB changes inferred UTC by about69s, negligible relative to6h meteorological sampling.",
    }
    (RESULTS / "time_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return observed.utc, metadata


def download(variable: str, start: str, end: str):
    """Download only two times and the four surrounding reanalysis grid cells."""
    query = urlencode({"var": variable, "north": -30.0, "south": -32.5,
                       "west": -72.5, "east": -70.0,
                       "time_start": start, "time_end": end, "accept": "netcdf3"})
    url = BASE + variable + ".2020.nc?" + query
    path = CACHE / f"atmosphere_{variable}.nc"
    provenance_path = CACHE / f"atmosphere_{variable}.json"
    if not path.exists():
        with urlopen(url, timeout=90) as response:
            raw = response.read()
        # Validate before retaining a cache; the NCSS endpoint can return text errors.
        if raw[:3] != b"CDF":
            raise ValueError(f"Unexpected NOAA response: {raw[:200]!r}")
        path.write_bytes(raw)
    raw = path.read_bytes()
    if not provenance_path.exists():
        provenance = {"url": url, "cached_utc": datetime.now(timezone.utc).isoformat(),
                      "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                      "dataset": "NOAA PSL NCEP/NCAR Reanalysis 1; 6-hourly, 2.5 degree grid"}
        provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    with xr.open_dataset(path, engine="scipy") as dataset:
        return dataset.load(), json.loads(provenance_path.read_text())


def main():
    CACHE.mkdir(exist_ok=True)
    utc, metadata = time_metadata()
    start = "2020-12-14T00:00:00Z"
    end = "2020-12-14T06:00:00Z"
    midpoint = np.datetime64(utc.isot[len(utc) // 2])
    profiles, sources = {}, {}
    for variable in ["air", "hgt", "shum"]:
        dataset, source = download(variable, start, end)
        # Linear spatial and temporal interpolation of the four surrounding cells.
        selected = dataset[variable].interp(
            lat=SITE["latitude_deg"], lon=SITE["longitude_deg"] % 360,
            time=midpoint,
        )
        profiles[variable] = (selected.level.values / 1000., selected.values)
        sources[variable] = source
        print(variable, profiles[variable], flush=True)
    pressure, temperature = profiles["air"]
    height_pressure, height = profiles["hgt"]
    surface_pressure = float(np.exp(np.interp(SITE["altitude_m"], height,
                                              np.log(height_pressure))))
    humidity_pressure, specific_humidity = profiles["shum"]
    # Specific humidity is kg water / kg moist air; convert to mole fraction.
    epsilon = 18.01528 / 28.9647
    water_vmr = (specific_humidity / epsilon
                 / (1 - specific_humidity + specific_humidity / epsilon))
    if np.any(~np.isfinite(temperature)) or np.any(~np.isfinite(water_vmr)):
        raise ValueError("Non-finite interpolated reanalysis profile")
    metadata.update({
        "profile_kind": "dated_reanalysis_with_explicit_upper_atmosphere_extension",
        "dataset": "NCEP/NCAR Reanalysis 1, NOAA PSL",
        "dataset_reference": "https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html",
        "meteorology_utc_bracket": [start, end],
        "profile_utc_assumed": str(midpoint),
        "horizontal_interpolation": "Bilinear: lat[-32.5,-30.0], lon[-72.5,-70.0] degrees",
        "temporal_interpolation": "Linear at median observation time, fixed for this first nightly model",
        "surface_pressure_bar": surface_pressure,
        "surface_pressure_method": "Interpolate log pressure against reanalysis geopotential height at telescope altitude; not an on-site pressure measurement",
        "temperature_upper_extension": "Reanalysis ends at10hPa. Interpolate in log pressure from actual10hPa T to270.65K at1hPa (standard-atmosphere-inspired upper boundary). Atmosphere above1hPa omitted.",
        "water_upper_extension": "Reanalysis specific humidity ends at300hPa. Log-interpolate water VMR to3ppm at100hPa and keep3ppm above; assumed stratospheric water, not measured.",
        "reference_dry_vmr": {"CO2": 410e-6, "CH4": 1.9e-6, "N2O": .33e-6, "CO": .1e-6},
        "composition_note": "Dry-species reference VMRs are illustrative 2020 abundances, not obtained from weather data. Each may be fitted with a multiplier. Small dry-to-moist dilution is handled in load_profile.",
        "mean_dry_air_molecular_weight": 28.9647,
        "gravity_cgs": 978.0,
        "limitations": ["Coarse reanalysis does not resolve mountain-site boundary layer",
                        "A single interpolated nightly T/P/H2O shape; fit H2O amplitude per exposure",
                        "CSV humidity is not treated as PWV or used to override vertical humidity",
                        "Timescale and midpoint convention are inferred, not explicitly identified in archive"],
        "sources": sources,
    })
    np.savez_compressed(CACHE / "atmosphere_profile.npz",
                        pressure_bar=pressure, temperature_K=temperature,
                        height_m=height, humidity_pressure_bar=humidity_pressure,
                        water_vmr=water_vmr, surface_pressure_bar=surface_pressure)
    (CACHE / "atmosphere_profile.json").write_text(json.dumps(metadata, indent=2) + "\n")
    # Keep a compact diagnostic of the actual input and discretized model.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from atmosphere_profile import load_profile

    layered = load_profile(20)
    fig, axes = plt.subplots(1, 2, figsize=(9, 5), sharey=True)
    axes[0].plot(temperature, pressure, "o-", ms=3, label="Dated reanalysis")
    axes[0].plot(layered["temperature_K"], layered["pressure_bar"], ".--", label="20-layer model")
    axes[0].set_xlabel("Temperature (K)")
    axes[0].set_ylabel("Pressure (bar)")
    axes[1].semilogx(water_vmr, humidity_pressure, "o-", ms=3, label="Dated reanalysis")
    axes[1].semilogx(layered["vmr"]["H2O"], layered["pressure_bar"], ".--", label="20-layer model")
    axes[1].set_xlabel("Water mole fraction")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_ylim(1.05, .0008)
        axis.axhline(surface_pressure, color="0.3", ls=":", label="Telescope altitude")
        axis.legend(fontsize=8)
    fig.suptitle("Gemini South, 2020-12-14 03:02 UTC (inferred)\nNCEP/NCAR reanalysis; upper extensions are assumptions")
    fig.tight_layout()
    fig.savefig(RESULTS / "atmosphere_profile.png", dpi=160)
    plt.close(fig)
    print(json.dumps({k: metadata[k] for k in ["utc_start_assumed", "utc_end_assumed", "surface_pressure_bar", "profile_utc_assumed"]}, indent=2))


if __name__ == "__main__":
    main()
