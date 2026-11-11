"""Bound omitted N2-air continuum absorption in the fitted K-band order."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from atmosphere_profile import load_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    path = ROOT / "cache/N2-air_2018.cia"
    lines = path.read_text().splitlines()
    rows, temperatures = [], []
    k = 0
    while k < len(lines):
        header = lines[k].split()
        n = int(header[3])
        if float(header[1]) == 4300. and float(header[2]) == 5000.:
            values = np.asarray([list(map(float, line.split())) for line in lines[k+1:k+n+1]])
            rows.append(values[:, 1]); temperatures.append(float(header[4]))
            nu = values[:, 0]
        k += n+1
    coeff = np.asarray(rows)
    temperatures = np.asarray(temperatures)
    assert coeff.shape == (14, 701)
    profile = load_profile(20)
    assert np.min(profile["temperature_K"]) >= temperatures[0]
    assert np.max(profile["temperature_K"]) <= temperatures[-1]
    # The rounded RFM N2/O2 entries are 789000/212000 ppmv. Normalize
    # their sum to unity for this approximate two-component dry mixture.
    n2_dry, o2_dry = .789/1.001, .212/1.001
    water = profile["vmr"]["H2O"]
    density = 1e6*profile["pressure_bar"]/(1.380649e-16*profile["temperature_K"])
    pair_column = (profile["number_column_cm2"]*density
                   *n2_dry*(n2_dry+o2_dry)*(1-water)**2)
    tau = np.zeros(len(nu))
    for temperature, column in zip(profile["temperature_K"], pair_column):
        upper = np.clip(np.searchsorted(temperatures, temperature), 1, len(temperatures)-1)
        fraction = (temperature-temperatures[upper-1])/(temperatures[upper]-temperatures[upper-1])
        tau += column*((1-fraction)*coeff[upper-1]+fraction*coeff[upper])
    with np.load(OUT/"focus_ip.npz") as saved:
        data = {key:saved[key] for key in saved.files}
    info = pd.read_csv(ROOT/"data/20201214_info.csv")
    b = int(np.flatnonzero(data["orders"] == 13)[0])
    wavelength = data["wavelength_um"][b]
    tau_sample = np.interp(1e4/wavelength, nu, tau)
    basis = np.polynomial.chebyshev.chebvander(np.linspace(-1, 1, len(wavelength)), 3)
    summaries = []
    for e, exposure in enumerate(data["exposures"]):
        airmass = float(info.iloc[exposure]["Air Mass"])
        transmission = np.exp(-airmass*tau_sample)
        # Smooth CIA is evaluated at data samples; this diagnostic does not
        # claim an exact re-convolution of the resolved line spectrum.
        new_component = data["component"][b,e]*transmission
        use = data["train_mask"][b,e] | data["calibration_mask"][b,e]
        design = basis*new_component[:,None]
        coeffs = np.linalg.solve(design[use].T@design[use]+1e-12*np.eye(4),
                                design[use].T@data["observed"][b,e,use])
        model = new_component*(basis@coeffs)
        delta = (model-data["model"][b,e])/data["continuum"][b,e]
        summaries.append(dict(exposure=int(exposure), airmass=airmass,
                              max_absorption_percent=float(100*np.max(1-transmission)),
                              min_absorption_percent=float(100*np.min(1-transmission)),
                              max_model_change_after_continuum_refit_percent=float(100*np.max(abs(delta))),
                              rms_model_change_after_continuum_refit_percent=float(100*np.sqrt(np.mean(delta**2)))))
    report = dict(
        species_pair="N2-Air", order=13, band_wavenumber_cm1=[4300.,5000.],
        units="cm5 molecule-2; multiplied by N2 number density and (N2+O2) number density",
        source="https://hitran.org/data/CIA/main/N2-air_2018.cia",
        definition_source="https://hitran.org/data/CIA/CIA_Readme_2024.pdf",
        composition_source="https://eodg.atm.ox.ac.uk/RFM/atm/ngt.atm",
        dry_N2_vmr=n2_dry, dry_O2_vmr=o2_dry, layers=20,
        temperature_interpolation="Linear in CIA coefficient, no extrapolation",
        approximations=["Smooth transmission multiplied after resolved IP convolution",
                        "Rounded RFM N2/O2 reference abundances renormalized to unity; argon omitted from this diagnostic mixture",
                        "Wavelength correction neglected for this broad continuum only",
                        "Fixed reference H2O dilution; fitted H2O multiplier not applied to dry-air density",
                        "No CIA band coverage for orders34/40 in the relevant N2/O2 Main datasets; this is not proof of zero unmeasured continuum"],
        results=summaries,
        source_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (OUT/"n2_cia_check.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT/"n2_cia_check.npz", wavelength_um=wavelength, vertical_tau=tau_sample,
                        full_wavenumber=nu, full_vertical_tau=tau)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
