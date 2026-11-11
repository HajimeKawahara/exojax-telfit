"""Compare ExoJAX cross sections against the independent HAPI Voigt kernel."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import hapi
import jax
import numpy as np

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.utils.constants import Patm

from fit_joint import cross_section
from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent


def main():
    data = np.load(ROOT / "results/focus_ip.npz")
    cases = []
    for order, species, molecule in [(13, "CO2", 2), (34, "CO2", 2), (40, "H2O", 1)]:
        b = int(np.flatnonzero(data["orders"] == order)[0])
        nu = 1e4/data["wavelength_um"][b]
        grid = np.linspace(nu.min(), nu.max(), 1500)
        db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
        opa = OpaDirect(db, grid)
        exo = jax.jit(lambda t,p:cross_section(opa,t,p,0.,True))
        for temperature, pressure in [(296., .735), (260.5, .3), (220.25, .03)]:
            actual = np.asarray(exo(temperature, pressure))
            q = {int(iso): (hapi.partitionSum(molecule,int(iso),temperature),
                            hapi.partitionSum(molecule,int(iso),296.))
                 for iso in np.unique(db.isoid)}
            hapi_xs = np.zeros(len(grid))
            shared_kernel = np.zeros(len(grid))
            strength_exo = np.asarray(line_strength(temperature,db.logsij0,db.nu_lines,db.elower,
                                                    db.qr_interp_lines(temperature,296.),296.))
            width_exo = np.asarray(gamma_hitran(pressure,temperature,0.,db.n_air,db.gamma_air,db.gamma_self)
                                   + gamma_natural(db.A))
            sigma_exo = np.asarray(doppler_sigma(db.nu_lines,temperature,db.molmass))
            for j in range(len(db.nu_lines)):
                iso = int(db.isoid[j])
                center = float(db.nu_lines[j])
                strength = hapi.EnvironmentDependency_Intensity(float(db.Sij0[j]),temperature,296.,
                                                                q[iso][0],q[iso][1],float(db.elower[j]),center)
                gamma = hapi.EnvironmentDependency_Gamma0(float(db.gamma_air[j]),temperature,296.,
                                                          pressure/Patm,1.,float(db.n_air[j]))
                gamma += float(db.A[j])/(4*np.pi*2.99792458e10)
                trans = dict(molec_id=molecule,local_iso_id=iso,nu=center,T=temperature,p=pressure/Patm,
                             Diluent={"air":1.})
                doppler = float(hapi.calculate_parameter_GammaD(None,trans))
                shift = float(db.delta_air[j])*pressure/Patm
                hapi_xs += hapi.PROFILE_VOIGT(center,doppler,gamma,shift,grid,Sw=strength)
                shared_kernel += hapi.PROFILE_VOIGT(center,sigma_exo[j]*np.sqrt(2*np.log(2)),
                                                   width_exo[j],shift,grid,Sw=strength_exo[j])
            norm = np.linalg.norm(actual)
            record = dict(order=order,species=species,n_lines=len(db.nu_lines),temperature_K=temperature,
                          pressure_bar=pressure,
                          independent_physics_relative_l2=float(np.linalg.norm(hapi_xs-actual)/norm),
                          independent_physics_peak_normalized_max=float(np.max(abs(hapi_xs-actual))/np.max(actual)),
                          shared_parameters_kernel_relative_l2=float(np.linalg.norm(shared_kernel-actual)/norm),
                          shared_parameters_kernel_peak_normalized_max=float(np.max(abs(shared_kernel-actual))/np.max(actual)))
            cases.append(record)
            print(json.dumps(record),flush=True)
    report = dict(hapi_version=hapi.HAPI_VERSION, cases=cases,
                  scope="Same cached isotope-resolved HITRAN lines; no line-by-line wing truncation; air widths and pressure shift",
                  shared_parameters="ExoJAX S(T), Doppler sigma, Lorentz width passed to HAPI profile with sigma-to-HWHM conversion",
                  independent_physics="HAPI intensity, TIPS partitionSum, Doppler calculation, pressure broadening; natural width A/(4*pi*c) added",
                  known_constant_difference="HAPI1.2.2.3 intensity uses c2=1.4388028496642257; ExoJAX uses its own current c2. ExoJAX cached Q(T) is linearly interpolated on 1K grid.",
                  max_independent_relative_l2=max(x["independent_physics_relative_l2"] for x in cases),
                  max_shared_kernel_relative_l2=max(x["shared_parameters_kernel_relative_l2"] for x in cases))
    assert report["max_independent_relative_l2"] < 1e-3
    assert report["max_shared_kernel_relative_l2"] < 1e-4
    (ROOT/"results/hapi_voigt_check.json").write_text(json.dumps(report,indent=2)+"\n")


if __name__ == "__main__":
    main()
