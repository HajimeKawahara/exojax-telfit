"""Build explicitly empirical H2O speed-dependence sensitivity templates.

The ratios below are test values, not measured HITRAN coefficients. The mean
Voigt width and pressure shift remain unchanged; Gamma2 = ratio * collisional
Gamma0. This tests a missing degree of freedom without claiming a new line list.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")

import hapi
import numpy as np

from exojax.database.core.broadening import doppler_sigma, gamma_hitran, gamma_natural
from exojax.database.core.line_strength import line_strength
from exojax.opacity import OpaDirect
from exojax.utils.constants import Patm

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent
RATIOS = [.1,.2]


def main():
    profile = load_profile(20)
    columns = profile["number_column_cm2"]*profile["vmr"]["H2O"]
    for order in [13,34,40]:
        started = time.monotonic()
        with np.load(ROOT/"cache"/f"advanced_tau_o{order}_l20.npz") as data:
            grid = data["grid"]
        db = load_mdb("H2O",grid[0]-3,grid[-1]+3,crit=1e-28)
        tau = np.zeros((len(RATIOS),len(columns),len(grid)))
        for layer,(temperature,pressure,column) in enumerate(zip(profile["temperature_K"],profile["pressure_bar"],columns)):
            strength = np.asarray(line_strength(temperature,db.logsij0,db.nu_lines,db.elower,
                                                db.qr_interp_lines(temperature,296.),296.))
            gd = np.asarray(doppler_sigma(db.nu_lines,temperature,db.molmass))*np.sqrt(2*np.log(2))
            collisional = np.asarray(gamma_hitran(pressure,temperature,0.,db.n_air,db.gamma_air,db.gamma_self))
            gamma0 = collisional+np.asarray(gamma_natural(db.A))
            delta0 = np.asarray(db.delta_air)*pressure/Patm
            for j,center in enumerate(db.nu_lines):
                for k,ratio in enumerate(RATIOS):
                    tau[k,layer] += column*strength[j]*hapi.PROFILE_SDVOIGT(center,gd[j],gamma0[j],
                                                                         ratio*collisional[j],delta0[j],0.,grid)
        baseline,_ = profile_tau(OpaDirect(db,grid),grid,profile,"H2O",True,False)
        result = dict(order=order,species="H2O",ratios=RATIOS,layers=20,n_lines=len(db.nu_lines),
                      line_strength_cutoff=1e-28,line_center_margin_cm1=3.,
                      vertical_column_cm2=float(np.sum(columns)),pressure_shift=True,
                      reference_self_broadening=False,temperature_offset_K=0.,pressure_scale=1.,
                      seconds=time.monotonic()-started,
                      interpretation="Empirical sensitivity templates; not measured or inferred HITRAN Gamma2 values",
                      assumptions=["Gamma2/Gamma0 fixed at the selected ratio for all H2O transitions and layers",
                                   "Gamma0 means collisional width only; natural width is excluded from Gamma2",
                                   "Same temperature exponent for Gamma0 and Gamma2",
                                   "Voigt mean width and pressure shift are retained",
                                   "No speed-dependent shift, line mixing, or velocity-changing collisions"])
        out = ROOT/"cache"/f"water_sdv_tau_o{order}_l20"
        np.savez_compressed(out.with_suffix(".npz"),grid=grid,tau_voigt=baseline,
                            **{f"tau_ratio_{ratio:g}":tau[k].sum(axis=0) for k,ratio in enumerate(RATIOS)},
                            **{f"layer_tau_ratio_{ratio:g}":tau[k] for k,ratio in enumerate(RATIOS)},
                            layer_column_cm2=columns,pressure_bar=profile["pressure_bar"],
                            temperature_K=profile["temperature_K"],metadata_json=json.dumps(result))
        out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
        print(json.dumps(result),flush=True)


if __name__ == "__main__":
    main()
