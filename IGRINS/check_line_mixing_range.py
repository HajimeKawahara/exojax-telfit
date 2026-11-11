"""Check CO2 first-order line-mixing sensitivity to line-center coverage."""
from __future__ import annotations

import argparse
import hashlib
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
from exojax.utils.constants import Patm

from hitran_cache import load_mdb
from check_opacity_completeness import subset_mdb
from prepare_advanced_tau import load_advanced

ROOT = Path(__file__).resolve().parent
MARGINS = [3.,25.,50.,100.,200.]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    path = ROOT/"cache"/f"advanced_tau_o{args.order}_l20.npz"
    with np.load(path) as data:
        grid = data["grid"]
        pressure = data["pressure_bar"]
        temperature = data["temperature_K"]
        columns = data["layer_column_cm2"]
        tau = {name:np.broadcast_to(data[f"layer_tau_{source}"],(len(MARGINS),len(columns),len(grid))).copy()
               for name,source in [("voigt","hapi_voigt"),("sdv","sdv"),("mixing","sdv_mixing")]}
    db = load_mdb("CO2",grid[0]-200,grid[-1]+200,crit=1e-28)
    db = subset_mdb(db,(db.nu_lines < grid[0]-3)|(db.nu_lines > grid[-1]+3))
    advanced,csv_path = load_advanced(args.order,db,suffix="_wide")
    distance = np.maximum(grid[0]-db.nu_lines,db.nu_lines-grid[-1])
    natural = np.asarray(gamma_natural(db.A))
    print(f"Order {args.order}: {len(db.nu_lines)} additional lines",flush=True)
    for layer,(temp,pres,column) in enumerate(zip(temperature,pressure,columns)):
        strength = np.asarray(line_strength(temp,db.logsij0,db.nu_lines,db.elower,
                                            db.qr_interp_lines(temp,296.),296.))
        gd = np.asarray(doppler_sigma(db.nu_lines,temp,db.molmass))*np.sqrt(2*np.log(2))
        gamma_v = np.asarray(gamma_hitran(pres,temp,0.,db.n_air,db.gamma_air,db.gamma_self))+natural
        delta_v = np.asarray(db.delta_air)*pres/Patm
        gamma0 = advanced["gamma_SDV_0_air_296"]*(296/temp)**advanced["n_SDV_air_296"]*pres/Patm+natural
        gamma2 = advanced["gamma_SDV_2_air_296"]*(296/temp)**advanced["n_gamma_SDV_2_air_296"]*pres/Patm
        delta0 = (advanced["delta_SDV_0_air_296"]+np.nan_to_num(advanced["deltap_SDV_air_296"])*(temp-296))*pres/Patm
        mixing = advanced["Y_SDV_air_296"]*(296/temp)**advanced["n_Y_SDV_air_296"]*pres/Patm
        for j,center in enumerate(db.nu_lines):
            weight = column*strength[j]
            profile_v = weight*hapi.PROFILE_VOIGT(center,gd[j],gamma_v[j],delta_v[j],grid)
            real,imag = hapi.pcqsdhc(center,gd[j],gamma0[j],gamma2[j],delta0[j],0.,0.,0.,grid)
            profile_s = weight*real
            profile_m = weight*(real+mixing[j]*imag)
            indices = np.flatnonzero(np.asarray(MARGINS) >= distance[j])
            tau["voigt"][indices,layer] += profile_v
            tau["sdv"][indices,layer] += profile_s
            tau["mixing"][indices,layer] += profile_m
        print(f"Order {args.order}: layer {layer+1}/20, {time.monotonic()-started:.1f}s",flush=True)
    sums = {name:np.sum(values,axis=1) for name,values in tau.items()}
    result = dict(order=args.order,species="CO2",layers=20,additional_line_count=len(db.nu_lines),
                  line_strength_cutoff=1e-28,margins_cm1=MARGINS,seconds=time.monotonic()-started,
                  numerical_profile="Exact HAPI Voigt and qSDVoigt complex profiles for every retained line and layer",
                  minimum_tau={name:np.min(value,axis=1).tolist() for name,value in sums.items()},
                  max_tau_difference_from_200={name:np.max(abs(value-value[-1]),axis=1).tolist() for name,value in sums.items()},
                  assumptions=["No extrapolation beyond the downloaded line list",
                               "No negative-opacity clipping",
                               "First-order Rosenkranz mixing only; not full relaxation-matrix line mixing",
                               "All other settings match the fixed 20-layer focus_ip atmosphere"],
                  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in [Path(__file__),path,csv_path]})
    out = ROOT/"cache"/f"advanced_tau_o{args.order}_l20_wide"
    np.savez_compressed(out.with_suffix(".npz"),grid=grid,margins=MARGINS,
                        **{f"tau_{name}_m{margin:g}":value[k] for name,value in sums.items() for k,margin in enumerate(MARGINS)},
                        **{f"layer_tau_{name}_m{margin:g}":value[k] for name,value in tau.items() for k,margin in enumerate(MARGINS)},
                        pressure_bar=pressure,temperature_K=temperature,layer_column_cm2=columns,
                        metadata_json=json.dumps(result))
    out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
