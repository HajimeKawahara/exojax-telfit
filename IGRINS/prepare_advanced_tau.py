"""Compute CO2 SDVoigt and first-order line mixing on the exact fit grid.

HITRAN SDV broadening, pressure-shift and mixing parameters are separated so
their effects are not conflated. Abundances, atmospheric layers and IP are not
fitted here. Arrays can replace the fixed CO2 vertical tau in fit_joint.
"""
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
from exojax.opacity import OpaDirect
from exojax.utils.constants import c, Patm

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb

ROOT = Path(__file__).resolve().parent
VARIANTS = ["hapi_voigt", "sdv_widths", "sdv_width_shift", "sdv", "sdv_mixing"]


def load_advanced(order, db, suffix=""):
    path = ROOT / "cache" / f"advanced_CO2_o{order}{suffix}.csv"
    parameters = json.loads(path.with_suffix(".json").read_text())["parameters"]
    mapping = {}
    for line in path.read_text().splitlines():
        fields = line.split(",")
        raw = fields[0]
        iso = int(raw[2]) if raw[2].isdigit() and raw[2] != "0" else 10 if raw[2] == "0" else ord(raw[2])-ord("A")+11
        key = (iso, float(raw[3:15]), float(raw[15:25]))
        mapping[key] = {name: (float(value) if value != "#" else np.nan)
                        for name,value in zip(parameters[1:],fields[1:])}
    rows = [mapping[(int(iso),float(nu),float(sw))]
            for iso,nu,sw in zip(db.isoid,db.nu_lines,db.Sij0)]
    result = {name:np.asarray([row[name] for row in rows]) for name in parameters[1:]}
    for name in ["gamma_SDV_0_air_296", "n_SDV_air_296", "gamma_SDV_2_air_296",
                 "n_gamma_SDV_2_air_296", "delta_SDV_0_air_296", "Y_SDV_air_296", "n_Y_SDV_air_296"]:
        if not np.all(np.isfinite(result[name])):
            raise ValueError(f"Missing required SDV coefficient: {name}")
    return result, path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    meta = json.loads((ROOT/"results/focus_ip.json").read_text())
    with np.load(ROOT/"results/focus_ip.npz") as data:
        b = int(np.flatnonzero(data["orders"] == args.order)[0])
        nu = 1e4/data["wavelength_um"][b]
    minimum,maximum = nu[0]*np.exp(-220/c),nu[-1]*np.exp(220/c)
    n = int(np.ceil(np.log(maximum/minimum)*c/meta["dv_kms"]))+1
    n += n % 2
    grid = np.geomspace(minimum,maximum,n)
    db = load_mdb("CO2", grid[0]-3,grid[-1]+3,crit=1e-28)
    advanced, csv_path = load_advanced(args.order, db)
    profile = load_profile(meta["layers"])
    columns = profile["number_column_cm2"]*profile["vmr"]["CO2"]
    tau = {name:np.zeros((len(columns),len(grid))) for name in VARIANTS}
    for layer,(temperature,pressure,column) in enumerate(zip(profile["temperature_K"],profile["pressure_bar"],columns)):
        strength = np.asarray(line_strength(temperature,db.logsij0,db.nu_lines,db.elower,
                                            db.qr_interp_lines(temperature,296.),296.))
        gd = np.asarray(doppler_sigma(db.nu_lines,temperature,db.molmass))*np.sqrt(2*np.log(2))
        natural = np.asarray(gamma_natural(db.A))
        gamma_v = np.asarray(gamma_hitran(pressure,temperature,0.,db.n_air,db.gamma_air,db.gamma_self))+natural
        delta_v = np.asarray(db.delta_air)*pressure/Patm
        gamma0 = advanced["gamma_SDV_0_air_296"]*(296/temperature)**advanced["n_SDV_air_296"]*pressure/Patm+natural
        gamma2 = advanced["gamma_SDV_2_air_296"]*(296/temperature)**advanced["n_gamma_SDV_2_air_296"]*pressure/Patm
        deltap = np.nan_to_num(advanced["deltap_SDV_air_296"],nan=0.)
        delta0 = (advanced["delta_SDV_0_air_296"]+deltap*(temperature-296))*pressure/Patm
        mixing = advanced["Y_SDV_air_296"]*(296/temperature)**advanced["n_Y_SDV_air_296"]*pressure/Patm
        for j, center in enumerate(db.nu_lines):
            weighted = column*strength[j]
            tau["hapi_voigt"][layer] += weighted*hapi.PROFILE_VOIGT(center,gd[j],gamma_v[j],delta_v[j],grid)
            tau["sdv_widths"][layer] += weighted*hapi.PROFILE_VOIGT(center,gd[j],gamma0[j],delta_v[j],grid)
            tau["sdv_width_shift"][layer] += weighted*hapi.PROFILE_VOIGT(center,gd[j],gamma0[j],delta0[j],grid)
            real,imag = hapi.pcqsdhc(center,gd[j],gamma0[j],gamma2[j],delta0[j],0.,0.,0.,grid)
            tau["sdv"][layer] += weighted*real
            tau["sdv_mixing"][layer] += weighted*(real+mixing[j]*imag)
        print(f"Order {args.order}: layer {layer+1}/{len(columns)}, {time.monotonic()-started:.1f}s",flush=True)
    exo_tau,_ = profile_tau(OpaDirect(db,grid),grid,profile,"CO2",True,False)
    summed = {name:np.sum(values,axis=0) for name,values in tau.items()}
    relative = float(np.linalg.norm(summed["hapi_voigt"]-exo_tau)/np.linalg.norm(exo_tau))
    assert relative < 1e-4
    if np.any(summed["sdv_mixing"] < 0):
        print("WARNING: Negative line-mixed total tau; inspect line-range convergence",flush=True)
    result = dict(order=args.order,species="CO2",layers=meta["layers"],n_lines=len(db.nu_lines),
                  line_strength_cutoff=1e-28,line_center_margin_cm1=3.,
                  vertical_column_cm2=float(np.sum(columns)),pressure_shift=True,
                  reference_self_broadening=False,temperature_offset_K=0.,pressure_scale=1.,
                  hapi_voigt_relative_l2=relative,seconds=time.monotonic()-started,
                  minimum_tau={name:float(value.min()) for name,value in summed.items()},
                  variants={"voigt":"Existing ExoJAX Voigt, standard .par gamma_air/n_air/delta_air",
                            "hapi_voigt":"Independent HAPI kernel, otherwise same ExoJAX physical parameters",
                            "sdv_widths":"HITRAN SDV gamma0 and n0, no speed dependence; standard pressure shift",
                            "sdv_width_shift":"SDV gamma0/n0 and SDV pressure shift, no speed dependence",
                            "sdv":"SDV gamma0/gamma2/n0/n2 and pressure shift, no line mixing",
                            "sdv_mixing":"SDV plus HITRAN first-order Y coefficient and its temperature exponent"},
                  assumptions=["HITRAN SDV deltap missing: set temperature derivative of pressure shift to zero",
                               "No speed dependence of line shift or velocity-changing collisions",
                               "Self widths omitted consistently with focus_ip",
                               "Line-mixing center-margin convergence must be checked before physical interpretation"],
                  source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in [Path(__file__),csv_path,ROOT/"cache/CO2.par",ROOT/"cache/atmosphere_profile.npz"]})
    out = ROOT/"cache"/f"advanced_tau_o{args.order}_l{meta['layers']}"
    np.savez_compressed(out.with_suffix(".npz"),grid=grid,tau_voigt=exo_tau,
                        **{f"tau_{name}":value for name,value in summed.items()},
                        **{f"layer_tau_{name}":value for name,value in tau.items()},
                        pressure_bar=profile["pressure_bar"],temperature_K=profile["temperature_K"],
                        layer_column_cm2=columns,metadata_json=json.dumps(result))
    out.with_suffix(".json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)


if __name__ == "__main__":
    main()
