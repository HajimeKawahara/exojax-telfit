"""Compare documented CO2 SDV variants at fixed atmospheric/IP parameters."""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from exojax.opacity import OpaDirect
from exojax.postproc.response import ipgauss, sampling
from exojax.postproc.spin_rotation import convolve_rigid_rotation
from exojax.utils.constants import c

from atmosphere_profile import load_profile
from fit_joint import profile_tau
from hitran_cache import load_mdb
from stellar_template import load_stellar

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
VARIANTS = ["voigt", "hapi_voigt", "sdv_widths", "sdv_width_shift", "sdv", "sdv_mixing"]


def rms(values):
    return float(np.sqrt(np.mean(np.asarray(values)**2)))


def main():
    meta = json.loads((OUT/"focus_ip.json").read_text())
    p = meta["parameters"]
    with np.load(OUT/"focus_ip.npz") as saved:
        data = {key:saved[key] for key in saved.files}
    info = pd.read_csv(ROOT/"data/20201214_info.csv")
    profile = load_profile(meta["layers"])
    component = {name:np.zeros_like(data["component"]) for name in VARIANTS}
    for b,order in enumerate(data["orders"]):
        with np.load(ROOT/"cache"/f"advanced_tau_o{order}_l20.npz") as saved:
            grid = saved["grid"]
            co2_tau = {name:saved[f"tau_{name}"] for name in VARIANTS}
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv),int(60/dv)+1)*dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)),velocity,4.,.6,0.)
        star = ipgauss(star,velocity,p["stellar_extra_sigma_kms"])
        db = load_mdb("H2O",grid[0]-3,grid[-1]+3,crit=1e-28)
        water_tau = profile_tau(OpaDirect(db,grid),grid,profile,"H2O",True,False)[0]
        x = np.linspace(-1,1,len(data["wavelength_um"][b]))
        nu = 1e4/data["wavelength_um"][b]*(1+p[f"stretch_o{order}_kms"]*x/c)
        for e,exposure in enumerate(data["exposures"]):
            am = float(info.iloc[exposure]["Air Mass"])
            relative_rv = float(info.iloc[exposure]["RV [km/s]"]-info.iloc[data["exposures"][0]]["RV [km/s]"])
            shifted = sampling(grid,grid,star,p["stellar_rv_reference_kms"]+relative_rv)
            water = 10**p[f"log10_water_e{exposure}"]
            co2 = 10**p["log10_scale_CO2"]
            beta = p[f"ip_sigma_o{order}_kms"]
            wing = p["ip_wing_fraction"]
            rv = p[f"calibration_rv_o{order}_kms"]+p.get(f"calibration_rv_e{exposure}_kms",0.)
            for name in VARIANTS:
                intrinsic = shifted*jnp.exp(-am*(water*water_tau+co2*co2_tau[name]))
                convolved = (1-wing)*ipgauss(intrinsic,velocity,beta)+wing*ipgauss(intrinsic,velocity,2*beta)
                component[name][b,e] = np.asarray(sampling(nu,grid,convolved,rv))
    error = float(np.max(abs(component["voigt"]-data["component"])))
    np.testing.assert_allclose(component["voigt"],data["component"],rtol=2e-12,atol=2e-12)
    basis = np.polynomial.chebyshev.chebvander(np.linspace(-1,1,data["component"].shape[-1]),3)
    records = {}
    predictions = {}
    use = data["train_mask"] | data["calibration_mask"]
    selected = np.broadcast_to(data["included_mask"][:,None,:],data["component"].shape)
    for name in VARIANTS:
        fixed = component[name]*data["continuum"]
        projected = np.zeros_like(fixed)
        for b in range(len(data["orders"])):
            for e in range(len(data["exposures"])):
                design = component[name][b,e,:,None]*basis
                weight = use[b,e]
                coeffs = np.linalg.solve(design[weight].T@design[weight]+1e-12*np.eye(4),
                                        design[weight].T@data["observed"][b,e,weight])
                projected[b,e] = component[name][b,e]*(basis@coeffs)
        predictions[name] = projected
        difference = component[name]-component["voigt"]
        records[name] = dict(max_model_difference_percent=float(100*np.max(abs(difference[selected]))),
                             rms_model_difference_percent=100*rms(difference[selected]),
                             train_rms_fixed_continuum_percent=100*rms((data["observed"]-fixed)[data["train_mask"]]),
                             score_rms_fixed_continuum_percent=100*rms((data["observed"]-fixed)[data["score_mask"]]),
                             train_rms_refit_continuum_percent=100*rms((data["observed"]-projected)[data["train_mask"]]),
                             score_rms_refit_continuum_percent=100*rms((data["observed"]-projected)[data["score_mask"]]),
                             per_order=[dict(order=int(order),
                                             score_rms_refit_continuum_percent=100*rms((data["observed"]-projected)[b][data["score_mask"][b]]))
                                        for b,order in enumerate(data["orders"])])
    report = dict(scope="Fixed focus_ip physical parameters; only continuum can be reprojected",
                  reconstruction_max_error=error,line_center_margin_cm1=3.,
                  mixing_note="Preliminary: distant-branch mixing convergence not yet tested; do not treat this as final mixed-line opacity",
                  variants=records)
    (OUT/"advanced_opacity_comparison.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(OUT/"advanced_opacity_comparison.npz",orders=data["orders"],exposures=data["exposures"],
                        wavelength_um=data["wavelength_um"],
                        **{f"component_{name}":value for name,value in component.items()},
                        **{f"model_{name}":value for name,value in predictions.items()})
    fig,axes = plt.subplots(3,1,figsize=(14,9),constrained_layout=True)
    for b,(order,ax) in enumerate(zip(data["orders"],axes)):
        for name in ["sdv_widths","sdv_width_shift","sdv","sdv_mixing"]:
            ax.plot(data["wavelength_um"][b],100*(component[name][b,1]-component["voigt"][b,1]),lw=.9,label=name)
        ax.set(title=f"Order {order}; exposure78; fixed atmosphere and IP",ylabel="Model change (%)")
        ax.grid(alpha=.2)
    axes[0].legend()
    axes[-1].set_xlabel("Observed vacuum wavelength (micron; assumed convention)")
    fig.savefig(OUT/"advanced_opacity_comparison.png",dpi=150)
    plt.close(fig)
    print(json.dumps(records,indent=2),flush=True)


if __name__ == "__main__":
    main()
