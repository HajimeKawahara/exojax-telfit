"""Locate persistent residuals and reconstruct physical component contributions.

This is a post-fit diagnostic of the three fitted orders/exposures, not a
new fit or an independent validation. All thresholds are descriptive.
"""
from __future__ import annotations

import csv
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
from matplotlib.ticker import FormatStrFormatter

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


def regions_from_residual(residual):
    """Connect coherent 3% excursions, retaining peaks above 5%."""
    median = np.median(residual, axis=0)
    indices = np.flatnonzero(np.abs(median) > .03)
    if not len(indices):
        return []
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 3)+1)
    intervals = [(max(0, int(g[0])-2), min(len(median), int(g[-1])+3))
                 for g in groups if np.max(np.abs(median[g])) >= .05]
    merged = []
    for a,b in intervals:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a,b))
    return merged


def main():
    meta = json.loads((OUT / "joint_star_profile.json").read_text())
    with np.load(OUT / "joint_star_profile.npz") as source:
        data = {key: source[key] for key in source.files}
    p = meta["parameters"]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    profile = load_profile(meta["layers"])
    orders, exposures = data["orders"], data["exposures"]
    shape = data["component"].shape
    components = {name: np.empty(shape) for name in
                  ["full", "stellar_only", "telluric_only", "without_H2O", "without_CO2"]}
    rest_wavelength = np.empty(shape)
    for b, order in enumerate(orders):
        wave = data["wavelength_um"][b]
        nu = 1e4/wave
        minimum, maximum = nu[0]*np.exp(-220/c), nu[-1]*np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/meta["dv_kms"]))+1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c*np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1)*dv
        stellar = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        stellar = ipgauss(stellar, velocity, p["stellar_extra_sigma_kms"])
        tau = {}
        for species in ["H2O", "CO2"]:
            db = load_mdb(species, grid[0]-3, grid[-1]+3, crit=1e-28)
            opa = OpaDirect(db, grid)
            tau[species] = profile_tau(opa, grid, profile, species,
                                      meta["pressure_shift"], meta["reference_self_broadening"])[0]
        for e, exposure in enumerate(exposures):
            airmass = float(info.iloc[exposure]["Air Mass"])
            stellar_rv = p["stellar_rv_reference_kms"] + float(info.iloc[exposure]["RV [km/s]"] - info.iloc[exposures[0]]["RV [km/s]"])
            star = sampling(grid, grid, stellar, stellar_rv)
            water = jnp.exp(-airmass * 10**p[f"log10_water_e{exposure}"] * tau["H2O"])
            co2 = jnp.exp(-airmass * 10**p["log10_scale_CO2"] * tau["CO2"])
            x = np.linspace(-1, 1, len(wave))
            calibrated_nu = nu*(1+p[f"stretch_o{order}_kms"]*x/c)
            rv = p[f"calibration_rv_o{order}_kms"] + p.get(f"calibration_rv_e{exposure}_kms", 0.)
            beta = p[f"ip_sigma_o{order}_kms"]
            spectra = dict(full=star*water*co2, stellar_only=star, telluric_only=water*co2,
                           without_H2O=star*co2, without_CO2=star*water)
            for name, spectrum in spectra.items():
                components[name][b,e] = np.asarray(sampling(calibrated_nu, grid,
                                                           ipgauss(spectrum, velocity, beta), rv))
            rest_wavelength[b,e] = 1e4/(calibrated_nu*(1+rv/c)*(1+stellar_rv/c))
    reconstruction_error = float(np.max(np.abs(components["full"]-data["component"])))
    np.testing.assert_allclose(components["full"], data["component"], rtol=1e-12, atol=1e-12)
    components["stellar_marginal_absorption"] = components["telluric_only"]-components["full"]
    components["telluric_marginal_absorption"] = components["stellar_only"]-components["full"]
    components["H2O_marginal_absorption"] = components["without_H2O"]-components["full"]
    components["CO2_marginal_absorption"] = components["without_CO2"]-components["full"]
    np.savez_compressed(OUT / "residual_components.npz", **components,
                        stellar_rest_wavelength_um=rest_wavelength,
                        wavelength_um=data["wavelength_um"], orders=orders, exposures=exposures)
    residual = data["residual_continuum"]
    fixed = data["observed"]-data["model"]
    total_sse = float(np.sum(fixed**2))
    validation_sse = float(np.sum(fixed[data["score_mask"]]**2))
    summaries, regions = [], []
    for b,order in enumerate(orders):
        summaries.append(dict(order=int(order), fixed_sse_percent=100*float(np.sum(fixed[b]**2))/total_sse,
                              validation_sse_percent=100*float(np.sum(fixed[b][data["score_mask"][b]]**2))/validation_sse,
                              residual_correlation=np.corrcoef(residual[b]).tolist()))
        median = np.median(residual[b], axis=0)
        for a,z in regions_from_residual(residual[b]):
            k = int(a+np.argmax(np.abs(median[a:z])))
            star_depth = float(np.median(components["stellar_marginal_absorption"][b,:,k]))
            tell_depth = float(np.median(components["telluric_marginal_absorption"][b,:,k]))
            species = {s: float(np.median(components[f"{s}_marginal_absorption"][b,:,k])) for s in ["H2O", "CO2"]}
            if max(star_depth, tell_depth) < .02:
                attribution = "Weak modeled absorption / continuum or missing feature"
            elif star_depth > 3*max(tell_depth, .001):
                attribution = "Stellar-dominated feature"
            elif tell_depth > 3*max(star_depth, .001):
                attribution = max(species, key=species.get) + "-dominated telluric feature"
            else:
                attribution = "Stellar/telluric blend"
            score = data["score_mask"][b,:,a:z]
            entry = dict(order=int(order), array_interval=[a,z],
                         pixel_inclusive=[int(meta["pixels"][1]-z), int(meta["pixels"][1]-1-a)],
                         wavelength_min_um=float(np.min(data["wavelength_um"][b,a:z])),
                         wavelength_max_um=float(np.max(data["wavelength_um"][b,a:z])),
                         peak_wavelength_um=float(data["wavelength_um"][b,k]), peak_index=k,
                         stellar_rest_peak_um_e0=float(rest_wavelength[b,0,k]),
                         peak_residual_percent=(100*residual[b,:,k]).tolist(),
                         median_peak_percent=float(100*median[k]),
                         fixed_sse_percent=100*float(np.sum(fixed[b,:,a:z]**2))/total_sse,
                         validation_sse_percent=100*float(np.sum(fixed[b,:,a:z][score]**2))/validation_sse,
                         peak_in_validation=bool(data["score_mask"][b,-1,k]),
                         stellar_marginal_depth_percent=100*star_depth,
                         telluric_marginal_depth_percent=100*tell_depth,
                         species_marginal_depth_percent={s:100*v for s,v in species.items()},
                         attribution=attribution)
            regions.append(entry)
    regions.sort(key=lambda row: row["fixed_sse_percent"], reverse=True)
    for rank, row in enumerate(regions, 1):
        row["rank"] = rank
    result = dict(scope="Orders13,34,40; pixels600:1500; exposures0,78,39; final saved joint profile model",
                  sign="(data-model)/fitted continuum; positive means model absorption too deep",
                  selection="abs(median signed residual)>3%; bridge <=2-pixel gaps; peak>=5%; pad2pixels; merge overlaps",
                  selection_note="Post-hoc descriptive selection, not a new independent validation",
                  attribution_note="Component strength at residual peak identifies overlap, not a unique physical cause or elemental line identification",
                  reconstruction_max_error=reconstruction_error,
                  orders=summaries, regions=regions,
                  region_pixel_count=sum(z-a for row in regions for a,z in [row["array_interval"]]),
                  total_wavelength_pixels=int(shape[0]*shape[-1]),
                  regions_sse_percent=sum(row["fixed_sse_percent"] for row in regions),
                  top6_sse_percent=sum(row["fixed_sse_percent"] for row in regions[:6]))
    comparisons = {}
    for name in ["joint_star_slab", "joint_star_profile_reference_self"]:
        with np.load(OUT / f"{name}.npz") as other:
            difference = (other["model"]-data["model"])/data["continuum"]
        comparisons[name] = {
            "rms_model_difference_by_order_percent": (100*np.sqrt(np.mean(difference**2,axis=(1,2)))).tolist(),
            "max_model_difference_by_order_percent": (100*np.max(np.abs(difference),axis=(1,2))).tolist(),
        }
    result["refitted_model_comparisons"] = comparisons
    (OUT / "residual_diagnostics.json").write_text(json.dumps(result, indent=2)+"\n")
    columns = ["rank", "order", "wavelength_min_um", "wavelength_max_um", "peak_wavelength_um",
               "median_peak_percent", "fixed_sse_percent", "validation_sse_percent", "peak_in_validation",
               "stellar_marginal_depth_percent", "telluric_marginal_depth_percent", "attribution"]
    with (OUT / "residual_regions.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader(); writer.writerows(regions)
    report = f"""IGRINS residual localization: final PHOENIX + 20-layer atmosphere

Scope: orders13,34,40, original pixels600:1500, exposures0/78/39 only.
Wavelengths below are observed-grid vacuum microns (not stellar rest wavelengths).
Residual = (data-model)/fitted continuum. Positive: model flux too low;
negative: model flux too high. Percentages are continuum percentage points.
The model is unchanged. Component reconstruction matches the saved full model
to maximum absolute error {reconstruction_error:.3g}.

Selection is exploratory: absolute median signed residual across3exposures
above3%, connect gaps<=2pixels, retain peaks>=5%, add2pixel margins and merge.
Rank by residual squared sum in the original fixed observed-flux normalization.
{len(regions)} regions use{result['region_pixel_count']}/{result['total_wavelength_pixels']} wavelength pixels
and contain{result['regions_sse_percent']:.2f}% of total fixed-scale squared residual.
The top6 contain{result['top6_sse_percent']:.2f}%. These are not detection significances.

Rank Order Peak_um    Median(%)  SSE_share(%)  e0/e78/e39_residual(%)    Component overlap
"""
    for row in regions:
        residual_text = "/".join(f"{v:+.2f}" for v in row["peak_residual_percent"])
        report += (f"{row['rank']:>3} {row['order']:>4} {row['peak_wavelength_um']:.8f} "
                   f"{row['median_peak_percent']:+8.2f} {row['fixed_sse_percent']:12.3f} "
                   f"{residual_text:25s} {row['attribution']}\n")
    report += """
Component interpretation
------------------------
Separate star-only and telluric-only curves are diagnostic counterfactuals.
The actual model always multiplies intrinsic star and transmission BEFORE IP.
Marginal stellar absorption = model without stellar lines - full model;
marginal telluric absorption = model without tellurics - full model.
The component label uses their median depths at the residual peak: >3:1
dominance, with a2% minimum relevant absorption; otherwise blend or weakmodel.
This identifies overlap with modeled absorption, not a unique cause/element.

The top5 regions overlap predominantly stellar features, with modeled telluric
contributions below1% at their peaks. At1.498690 and1.501472um PHOENIX predicts
substantially deeper lines than observed. At1.577624 and1.577960um observed
absorption is stronger. The1.579461um region has opposite-sign lobes and may
include a line-position/blend mismatch, beyond a simple depth error.
At1.573526um the observations have an extra absorption feature: both the
current stellar and H2O/CO2 predictions are weak there. Its physical carrier
has not been identified. Do not label it a confirmed stellar/atomic line.
Both over- and under-predicted features remain, so a single global line-depth
scaling cannot remove them all. Candidate issues are template line strengths,
blend positions, stellar parameters and local normalization; no unique cause
is inferred from this diagnostic alone.

Repeated structure
------------------
Residual correlations across the3exposures:
"""
    for summary in summaries:
        corr = np.asarray(summary["residual_correlation"])[np.triu_indices(3,1)]
        report += f"Order{summary['order']}: {min(corr):.3f}-{max(corr):.3f}; total SSEshare {summary['fixed_sse_percent']:.2f}%; validation SSEshare {summary['validation_sse_percent']:.2f}%\n"
    report += """
These repeated patterns support a systematic mismatch rather than isolated
noise spikes. The barycentric span is small, so temporal persistence alone
does not distinguish a stellar feature from a stable telluric error.
The archived reference-self model differs by much less than the largest
10-13% residuals; see JSON for exact refitted-model differences.

Validation caveat
-----------------
The maximum1.498690um peak and four other top6regions are outside the
previously defined validation score mask. They are still real full-window
model mismatches, but they do not contribute to that heldout score.
The largest scored-region contributions are:
"""
    for row in sorted(regions,key=lambda row:row["validation_sse_percent"],reverse=True)[:6]:
        report += f"Order{row['order']} {row['peak_wavelength_um']:.8f}um: {row['median_peak_percent']:+.2f}%, {row['validation_sse_percent']:.2f}% of validation SSE\n"
    report += """
Artifacts / reproduction
------------------------
python IGRINS/diagnose_residuals.py
residual_overview.png: all3exposures, numbered selected regions.
residual_hotspots.png: top6full-window regions, model component spectra.
residual_validation_hotspots.png: top6validation-score regions.
residual_regions.csv: ranked wavelength intervals and component overlaps.
residual_diagnostics.json: full numerical results and scope.
residual_components.npz: reconstructed counterfactual spectra and stellar rest
wavelength mapping. The original fit and observational data were not edited.
"""
    missing_path = OUT / "missing_feature_check.json"
    if missing_path.exists():
        report += """
Additional check of the unmodeled1.57352647um absorption
-----------------------------------------------------
check_missing_feature.py uses the same dated20layer profile and fittedIP,
with much weaker cached lines retained (S296>=1e-30). At the residual peak,
nominal CH4/N2O/CO together absorb0.003695%; maximum within10km/s is0.004231%.
Even10times their nominal columns yields a maximum0.04230% within10km/s.
Those three omitted molecular species do not explain the~11% absorption.
This is not an exhaustive search for all possible species or data artifacts;
the feature's physical carrier remains unidentified.
Reproduce: python IGRINS/check_missing_feature.py
Numerical details: missing_feature_check.json.
"""
    (OUT / "RESIDUAL_REPORT.txt").write_text(report)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    for b,(order,ax) in enumerate(zip(orders,axes)):
        wave = data["wavelength_um"][b]
        for e,exposure in enumerate(exposures):
            ax.plot(wave, 100*residual[b,e], lw=.65, alpha=.65, label=f"Exposure {exposure}")
        ax.plot(wave, 100*np.median(residual[b], axis=0), color="black", lw=.9, label="Median")
        for row in regions:
            if row["order"] != order:
                continue
            ax.axvspan(row["wavelength_min_um"], row["wavelength_max_um"], color="tab:red", alpha=.08)
            if row["rank"] <= 12:
                ax.annotate(str(row["rank"]), (row["peak_wavelength_um"], row["median_peak_percent"]),
                            xytext=(0, 9 if row["median_peak_percent"]>0 else -13), textcoords="offset points",
                            ha="center", fontsize=9, color="tab:red")
        ax.axhline(0, color="0.4", lw=.6)
        ax.set_title(f"Order {order}; fixed-scale residual SSE share {summaries[b]['fixed_sse_percent']:.1f}%")
        ax.set_ylabel("(Data - model) / continuum (%)")
        ax.set_xlabel("Observed vacuum wavelength (micron)")
        ax.set_ylim(-15,16)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[0].legend(ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "residual_overview.png", dpi=170); plt.close(fig)

    def zoom_plot(selected, filename):
        fig = plt.figure(figsize=(16, 10))
        outer = fig.add_gridspec(2, 3, wspace=.25, hspace=.36)
        for position, row in enumerate(selected):
            b = list(orders).index(row["order"])
            a,z = row["array_interval"]; a=max(0,a-9); z=min(shape[-1],z+9)
            sl = slice(a,z); wave = data["wavelength_um"][b,sl]
            sub = outer[position//3,position%3].subgridspec(2,1,height_ratios=[3,2],hspace=.07)
            top = fig.add_subplot(sub[0]); bottom=fig.add_subplot(sub[1],sharex=top)
            top.plot(wave, (data["observed"]/data["continuum"])[b,0,sl], color="0.25", lw=1., label="Data e0 / continuum")
            top.plot(wave, components["full"][b,0,sl], color="tab:orange", lw=1., label="Full model")
            top.plot(wave, components["stellar_only"][b,0,sl], color="tab:purple", ls="--", lw=.85, label="Star only, with IP")
            top.plot(wave, components["telluric_only"][b,0,sl], color="tab:cyan", ls=":", lw=1., label="Telluric only, with IP")
            label = row["attribution"]
            if label.startswith("Weak modeled"):
                label = "Unmodeled absorption; carrier unidentified"
            top.set_title(f"#{row['rank']} | Order {row['order']} | {row['peak_wavelength_um']:.6f} micron\n{label}", fontsize=10)
            top.set_ylabel("Normalized flux"); top.tick_params(labelbottom=False)
            for e,exposure in enumerate(exposures):
                bottom.plot(wave, 100*residual[b,e,sl], lw=.9, label=f"e{exposure}")
            bottom.axhline(0, color="0.4", lw=.6)
            bottom.axvspan(row["wavelength_min_um"],row["wavelength_max_um"],color="tab:red",alpha=.07)
            bottom.set_ylabel("Residual (%)")
            bottom.set_xlabel("Observed vacuum wavelength (micron)")
            bottom.xaxis.set_major_formatter(FormatStrFormatter("%.6f"))
            bottom.tick_params(axis="x",labelrotation=20,labelsize=8)
            if position==0:
                top.legend(fontsize=7); bottom.legend(fontsize=7,ncol=3)
        fig.savefig(OUT / filename, dpi=170, bbox_inches="tight"); plt.close(fig)
    zoom_plot(regions[:6], "residual_hotspots.png")
    zoom_plot(sorted(regions,key=lambda row:row["validation_sse_percent"],reverse=True)[:6], "residual_validation_hotspots.png")
    print(json.dumps({k:result[k] for k in ["reconstruction_max_error","region_pixel_count","regions_sse_percent","top6_sse_percent"]},indent=2))
    for row in regions[:16]:
        print(row["rank"],row["order"],round(row["peak_wavelength_um"],8),round(row["median_peak_percent"],2),
              row["attribution"],"star/tell depth",round(row["stellar_marginal_depth_percent"],2),round(row["telluric_marginal_depth_percent"],2))


if __name__ == "__main__":
    main()
