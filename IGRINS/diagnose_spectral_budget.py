"""Partition current residual power by modeled absorption and fixed intervals.

All categories and intervals use the five training exposures only. Categories
identify wavelength overlap, not mutually independent physical causes.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOURCE = OUT / "motion_clean_control"
LABELS = ["H2O-dominant", "CO2-dominant", "Mixed H2O/CO2", "Weak line / continuum"]
PLAN = dict(
    training_exposures=[0, 19, 39, 59, 78],
    molecular_absorption_minimum=0.02,
    dominance_ratio=3.0,
    region_mean_absolute_threshold=0.02,
    region_peak_absolute_threshold=0.03,
    region_bridge_gap_pixels=2,
    region_padding_pixels=2,
    rule="Use maximum marginal molecular absorption over training exposures. "
         "Below 2% for both species is weak line / continuum; otherwise a "
         "3:1 ratio defines dominance. Remaining pixels are mixed. Select "
         "intervals on mean signed training continuum residual only. Never "
         "bridge excluded pixels. Rank intervals by training fixed-scale SSE.",
    clipping="None; every included scoring pixel is retained.",
    interpretation="Wavelength categories are disjoint; they are not causal "
                   "variance fractions. Weak line includes weak modeled absorption.",
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_npz(path):
    with np.load(path) as source:
        return {key: source[key] for key in source.files}


def reconstruct(data, meta):
    """Reproduce the current full model and its molecular counterfactuals."""
    cache = OUT / "nonstellar_components.npz"
    source_hash = digest(SOURCE.with_suffix(".npz"))
    if cache.exists():
        components = load_npz(cache)
        if str(components.get("source_sha256", "")) == source_hash:
            return components
    import jax.numpy as jnp
    import pandas as pd
    from exojax.opacity import OpaDirect
    from exojax.postproc.response import ipgauss, sampling
    from exojax.postproc.spin_rotation import convolve_rigid_rotation
    from exojax.utils.constants import c
    from atmosphere_profile import load_profile
    from fit_joint import profile_tau
    from hitran_cache import load_mdb
    from stellar_template import load_stellar

    p = meta["parameters"]
    info = pd.read_csv(ROOT / "data/20201214_info.csv")
    profile = load_profile(meta["layers"])
    components = {name: np.empty(data["component"].shape) for name in
                  ["full", "stellar_only", "telluric_only", "without_H2O", "without_CO2"]}
    for b, order in enumerate(data["orders"]):
        wave = data["wavelength_um"][b]
        nu = 1e4 / wave
        minimum, maximum = nu[0] * np.exp(-220/c), nu[-1] * np.exp(220/c)
        n = int(np.ceil(np.log(maximum/minimum)*c/meta["dv_kms"])) + 1
        n += n % 2
        grid = np.geomspace(minimum, maximum, n)
        dv = float(c * np.log(grid[1]/grid[0]))
        velocity = jnp.arange(-int(60/dv), int(60/dv)+1) * dv
        star = convolve_rigid_rotation(jnp.asarray(load_stellar(grid)), velocity, 4., .6, 0.)
        star = ipgauss(star, velocity, p["stellar_extra_sigma_kms"])
        tau = {}
        for species in ["H2O", "CO2"]:
            margin = meta["line_margin_cm1"]
            db = load_mdb(species, grid[0]-margin, grid[-1]+margin, crit=meta["line_cutoff"])
            opa = OpaDirect(db, grid)
            tau[species] = profile_tau(opa, grid, profile, species,
                                      meta["pressure_shift"], meta["reference_self_broadening"])[0]
        for e, exposure in enumerate(data["exposures"]):
            airmass = float(info.iloc[exposure]["Air Mass"])
            stellar_rv = p["stellar_rv_reference_kms"] + float(
                info.iloc[exposure]["RV [km/s]"] - info.iloc[0]["RV [km/s]"])
            shifted_star = sampling(grid, grid, star, stellar_rv)
            water = jnp.exp(-airmass * 10**p[f"log10_water_e{exposure}"] * tau["H2O"])
            co2 = jnp.exp(-airmass * 10**p["log10_scale_CO2"] * tau["CO2"])
            x = np.linspace(-1, 1, len(wave))
            target = nu * (1 + p[f"stretch_o{order}_kms"] * x/c)
            rv = p[f"calibration_rv_o{order}_kms"] + p.get(f"calibration_rv_e{exposure}_kms", 0.)
            width = p[f"ip_sigma_o{order}_kms"] * np.exp(p.get(f"log_ip_width_e{exposure}", 0.))
            fraction = p["ip_wing_fraction"]
            spectra = dict(full=shifted_star*water*co2, stellar_only=shifted_star,
                           telluric_only=water*co2, without_H2O=shifted_star*co2,
                           without_CO2=shifted_star*water)
            for name, spectrum in spectra.items():
                core = sampling(target, grid, ipgauss(spectrum, velocity, width), rv)
                wing = sampling(target, grid, ipgauss(spectrum, velocity, 2*width), rv)
                components[name][b, e] = np.asarray((1-fraction)*core + fraction*wing)
        print(f"Reconstructed order {order}", flush=True)
    np.testing.assert_allclose(components["full"], data["component"], rtol=1e-11, atol=1e-12)
    components["source_sha256"] = np.asarray(source_hash)
    np.savez_compressed(cache, **components)
    return components


def intervals(residual, allowed):
    """Select coherent intervals using training residuals without final data."""
    selected = np.flatnonzero(allowed & (np.abs(residual) > PLAN["region_mean_absolute_threshold"]))
    groups = []
    for pixel in selected:
        if (not groups or pixel-groups[-1][-1] > PLAN["region_bridge_gap_pixels"]+1
                or not np.all(allowed[groups[-1][-1]:pixel+1])):
            groups.append([])
        groups[-1].append(int(pixel))
    output = []
    for group in groups:
        if np.max(np.abs(residual[group])) < PLAN["region_peak_absolute_threshold"]:
            continue
        a, z = group[0], group[-1]+1
        for _ in range(PLAN["region_padding_pixels"]):
            if a > 0 and allowed[a-1]:
                a -= 1
            if z < len(allowed) and allowed[z]:
                z += 1
        if output and a <= output[-1][1]:
            output[-1] = (output[-1][0], max(z, output[-1][1]))
        else:
            output.append((a, z))
    return output


def metrics(fixed, continuum, mask, total_sse):
    values = fixed[mask]
    normalized = continuum[mask]
    sse = float(np.sum(values**2))
    return dict(n_samples=int(len(values)), fixed_sse=sse,
                fixed_sse_percent=100*sse/total_sse if total_sse else None,
                fixed_rms_percent=float(100*np.sqrt(np.mean(values**2))) if len(values) else None,
                continuum_rms_percent=float(100*np.sqrt(np.mean(normalized**2))) if len(values) else None,
                fixed_bias_percent=float(100*np.mean(values)) if len(values) else None,
                continuum_bias_percent=float(100*np.mean(normalized)) if len(values) else None)


def make_scope(data, mask_name):
    return dict(fixed=data["observed"]-data["model"], continuum=data["residual_continuum"],
                mask=data[mask_name])


def main():
    # Write the declared protocol before any final prediction is read.
    plan_path = OUT / "nonstellar_spectral_plan.json"
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == PLAN
    else:
        plan_path.write_text(json.dumps(PLAN, indent=2) + "\n")
    data = load_npz(SOURCE.with_suffix(".npz"))
    meta = json.loads(SOURCE.with_suffix(".json").read_text())
    components = reconstruct(data, meta)
    np.testing.assert_allclose(components["full"], data["component"], rtol=1e-11, atol=1e-12)
    allowed = data["included_mask"]
    training = np.flatnonzero(np.isin(data["exposures"], PLAN["training_exposures"]))
    assert data["exposures"][training].tolist() == PLAN["training_exposures"]
    h2o = np.max((components["without_H2O"]-components["full"])[:, training], axis=1)
    co2 = np.max((components["without_CO2"]-components["full"])[:, training], axis=1)
    stellar = np.max((components["telluric_only"]-components["full"])[:, training], axis=1)
    category = np.full(allowed.shape, 2, dtype=int)
    category[h2o >= PLAN["dominance_ratio"]*co2] = 0
    category[co2 >= PLAN["dominance_ratio"]*h2o] = 1
    category[np.maximum(h2o, co2) < PLAN["molecular_absorption_minimum"]] = 3
    category[~allowed] = -1
    assert np.all(np.sum(np.stack([category == i for i in range(4)]), axis=0) == allowed)
    train_mean = np.mean(data["residual_continuum"][:, training], axis=1)
    fixed_train = (data["observed"]-data["model"])[:, training]
    regions = []
    selected_mask = np.zeros_like(allowed)
    for b, order in enumerate(data["orders"]):
        for a, z in intervals(train_mean[b], allowed[b]):
            peak = int(a + np.argmax(np.abs(train_mean[b, a:z])))
            assert not np.any(selected_mask[b, a:z]) and np.all(allowed[b, a:z])
            selected_mask[b, a:z] = True
            regions.append(dict(order=int(order), order_index=b, array_interval=[a, z],
                                pixel_inclusive=[1499-z+1, 1499-a], n_wavelength_pixels=z-a,
                                wavelength_min_um=float(data["wavelength_um"][b, z-1]),
                                wavelength_max_um=float(data["wavelength_um"][b, a]),
                                peak_wavelength_um=float(data["wavelength_um"][b, peak]),
                                peak_mean_continuum_percent=float(100*train_mean[b, peak]),
                                peak_category=LABELS[category[b, peak]],
                                peak_max_h2o_depth_percent=float(100*h2o[b, peak]),
                                peak_max_co2_depth_percent=float(100*co2[b, peak]),
                                peak_max_stellar_depth_percent=float(100*stellar[b, peak]),
                                training_sse=float(np.sum(fixed_train[b, :, a:z]**2))))
    regions.sort(key=lambda row: row["training_sse"], reverse=True)
    for rank, row in enumerate(regions, 1):
        row["rank"] = rank
    frozen = dict(plan=PLAN, source_sha256=digest(SOURCE.with_suffix(".npz")),
                  category_sha256=hashlib.sha256(category.tobytes()).hexdigest(), regions=regions)
    frozen_path = OUT / "nonstellar_spectral_selection.json"
    if frozen_path.exists():
        assert json.loads(frozen_path.read_text()) == frozen
    else:
        frozen_path.write_text(json.dumps(frozen, indent=2) + "\n")
    np.savez_compressed(OUT / "nonstellar_spectral_selection.npz", category=category,
                        region_mask=selected_mask, h2o_max_training_absorption=h2o,
                        co2_max_training_absorption=co2, stellar_max_training_absorption=stellar,
                        wavelength_um=data["wavelength_um"], orders=data["orders"])

    scopes = dict(training=make_scope(data, "train_mask"), development_e20=make_scope(data, "score_mask"))
    # Final data is first accessed after interval definitions and ranking are fixed.
    final_paths = [OUT / f"nonstellar_control_e{e}.npz" for e in [10, 30, 50, 70]]
    final_data = []
    if all(path.exists() for path in final_paths):
        for exposure, path in zip([10, 30, 50, 70], final_paths):
            prediction = load_npz(path)
            np.testing.assert_array_equal(prediction["included_mask"], allowed)
            np.testing.assert_array_equal(prediction["wavelength_um"], data["wavelength_um"])
            assert int(prediction["exposures"][-1]) == exposure
            scope = make_scope(prediction, "score_mask")
            scope = {key: value[:, -1:] for key, value in scope.items()}
            scopes[f"final_e{exposure}"] = scope
            final_data.append(scope)
        scopes["final_pooled"] = {key: np.concatenate([value[key] for value in final_data], axis=1)
                                  for key in final_data[0]}
    rows, totals = [], {}
    for scope_name, scope in scopes.items():
        total = float(np.sum(scope["fixed"][scope["mask"]]**2))
        totals[scope_name] = metrics(scope["fixed"], scope["continuum"], scope["mask"], total)
        for order in ["all", *map(int, data["orders"])]:
            selection = scope["mask"].copy()
            if order != "all":
                selection[data["orders"] != order] = False
            for i, label in enumerate(LABELS):
                category_mask = selection & (category[:, None, :] == i)
                rows.append(dict(scope=scope_name, order=order, category=label,
                                 n_wavelength_pixels=int(np.sum(np.any(category_mask, axis=1))),
                                 **metrics(scope["fixed"], scope["continuum"], category_mask, total)))
        same_scope = [row for row in rows if row["scope"] == scope_name and row["order"] == "all"]
        np.testing.assert_allclose(sum(row["fixed_sse"] for row in same_scope), total, rtol=1e-13)
        assert sum(row["n_samples"] for row in same_scope) == int(scope["mask"].sum())
        for row in regions:
            select = np.zeros_like(scope["mask"])
            b, (a, z) = row["order_index"], row["array_interval"]
            select[b, :, a:z] = scope["mask"][b, :, a:z]
            row[scope_name] = metrics(scope["fixed"], scope["continuum"], select, total)
        totals[scope_name]["selected_regions"] = metrics(
            scope["fixed"], scope["continuum"], scope["mask"] & selected_mask[:, None, :], total)
    output = dict(model="motion_clean_control: fixed profile, epoch-variable IP, fixed stellar parameters",
                  plan=PLAN, selection_sha256=digest(frozen_path),
                  current_component_reconstruction_max_error=float(np.max(np.abs(components["full"]-data["component"]))),
                  residual_sign="data - model; positive means model flux is too low",
                  fixed_scale_definition="Saved flux divided by each exposure/order's data-derived 95th percentile; heldout scale uses calibration pixels only",
                  continuum_definition="(observed - model) / fitted continuum",
                  included_wavelength_pixels=int(allowed.sum()),
                  category_wavelength_pixels={label: int(np.sum(category == i)) for i, label in enumerate(LABELS)},
                  totals=totals, category_metrics=rows, regions=regions,
                  caveats=["Final predictions calibrate water column, RV, IP width and continuum on disjoint calibration blocks; score blocks never enter those fits.",
                           "The stellar mask removes strong known stellar absorption, not all unknown stellar structure.",
                           "1.494855 um remains ambiguous: a stellar-motion feature fit reaches the width bound.",
                           "1.570499 um is a positive continuum/CO2-wing structure, not proof of a CO2 intensity error.",
                           "2.076826 um overlaps a CO2 feature; line shape, continuum, wavelength and IP effects remain confounded.",
                           "Category and region SSE fractions are localization statistics, not independently identified causal fractions."])
    (OUT / "nonstellar_spectral_budget.json").write_text(json.dumps(output, indent=2, allow_nan=False)+"\n")
    with (OUT / "nonstellar_spectral_budget.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    region_rows = []
    for row in regions:
        flat = {key: value for key, value in row.items() if key not in scopes}
        for name in scopes:
            flat[name+"_rms_percent"] = row[name]["fixed_rms_percent"]
            flat[name+"_sse_percent"] = row[name]["fixed_sse_percent"]
        region_rows.append(flat)
    if region_rows:
        with (OUT / "nonstellar_residual_regions.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(region_rows[0]))
            writer.writeheader()
            writer.writerows(region_rows)

    colors = ["#2468a0", "#dc8536", "#9c56a0", "#aaaaaa"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for b, (order, axis) in enumerate(zip(data["orders"], axes)):
        wave = data["wavelength_um"][b]
        for i, label in enumerate(LABELS):
            y = np.where(category[b] == i, 100*train_mean[b], np.nan)
            axis.plot(wave, y, ".", color=colors[i], ms=3.4, label=label)
        if final_data:
            final = scopes["final_pooled"]
            count = np.sum(final["mask"][b], axis=0)
            y = np.sum(np.where(final["mask"][b], final["continuum"][b], 0), axis=0)
            np.divide(y, count, out=y, where=count > 0)
            axis.plot(wave, np.where(count > 0, 100*y, np.nan), color="black", lw=.65, label="Final score mean")
        for row in regions:
            if row["order"] == order:
                axis.axvspan(row["wavelength_min_um"], row["wavelength_max_um"], color="red", alpha=.1)
                axis.text(row["peak_wavelength_um"], 5.5, str(row["rank"]), ha="center", fontsize=8)
        axis.axhline(0, color="gray", lw=.5)
        axis.set(title=f"Order {order}: training mean and final score mean", ylabel="Residual / continuum [%]",
                 xlabel="Observed vacuum wavelength [um]", ylim=(-6.5, 6.5))
    axes[0].legend(ncol=3, fontsize=8, loc="lower right")
    fig.savefig(OUT / "nonstellar_spectral_budget.png", dpi=170)
    plt.close(fig)
    report = [
        "Current nonstellar-region spectral residual budget",
        "",
        "Model: motion_clean_control, fixed 20-layer profile, mixture IP with epoch width, fixed stellar parameters.",
        "Mask: 1467 wavelength pixels in orders 13, 34, 40; original pixels 600:1500.",
        "Training exposures: 0, 19, 39, 59, 78. Development exposure: 20.",
        "Final exposures: 10, 30, 50, 70. Each contributes 523 disjoint scoring pixels.",
        "Final exposure water, RV, IP width and continuum are calibrated on separate wavelength blocks.",
        "Percent RMS uses the saved data-derived flux scale unless continuum RMS is explicitly stated.",
        "Positive residual means observed flux exceeds modeled flux. No residual clipping is applied.",
        "",
        "Scope                    Samples   Fixed RMS (%)  Continuum RMS (%)  Fixed bias (%)",
    ]
    for name in ["training", "development_e20", "final_pooled"]:
        if name in totals:
            row = totals[name]
            report.append(f"{name:24s} {row['n_samples']:7d} {row['fixed_rms_percent']:15.6f} "
                          f"{row['continuum_rms_percent']:18.6f} {row['fixed_bias_percent']:15.6f}")
    report += ["", "Disjoint wavelength categories: final pooled scoring pixels", "",
               "Category                  Wavelengths Samples RMS (%)  Bias (%) SSE share (%)"]
    for row in rows:
        if row["scope"] == "final_pooled" and row["order"] == "all":
            report.append(f"{row['category']:25s} {row['n_wavelength_pixels']:10d} {row['n_samples']:7d} "
                          f"{row['fixed_rms_percent']:8.4f} {row['fixed_bias_percent']:+9.4f} "
                          f"{row['fixed_sse_percent']:13.4f}")
    report += ["", "Categories use exact molecular counterfactuals of the current model.",
               "The reconstructed full component differs from the saved fit by at most "
               f"{output['current_component_reconstruction_max_error']:.3g} in normalized flux.",
               "The largest H2O and CO2 marginal absorptions over training exposures determine each label.",
               "Both below 2%: weak line / continuum. Otherwise, a 3:1 ratio defines dominance.",
               "The five mixed wavelength pixels yield only two scoring pixels per exposure; that RMS is unstable.",
               "These SSE percentages localize error, and do not identify the fraction caused by a molecule.",
               "", "Coherent intervals fixed using training exposures only", "",
               "Rank Order Peak (um) Mean residual (%) Pixels Final samples Final SSE share (%)"]
    for row in regions:
        final = row.get("final_pooled", {})
        report.append(f"{row['rank']:4d} {row['order']:5d} {row['peak_wavelength_um']:.9f} "
                      f"{row['peak_mean_continuum_percent']:+17.4f} {row['n_wavelength_pixels']:6d} "
                      f"{final.get('n_samples', 0):13d} {final.get('fixed_sse_percent', 0):19.4f}")
    report += ["", "Intervals: absolute mean training continuum residual >2%, peak >=3%, bridge <=2 pixels,",
               "pad two pixels, merge overlaps, and never cross a masked pixel. Rank uses training SSE.",
               "Only 15 of the selected 31 wavelength pixels occur in final scoring blocks.",
               "The 1.494845 and 1.500091 um intervals are not independently checked by the scoring blocks.",
               "The other four intervals use 60/2092 final samples and contain "
               f"{totals.get('final_pooled', {}).get('selected_regions', {}).get('fixed_sse_percent', 0):.4f}% of final SSE.",
               "", "Physical interpretation limits", "",
               *output["caveats"],
               "The 1.500091 and 1.500570 um positive features occur where modeled absorption is weak.",
               "A missing absorption line would lower the modeled flux and worsen these positive peaks.",
               "Continuum placement, residual stellar structure, IP contamination from nearby features,",
               "and reduction artifacts remain possible; this diagnostic does not distinguish them.",
               "No line list, atmospheric profile, IP, wavelength solution, or stellar model is refitted here."]
    (OUT / "NONSTELLAR_SPECTRAL_REPORT.txt").write_text("\n".join(report)+"\n")
    print(json.dumps(dict(categories=output["category_wavelength_pixels"], totals=totals,
                          regions=[{k: v for k, v in row.items() if k not in scopes} for row in regions]), indent=2))


if __name__ == "__main__":
    main()
