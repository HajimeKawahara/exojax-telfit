"""Summarize the declared telluric experiments and frozen final predictions."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
LABELS = {
    "inv_control": "Reference: fixed profile, shifted lines, Gaussian-mixture IP",
    "inv_tp1": "One free temperature offset",
    "inv_tp3": "Three free temperature nodes",
    "inv_watertilt": "One water-profile tilt",
    "inv_water2": "Two water-profile contrasts",
    "inv_tp3water2": "Three temperature nodes + two water contrasts",
    "inv_tp3epoch": "Three temperature nodes + exposure-dependent IP",
    "inv_tp3_restart": "Three temperature nodes, alternate initialization",
    "inv_wave2": "Quadratic wavelength correction",
    "inv_cont2": "Quadratic continuum",
    "inv_cont5": "Fifth-degree continuum",
    "inv_asymmetric": "Asymmetric Gaussian-mixture IP",
    "inv_direct": "Direct Gaussian sampling",
    "inv_ipwave": "Wavelength-dependent IP width, direct sampling",
    "inv_ipepoch": "Exposure-dependent IP width",
    "inv_self": "Reference self broadening",
    "inv_p097": "Pressure profile multiplied by 0.97",
    "inv_p103": "Pressure profile multiplied by 1.03",
    "inv_hapi_voigt": "Independent HAPI Voigt control",
    "inv_sdv_widths": "CO2 SDV database widths only",
    "inv_sdv_width_shift": "CO2 SDV database widths and shifts",
    "inv_sdv": "CO2 speed-dependent Voigt",
    "inv_sdv_mixing": "CO2 SDV + line mixing, narrow line margin",
    "inv_co2wide_voigt": "CO2 Voigt, 200 cm^-1 line margin in orders 13/34",
    "inv_co2wide_sdv": "CO2 SDV, 200 cm^-1 line margin in orders 13/34",
    "inv_co2wide_mixing": "CO2 SDV + first-order mixing, 200 cm^-1 line margin",
    "inv_water_sdv01": "H2O empirical SDV sensitivity: Gamma2/Gamma0 = 0.1",
    "inv_water_sdv02": "H2O empirical SDV sensitivity: Gamma2/Gamma0 = 0.2",
    "inv_robust": "Development-only soft-L1 nonlinear objective",
    "inv_robust_tp3": "Development-only soft-L1 objective + three T nodes",
}
FINAL_LABELS = {"control": "Reference", "tp3": "Free T(P)",
                "ipepoch": "Exposure-dependent IP", "tp3epoch": "Free T(P) + variable IP"}


def read_fit(tag):
    return json.loads((OUT / f"{tag}.json").read_text())


def main():
    selection = json.loads((OUT / "investigation/final_selection.json").read_text())
    exposures = selection["final_exposures"]
    screening = []
    for path in sorted(OUT.glob("inv*.json")):
        data = json.loads(path.read_text())
        if "training_fixed_scale_metrics" not in data:
            continue
        tag = path.stem
        screening.append(dict(
            tag=tag, description=LABELS.get(tag, tag),
            training_exposures=" ".join(map(str, data["train_exposures"])),
            development_exposure=data["heldout_exposure"],
            train_rms_percent=100*data["training_fixed_scale_metrics"]["rms"],
            development_rms_percent=100*data["heldout_fixed_scale_metrics"]["rms"],
            training_parameter_count=len(data["training_fit"].get("optimized_parameters", [])),
            bounds_hit=" ".join(data["bounds_hit"]),
            success=data["training_fit"]["success"] and data["heldout_calibration_fit"]["success"]))
    with (OUT / "investigation_screening.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(screening[0]))
        writer.writeheader(); writer.writerows(screening)
    with np.load(OUT / "telluric_mask.npz") as saved:
        subsets = {"all": saved["mask"], "H2O": saved["H2O_mask"], "CO2": saved["CO2_mask"]}
    rows, aggregate = [], {}
    for label in FINAL_LABELS:
        pooled = {subset: [] for subset in subsets}
        for exposure in exposures:
            tag = f"final_{label}_e{exposure}"
            data = read_fit(tag)
            if not data["training_fit"].get("reused"):
                raise ValueError(f"Final shared model was not frozen: {tag}")
            with np.load(OUT / f"{tag}.npz") as saved:
                residual = saved["observed"] - saved["model"]
                for subset, mask in subsets.items():
                    selected = saved["score_mask"] & mask[:, None, :]
                    values = residual[selected]
                    pooled[subset].append(values)
                    rows.append(dict(model=label, exposure=exposure, subset=subset,
                                     n_pixels=len(values), rms_percent=100*float(np.sqrt(np.mean(values**2)))))
        aggregate[label] = {}
        for subset, values in pooled.items():
            values = np.concatenate(values)
            aggregate[label][subset] = dict(n_pixels=len(values),
                                            rms_percent=100*float(np.sqrt(np.mean(values**2))))
    with (OUT / "investigation_final.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                   selection=selection, screening=screening, final_rows=rows,
                   final_pooled=aggregate,
                   metric="Observed p95 scaling fixed across matched models; guarded scoring blocks only",
                   sources_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in [OUT / "investigation/plan.json",
                                                OUT / "investigation/final_selection.json",
                                                OUT / "telluric_mask.npz", ROOT / "fit_joint.py",
                                                ROOT / "profile_opacity.py", Path(__file__)]})
    (OUT / "investigation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for label, title in FINAL_LABELS.items():
        values = [next(row["rms_percent"] for row in rows
                       if row["model"] == label and row["exposure"] == exposure and row["subset"] == "all")
                  for exposure in exposures]
        axes[0].plot(exposures, values, "o-", label=title)
        base = [next(row["rms_percent"] for row in rows
                     if row["model"] == "control" and row["exposure"] == exposure and row["subset"] == "all")
                for exposure in exposures]
        axes[1].plot(exposures, np.asarray(values)-base, "o-", label=title)
    axes[0].set(ylabel="Scoring RMS (%)", xlabel="Reserved exposure index")
    axes[1].set(ylabel="RMS change from reference (percentage points)", xlabel="Reserved exposure index")
    axes[1].axhline(0, color=".5", lw=.7)
    axes[0].legend(fontsize=8)
    fig.suptitle("Frozen same-night predictions | identical stellar model and pixel mask")
    fig.savefig(OUT / "investigation_final.png", dpi=160); plt.close(fig)

    lines = ["WASP-77A IGRINS telluric investigation", "",
             "Scope and decision protocol", "---------------------------",
             "All work is local under the Git-ignored IGRINS directory. Core ExoJAX code was not changed in this investigation.",
             "The stellar template, stellar RV and stellar broadening stay fixed. The same 1,477 pixels with weak predicted",
             "stellar absorption in orders 13, 34 and 40 are used throughout. Selection used earlier residuals, so wavelength",
             "selection is exploratory. Weak predicted stellar absorption does not establish that the stellar model is exact.",
             "Screening used training exposures 0/78 and development exposure 20; selected models were refit to 0/19/39/59/78.",
             "The primary free-T(P) candidate and three comparators were frozen before opening reserved exposures 10/30/50/70.",
             "Only held-out water, wavelength offset, continuum, and IP width in variable-IP variants use calibration blocks.",
             "Scoring blocks are disjoint and guarded. Shared parameters are reused exactly, not reoptimized.",
             "This validates time transfer within the same night, not a new night or an independently selected wavelength region.", "",
             "Final frozen predictions: fixed-data-scale RMS (%)", "--------------------------------------------------",
             f"{'Model':34s} {'All':>9s} {'H2O subset':>12s} {'CO2 subset':>12s}"]
    for label, title in FINAL_LABELS.items():
        row = aggregate[label]
        lines.append(f"{title:34s} {row['all']['rms_percent']:9.5f} {row['H2O']['rms_percent']:12.5f} {row['CO2']['rms_percent']:12.5f}")
    lines += ["", "Interpretation of the reserved exposures",
              "---------------------------------------",
              "Free T(P) slightly reduces the all-pixel RMS, but both fixed H2O and CO2 line subsets become slightly worse.",
              "Thus the aggregate decrease is not consistent evidence for a better physical telluric profile.",
              "Exposure-dependent IP gives modest improvement in both molecular subsets; adding T(P) freedom does not improve",
              "those subsets further. Keep the free-profile results as sensitivity tests, not a unique atmospheric retrieval."]
    lines += ["", "Temperature and exposure-dependent IP adjustments produce modest same-night prediction improvements.",
              "The combined model has the lowest pooled RMS among these frozen candidates; the remaining differences are small",
              "and do not establish a unique physical T-P profile or a decisive explanation of the residuals.", "",
              "Screening and development results (not final validation)", "-------------------------------------------------------"]
    for row in screening:
        lines.append(f"{row['tag']:25s} train={row['train_rms_percent']:.5f}% dev={row['development_rms_percent']:.5f}% "
                     f"Npar={row['training_parameter_count']} bounds={row['bounds_hit'] or 'none'}")
    if (OUT / "motion_clean_comparison.json").exists():
        followup = json.loads((OUT / "motion_clean_comparison.json").read_text())
        lines += ["", "Cross-night motion and a separate exploratory mask", "-------------------------------------------------",
                  "Public Dec06/Dec21 cubes were downloaded and checked against the Zenodo MD5 values.",
                  "Dec06 and Dec14 metadata match the cube frame counts. Dec21 has 51 cube frames but only 50 CSV rows;",
                  "its local index-offset 0/+1 cases are supplementary, not a resolved exposure-time mapping.",
                  "A weak absorption profile multiplied by telluric transmission BEFORE IP convolution was compared under",
                  "observer-stationary and CSV stellar-motion hypotheses, using the same target/anchor data and parameter counts.",
                  "At 1.501046 microns, trusted Dec06/Dec14 target RMS is 0.614% vs 0.302%, favoring stellar-frame motion.",
                  "The preference survives +/-10 K proxy-profile offsets and supplementary Dec21 mapping tests.",
                  "This supports excluding a blend from telluric evaluation; it does not identify the carrier or repair the stellar template.",
                  "The separate mask excludes 1.50098306300-1.50110573501 microns in order 40, removing 10 previously included pixels.",
                  "This interval also overlaps a positive residual: lower RMS on this mask is not evidence of atmospheric improvement.",
                  "The 1.494855-micron feature remains ambiguous because the local intrinsic width hits its upper bound.",
                  f"The follow-up mask retains {followup['retained_pixels']} pixels. The original mask and final 16 predictions are unchanged.",
                  "Both following models use variable IP, training exposures 0/19/39/59/78, and development exposure 20:"]
        for row in followup["models"]:
            lines.append(f"{row['tag']}: train {row['train_rms_percent']:.6f}%, development {row['development_rms_percent']:.6f}%, "
                         f"H2O subset {row['H2O_score_rms_percent']:.6f}%, CO2 subset {row['CO2_score_rms_percent']:.6f}%")
        lines += ["Freeing T(P) gives essentially no improvement on this motion-screened development comparison.",
                  "This is an exploratory follow-up, not a new untouched validation set. See CROSSNIGHT_REPORT.txt and motion_clean_comparison.json."]
    lines += ["", "Post-selection robust-loss diagnostic", "-------------------------------------",
              "Soft-L1 nonlinear fits with fixed scale 0.005 were tested only on development exposure 20; final selection was unchanged.",
              "No pixel was removed, and reported RMS is ordinary RMS. Continua still use masked ordinary linear least squares."]
    lines += ["", "What was tested", "---------------",
              "Temperature: one offset or three corrections at 0.1/0.35/0.7 bar, interpolated in log pressure and held constant",
              "outside the outer nodes. These are effective profile corrections; improved predictions do not establish a unique T-P retrieval.",
              "Water: one vertical tilt or two relative column weights at 0.1/0.35 bar with 0.7 bar anchored; total water column conserved.",
              "T3 converges to the same solution from a deliberately different initialization. Water-only and T3+water fits",
              "drive the upper water node to opposite bounds, so the water vertical shape is not robustly identified.",
              "Instrument/calibration: asymmetric IP, wavelength-dependent width, exposure-dependent width, direct sampling,",
              "quadratic wavelength correction, second/fifth-degree continua, reference self widths, and pressure scaling by +/-3%.",
              "Opacity: omitted standard species CH4/N2O/CO/O2/O3, no weak-line cutoff, extended far wings, and N2-air CIA.",
              "At nominal/reference abundances, the omitted standard species add at most 0.0336% absorption on included pixels;",
              "H2O/CO2 lines below the 1e-28 reference-strength cutoff add at most 0.00155%. These tested additions are too small",
              "to explain the localized 4-5% residuals. This audits lines present in the downloaded HITRAN data, not transitions",
              "absent from the database itself. Incorrect strengths, widths or other line parameters remain possible.",
              "CO2 advanced HITRAN coefficients: SDV widths, pressure shifts, quadratic speed dependence and first-order line mixing",
              "were compared with ordinary Voigt; see advanced_opacity_comparison and opacity_completeness artifacts.",
              "The main 160-character HITRAN records do not include these advanced coefficients; raw API responses are cached.",
              "CO2 line-center coverage was extended to 200 cm^-1 beyond each grid in orders 13/34, with a matched wide-Voigt control.",
              "For the mixed profiles, changing the margin from 100 to 200 cm^-1 bounds the unconvolved transmission change",
              "at the largest airmass to 0.0022% and 0.00083% in orders 13 and 34. Total optical depths remain positive.",
              "Refitting column amounts, IP and wavelength calibration with these fixed-T-P opacity templates improves training",
              "RMS for CO2 mixing but not the development score. This result concerns first-order mixing, not a full relaxation matrix.",
              "The main H2O order 40 has no retrieved air SDV/HT coefficients in the requested standard API fields. Hypothetical",
              "Gamma2/Gamma0 ratios of 0.1 and 0.2 were therefore tested separately; they are not measured line-list revisions.",
              "Their development-score improvement is less than 0.00014 percentage points. These five additional refits remained",
              "development-only diagnostics and did not alter the already frozen final-model selection.",
              "Individual line strengths, centers, air widths and shift uncertainties were tested as local response diagnostics.",
              "No unique line-list correction was established. Additive/background structure and continuum errors remain possible.", "",
              "Numerical and observational checks", "----------------------------------",
              "Temperature opacity tables use a 2 K grid with smooth positive interpolation, checked against direct ExoJAX.",
              "Independent HAPI Voigt checks cover selected spectral segments and three T/P combinations. The maximum relative",
              "L2 cross-section difference is 1.43e-5. This checks numerical consistency and units, not the accuracy of the line list.",
              "Saved exact-profile checks quantify opacity-table interpolation error for each fitted atmosphere.",
              "Independent checks reconstruct masked continuum fits and perturb excluded/scoring pixels to test leakage.",
              "Hydrostatic dry-column consistency changes the fitted-profile spectrum by only about 0.0019% RMS in a fixed-parameter check.",
              "Across 75 diagnostic exposures, residual structure is predominantly time-stable; the four reserved exposures were excluded.",
              "Time stability alone does not determine the cause of the remaining residual structure.",
              "No per-pixel uncertainties are supplied: small RMS differences are not formal significance measurements.",
              "Only three orders were physically fit; these results do not establish performance across all 44 orders or other nights.", "",
              "Separate frozen empirical diagnostic", "------------------------------------",
              "A mean residual template built from the five training exposures was frozen before the final exposures were scored.",
              "Adding that empirical template reduces pooled final scoring RMS from 1.00523% to 0.53845% across 2,132 pixels.",
              "This is evidence for a transferable same-night residual pattern, not a new physical atmosphere or an identification",
              "of its source. The correction could absorb several persistent modeling, calibration or data-reduction errors.",
              "It is not used to select the physical models above. See static_residual_final_summary.json and related",
              "static_residual_comparison/final artifacts; temporal_model_comparison removes static components separately.", "",
              "Reproduction and artifacts", "--------------------------",
              "Each fit JSON records command arguments, source-code hash, input fit hash, mask hash and any opacity-override hash.",
              "Run python IGRINS/check_investigation.py after the fits, then python IGRINS/summarize_investigation.py.",
              "Main tables: investigation_screening.csv, investigation_final.csv, investigation_summary.json.",
              "Main figures: investigation_final.png, profile_comparison.png, opacity_completeness.png, transfer_features.png.",
              "Supporting reports: OPACITY_AUDIT_REPORT.txt, TRANSFER_REPORT.txt, FOCUS_REPORT.txt, and the numerical-check JSON files.",
              "Opacity refit table: opacity_development_refits.json; coefficient coverage: advanced_parameter_availability.json.", "",
              "Primary sources", "---------------",
              "Data: https://zenodo.org/records/10382053",
              "Reduction: https://arxiv.org/html/2312.13069v1",
              "HITRAN definitions: https://hitran.org/docs/definitions-and-units/",
              "HITRAN uncertainties: https://hitran.org/docs/uncertainties/",
              "HITRAN2024: https://hitran.org/media/refs/HITRAN-2024.pdf",
              "HAPI profiles: https://hitran.org/hapi/",
              "HITRAN CIA definitions and band coverage: https://hitran.org/data/CIA/CIA_Readme_2024.pdf",
              "Reference O2/O3 profiles: https://eodg.atm.ox.ac.uk/RFM/atm/",
              "NOAA NCEP/NCAR reanalysis: https://psl.noaa.gov/data/gridded/data.ncep.reanalysis.html", ""]
    (OUT / "INVESTIGATION_REPORT.txt").write_text("\n".join(lines))
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
