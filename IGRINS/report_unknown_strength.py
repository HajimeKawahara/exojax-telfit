"""Summarize the explicit-prior experiment using only saved results."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    result = json.loads((OUT/"unknown_strength_fit.json").read_text())
    with np.load(OUT/"unknown_strength_fit.npz") as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(OUT/"premask_stellar_component.npz") as saved:
        star = saved["stellar_only"][0]
    nu = 1e4/data["wavelength_um"][0]
    frame = int(np.flatnonzero(data["exposures"] == 30)[0])
    fig, axes = plt.subplots(2, 2, figsize=(13, 7), sharex="col",
                             gridspec_kw={"height_ratios": [1.5, 1]})
    windows = [(4817.75, 4818.35, "CO2 4818.0227: intensity -2.47%"),
               (4819.05, 4820.55, "CO2 4819.4173: intensity +7.56%")]
    for col, (low, high, title) in enumerate(windows):
        ax, rx = axes[:, col]
        inside = (nu >= low) & (nu <= high)
        ax.plot(nu, data["y_over_c"][0, frame], "k.-", lw=.8, ms=3, label="Observed: frame 30")
        for name, label, color in [("adopted", "Adopted model", "tab:blue"),
                                   ("joint30", "Unknown strengths fitted", "tab:orange")]:
            residual = data[f"residual_{name}"][0]
            ax.plot(nu, data["y_over_c"][0, frame]-residual[frame], color=color, lw=1.1, label=label+" + Q5")
            rx.plot(nu, 100*np.where(data["included_mask"][0], residual[5:].mean(axis=0), np.nan),
                    color=color, lw=1.2, label=label)
        ax.plot(nu, star, "--", color="tab:green", lw=1, label="PHOENIX star alone")
        ax.set(title=title, ylim=(max(0, data["y_over_c"][0, frame, inside].min()-.05), 1.04), ylabel="Flux / original continuum")
        rx.set(ylabel="Mean evaluation residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)", ylim=(-1.8, 1.8))
        rx.axhline(0, color="0.5", lw=.7)
        edges = np.r_[nu[0]-(nu[1]-nu[0])/2, (nu[1:]+nu[:-1])/2, nu[-1]+(nu[-1]-nu[-2])/2]
        mask = ~data["included_mask"][0]
        starts = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == 1)
        ends = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)) == -1)
        for axis in [ax, rx]:
            for start, end in zip(starts, ends):
                axis.axvspan(edges[start], edges[end], color=".93", zorder=0)
            axis.set_xlim(low, high)
            axis.ticklabel_format(axis="x", style="plain", useOffset=False)
            axis.grid(alpha=.15)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    axes[1, 0].legend(fontsize=8)
    fig.suptitle("Previously fixed CO2 intensities with an explicit 30% prior\nUpper: frame 30 spectrum and unchanged star. Lower: four-frame mean; gray = original exclusions.")
    fig.tight_layout()
    fig.savefig(OUT/"unknown_strength_local_fit.png", dpi=170)
    plt.close(fig)

    lines = [
        "Explicit intensity priors for unreported and open-ended HITRAN uncertainties",
        result["created_utc"], "",
        "The fixed CO2 line at 4819.417298 cm-1 benefits from a +7.56% effective",
        "intensity correction. Its local included-pixel evaluation RMS falls from",
        "0.8973% to 0.4817%. The global score RMS changes only from 0.77461% to",
        "0.77114%, partly because most pixels in this particular trough are outside",
        "the original scoring subset. No adopted-model replacement is made.", "",
        "Assumptions fixed before evaluation:",
        "Eight otherwise eligible CO2 transitions in order13 are added: two code0",
        "(uncertainty unavailable) and six code3 (>=20%, no finite upper edge).",
        "The same S296>=1e-25 and 10 km/s included-pixel proximity rules apply.",
        "These lines were already present in opacity. Only delta_i*tau_i is added.",
        "For each new line, S=S0*(1+delta), delta>=-1, with no upper bound.",
        "The primary Gaussian shrinkage scale is 0.30*[0.1+0.9*d^2/(d^2+0.005^2)],",
        "where d is a frozen predicted line response, not an observed residual.",
        "Its weak-response limit is 3%; the eight actual scales range 15.7-30.0%.",
        "This is an explicit regularizer, not a measured or HITRAN-supplied sigma.",
        "10% and 100% maximum scales are sensitivity checks, not alternatives chosen",
        "using evaluation data. The 30% primary was declared in unknown_strength_plan.json.",
        "All 269 existing intensity parameters retain their saved weak-line priors.",
        "PHOENIX, TP, columns, original IP, wavelength solution, original continua,",
        "masks and Q5 filtering are unchanged. Five frames fit the shared strengths;",
        "four other frames evaluate them. Smooth Q5 coefficients remain profiled.", "",
        "Case | Score RMS % | Included RMS % | Order13 score RMS % | 4819.42 local included RMS %",
    ]
    for name, row in result["cases"].items():
        m = row["metrics"]
        lines.append(f"{name} | {m['evaluation_score']['rms_percent']:.8f} | {m['evaluation_included']['rms_percent']:.8f} | {m['evaluation_score']['per_order']['13']:.8f} | {row['local']['co2_4819_42']['allowed']['rms_percent']:.8f}")
    lines += ["", "Primary fitted new-line corrections:",
              "Center cm-1 | Code | Frozen contrast % | Assumed prior scale % | Intensity change %"]
    for row in result["cases"]["joint30"]["new_line_corrections"]:
        lines.append(f"{row['nu']:.6f} | {row['strength_code']} | {100*row['contrast']:.4f} | {100*row['assumed_fractional_prior_scale']:.4f} | {100*row['fractional_correction']:+.4f}")
    lines += ["", "Interpretation:",
        "The 4819.417 correction stays +7.53 to +7.56% across the three prior scales;",
        "the nearby 4820.434 correction stays +6.45 to +6.51%. This is evidence that",
        "these local absorption mismatches can be partly absorbed by reasonable",
        "intensity changes, rather than a need for very broad unconstrained errors.",
        "The 4818.023 correction has the opposite sign (-2.47%), despite sharing a",
        "band/reference with 4819.417. A common band-wide scale is not established.",
        "Other, less constrained transitions are prior-sensitive: the 100% stress",
        "case drives one line to 0.46% of its original strength and another to 202%.",
        "The nearly identical global RMS does not justify those large corrections.",
        "The primary has no new zero-strength line (new ratios range 0.814-1.271).", "",
        "The positive stellar-overlap feature at 4819.92 remains: its same local",
        "evaluation RMS changes only from 0.9895% to 0.9420%. It should not be treated",
        "as evidence that these CO2 strength parameters can replace a stellar model.",
        "Around 4819.42 the strength correction removes much of the negative trough,",
        "but the remaining core/wing structure can still reflect width, center,",
        "blending or response errors. Effective intensity changes are not laboratory",
        "measurements or proposed corrections to the HITRAN database.", "",
        "Local metric: 4819.25-4819.55 cm-1 has 10 included wavelength pixels, hence",
        "40 evaluation samples; only one of those pixels belongs to the score mask.",
        "The global score still contains 2,092 samples. The 4818.02 interval has no",
        "score pixels. No extra pixel is masked or added to improve these comparisons.",
        "4820.792590 remains fixed because it lies 19.22 km/s from an included pixel,",
        "outside the unchanged 10 km/s eligibility threshold.", "",
        "Checks and limitations:",
        f"Adopted reconstruction error: {result['adopted_reconstruction_error']:.3g}.",
        f"Directional derivative relative L2 error: {result['derivative_checks'][0]['relative_l2']:.3g}.",
        "The known-only control reproduces the adopted result to numerical precision.",
        "Independent Q5, objective, masks, prior mappings, source-hash and RMS audits",
        "are stored in unknown_strength_independent_audit.json.",
        "Primary order13 terminates on ftol, with small residual gradient (optimality",
        "2.53e-4); no global-optimum or calibrated-posterior claim is made.",
        "new_only30 fits only eight new strengths. Its printed objective omits fixed",
        "known-strength prior constants and must not be compared directly with joint30.",
        "Evaluation is conditional on previously explored masks and fitted nuisance",
        "parameters. The working residual weight is not a full correlated-noise model.", "",
        "Figures: unknown_strength_fit.png (order and residual overview);",
        "unknown_strength_local_fit.png (spectral fits with upper-panel stellar model).",
        "Reproduce: python IGRINS/fit_unknown_strength.py; python IGRINS/report_unknown_strength.py",
        "All experiment code and results remain inside Git-ignored IGRINS/.", "",
        "HITRAN uncertainty definitions: https://hitran.org/docs/uncertainties/",
    ]
    planet_path = OUT/"unknown_strength_planet.json"
    if planet_path.exists():
        lines += ["", "Paired moving-signal injection/refit results:",
                  "For FWHM6/Kp200, order13 gain changes from100.5555% to100.4208%:",
                  "a -0.1347 percentage-point change from the eight added parameters.",
                  "FWHM10/Kp100 changes97.8755% to97.8784% (+0.0029 points).",
                  "Reducing the first injected depth from1e-3 to1e-4 reproduces its result.",
                  "The stationary control changes36.9645% to36.5850%; both fits absorb",
                  "much of a stationary signal. Gains above100% reflect projection of",
                  "the transferred fit response, not extra information.",
                  "See UNKNOWN_STRENGTH_PLANET_REPORT.txt and unknown_strength_planet.json.",
                  "These compare the expanded fit with the known-strength control using",
                  "identical observed-resolution artificial forests. They do not cover",
                  "a physical planetary molecular spectrum, raw extraction, or earlier",
                  "nuisance fitting, and are not a measured detection significance."]
    (OUT/"UNKNOWN_STRENGTH_REPORT.txt").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    main()
