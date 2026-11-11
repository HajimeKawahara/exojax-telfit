"""Diagnose the 4820 cm-1 residual without modifying the adopted model."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from fit_line_strength import build_context, make_forward, predict_components

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    source = OUT / "adopted_telluric_fit.npz"
    original_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    with np.load(source) as saved:
        data = {k: saved[k] for k in saved.files}
    context = build_context()
    z = data["parameters"][data["line_parameter_indices"]]
    baseline = predict_components(context, z)
    np.testing.assert_allclose(baseline, data["component"], rtol=0, atol=1e-10)
    block = context["blocks"][0]
    grid = np.asarray(block["grid"])
    star = np.asarray(block["shifted_star"])
    search = (grid > 4819.75) & (grid < 4820.15)
    centers = grid[np.flatnonzero(search)[np.argmin(star[:, search], axis=1)]]
    distance = np.abs(grid[None, :] - centers[:, None])
    taper = .5 * (1 + np.cos(np.pi * np.clip((distance-.15)/.20, 0, 1)))
    # A diagnostic counterfactual: remove local stellar absorption before
    # multiplication by telluric transmission and the unchanged instrumental IP.
    modified_star = star + taper * np.maximum(1-star, 0)
    changed = dict(block, shifted_star=modified_star)
    _, forward = make_forward(changed)
    component = baseline.copy()
    component[0] = np.asarray(forward(z[block["parameter_slice"]]))
    residual = context["project"](data["y_over_c"] - component)
    effect = component - baseline
    filtered_effect = context["project"](effect)
    nu = 1e4/data["wavelength_um"][0]
    local = (nu >= 4819.74) & (nu <= 4820.10) & data["score_mask"][0]
    peak = int(np.argmin(np.abs(nu-4819.920802)))
    frame = int(np.flatnonzero(data["exposures"] == 30)[0])
    with np.load(OUT / "premask_stellar_component.npz") as saved:
        stellar = saved["stellar_only"][0]

    metrics = {}
    for name, rows in [("training", slice(0, 5)), ("evaluation", slice(5, None))]:
        before = data["residual_filtered"][0, rows]
        after = residual[0, rows]
        metrics[name] = dict(
            local_score_samples=int(before[:, local].size),
            rms_before_percent=float(100*np.sqrt(np.mean(before[:, local]**2))),
            rms_after_percent=float(100*np.sqrt(np.mean(after[:, local]**2))),
            peak_mean_before_percent=float(100*before[:, peak].mean()),
            peak_mean_after_percent=float(100*after[:, peak].mean()),
            peak_mean_raw_effect_percent=float(100*effect[0, rows, peak].mean()),
            peak_mean_filtered_effect_percent=float(100*filtered_effect[0, rows, peak].mean()))
    report = dict(
        adopted=False, refit=False, peak_wavenumber_cm1=float(nu[peak]),
        peak_wavelength_um=float(data["wavelength_um"][0, peak]),
        frame30_stellar_depth_percent=float(100*(1-stellar[peak])),
        local_interval_cm1=[4819.74, 4820.10], metrics=metrics,
        counterfactual="Remove positive stellar absorption around each predicted stellar minimum in 4819.75-4820.15 cm-1, with full removal within0.15 cm-1 and cosine taper to zero at0.35 cm-1; retain telluric transmission, IP, all parameters and original Q5 mask.",
        limitations=["An explanatory counterfactual, not a fitted or adopted stellar correction.",
                     "A matching feature can also contain extraction or telluric-wing errors; this is not a unique attribution.",
                     "Removing a predicted stellar feature does not identify its atomic/molecular transition.",
                     "The feature was inspected after seeing the data; local RMS is descriptive."],
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in [Path(__file__), source, OUT/"premask_stellar_component.npz",
                                 ROOT/"cache/stellar_phoenix.npz", ROOT/"fit_line_strength.py"]})
    (OUT/"feature4820_counterfactual.json").write_text(json.dumps(report, indent=2)+"\n")
    np.savez_compressed(OUT/"feature4820_counterfactual.npz", wavenumber_cm1=nu,
                        exposures=data["exposures"], component=component[0],
                        residual_filtered=residual[0], effect=effect[0],
                        filtered_effect=filtered_effect[0], local_score_mask=local)

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1.4, 1.4]})
    axes[0].plot(nu, data["y_over_c"][0, frame], "k.-", ms=3, lw=.8, label="Observed, frame 30")
    axes[0].plot(nu, data["y_over_c"][0, frame]-data["residual_filtered"][0, frame],
                 color="tab:blue", label="Adopted PHOENIX + telluric + Q5")
    axes[0].plot(nu, data["y_over_c"][0, frame]-residual[0, frame],
                 color="tab:orange", label="Local stellar absorption removed + Q5")
    axes[0].plot(nu, stellar, "--", color="tab:green", label="PHOENIX star alone")
    axes[0].set(ylabel="Flux / original continuum", ylim=(.35, 1.04))
    axes[0].legend(fontsize=8, ncol=2, loc="lower center")
    for j in range(5, 9):
        axes[1].plot(nu, 100*data["residual_filtered"][0, j], alpha=.35, lw=.8)
    axes[1].plot(nu, 100*data["residual_filtered"][0, 5:].mean(axis=0),
                 color="tab:blue", lw=1.5, label="Adopted: evaluation mean")
    axes[1].plot(nu, 100*residual[0, 5:].mean(axis=0), color="tab:orange",
                 lw=1.5, label="Stellar counterfactual: evaluation mean")
    axes[1].set(ylabel="Q5 residual (%)", ylim=(-2.8, 3.1))
    axes[1].legend(fontsize=8, ncol=2)
    axes[2].plot(nu, 100*data["residual_unfiltered"][0, 5:].mean(axis=0),
                 color="tab:purple", label="Before Q5: evaluation mean")
    axes[2].plot(nu, 100*effect[0, 5:].mean(axis=0), color="tab:green",
                 label="Exact model increase from removing local stellar absorption")
    axes[2].set(ylabel="Original continuum (%)", xlabel=r"Wavenumber (cm$^{-1}$)", ylim=(-2.8, 3.1))
    axes[2].legend(fontsize=8, loc="lower right")
    for ax in axes:
        ax.axvspan(4819.74, 4820.10, color="tab:green", alpha=.06)
        for center in [4819.417298, 4820.433968]:
            ax.axvline(center, color="tab:red", ls=":", lw=.8)
        ax.set_xlim(4818.5, 4821.5)
        ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        ax.grid(alpha=.15)
    axes[1].text(4819.417298, -2.55, "CO2 4819.417: strength fixed", ha="center", fontsize=8)
    axes[1].text(4820.433968, -2.55, "CO2 4820.434: strength fixed", ha="center", fontsize=8)
    for ax in axes[1:]:
        ax.axhline(0, color="0.5", lw=.7)
    fig.suptitle("4820 cm$^{-1}$: a weak stellar-feature mismatch is a leading explanation\nAll plotted pixels shown; local metric uses original score pixels. No fit or adoption.")
    fig.tight_layout()
    fig.savefig(OUT/"feature4820_diagnostic.png", dpi=170)
    plt.close(fig)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_hash
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
