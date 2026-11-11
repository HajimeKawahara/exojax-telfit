"""Restore the original report displays while retaining current results."""

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
ASSET_COMMIT = "59f4f98e87b89ef9c86b3e9788765e67b4b6e8d9"
ASSET_URL = (
    "https://raw.githubusercontent.com/HajimeKawahara/exojax/"
    f"{ASSET_COMMIT}/documents/figures/igrins-blase-20261111"
)


def replace_once(text: str, old: str, new: str) -> str:
    assert text.count(old) == 1, old
    return text.replace(old, new, 1)


def main() -> None:
    body = (RESULTS / "current_blase_issue_body.txt").read_text()
    tables = (RESULTS / "restored_comparison_tables.txt").read_text()
    body = replace_once(
        body,
        "All figures, tables and residual statistics below refer to this current model and its matched controls.",
        "The spectrum and residual figures show this current model. The configuration tables retain the earlier measured experiments alongside the new results, with the stellar model and evaluation sample stated explicitly.",
    )
    old_url = (
        "https://raw.githubusercontent.com/HajimeKawahara/exojax/"
        "168cbd95db3d177b50067cf3b1a91f13e31c9647/"
        "documents/figures/igrins-blase-20261111"
    )
    body = body.replace(old_url, ASSET_URL)
    body = replace_once(
        body,
        "![Current stellar components, stellar-plus-telluric spectral fits, and residuals]"
        f"({ASSET_URL}/current_blase_spectral_fit.png)",
        "![Observed spectra, current total fits, green dashed stellar-only models, and residuals]"
        f"({ASSET_URL}/restored_current_spectral_fit.png)",
    )
    body = replace_once(
        body,
        "**Top panels:** the stellar model alone, so its influence is visible directly. **Middle panels:** the observations and complete model, including the profiled smooth correction. **Bottom panels:** their residuals in percent of the original fitted continuum. Gray regions are the existing exclusions. Curves show four-frame means; RMS uses all individual-frame samples.",
        "The original full-width, vertically stacked layout is retained, using **evaluation frame 30**. For each order, the **upper panel** overlays the observed spectrum (black), the current complete fit including the degree-five smooth correction (red), and the current stellar model alone (green dashed). This makes the stellar contribution visible in the same wavelength coordinates. The **lower panel** shows residuals in percent of the original fitted continuum: gray points are included pixels and blue points are scoring pixels. Gray shaded regions mark the existing exclusions. No included residuals or stellar curves are clipped by the plot limits. The tables use all four evaluation frames, not just the displayed frame.",
    )
    with (ROOT / "data/20201214_info.csv").open(newline="") as stream:
        info = list(csv.DictReader(stream))
    noise_percent = [100 / float(info[e]["Med. SNR"]) for e in (10, 30, 50, 70)]
    pooled_noise = math.sqrt(sum(v * v for v in noise_percent) / 4)
    old_table = (
        "| Order | Current score RMS | Current all-included RMS |\n"
        "| --- | ---: | ---: |\n"
        "| 13 | 0.573685% | 0.575083% |\n"
        "| 34 | 0.752427% | 0.712321% |\n"
        "| 40 | 0.855023% | 0.957083% |\n"
        "| Pooled | **0.716343%** | **0.746884%** |"
    )
    table_lines = old_table.splitlines()
    table_lines[0] += " Estimated photon-noise scale (metadata S/N)† |"
    table_lines[1] += " ---: |"
    table_lines[2:] = [line + f" ≈{pooled_noise:.3f}% |" for line in table_lines[2:]]
    note = (
        "\n\n† Approximate photon-noise scale assuming the metadata S/N is photon dominated: "
        f"`sqrt(mean((100 / Med. SNR)^2)) = {pooled_noise:.6f}%` for evaluation frames "
        f"10, 30, 50 and 70 (individual frames: {min(noise_percent):.3f}–{max(noise_percent):.3f}%). "
        "The release supplies one median S/N per frame, so the same common proxy is shown "
        "for each order; these are not independent order-specific photon-noise estimates. "
        "This is the per-frame scale, not the uncertainty of a four-frame average. "
        "Per-pixel variance, electron counts and gain are unavailable; the estimate is "
        "before Q5, which changes the noise covariance."
    )
    body = replace_once(body, old_table, "\n".join(table_lines) + note)
    body = replace_once(
        body,
        "![Current residuals in the four evaluation frames and close-ups of the largest persistent scored features]"
        f"({ASSET_URL}/current_blase_residuals.png)",
        "![Current residuals across four evaluation frames, with the original four residual intervals enlarged]"
        f"({ASSET_URL}/restored_current_residuals.png)",
    )
    body = replace_once(
        body,
        "The residual close-ups are centered on the largest current four-frame mean residual within each order's scoring mask. This is a display diagnostic only: it does not select fitting pixels or create a new mask.",
        "The three full-width panels show all four evaluation frames in the original layout. The four close-ups retain the **same intervals selected in the earlier training analysis**, now displaying the current residuals; yellow bands identify those intervals. Keeping their positions fixed makes changes directly comparable. No additional mask is applied, and all included residual points fit within the displayed limits. Updated local RMS values appear below.",
    )
    configurations, residual_tables = tables.split(
        "### Where the residual lies: frozen molecular classification", 1
    )
    residual_tables = (
        "### Where the residual lies: frozen molecular classification"
        + residual_tables
    )
    configurations = configurations.replace(
        "## Configuration comparisons and numerical precision",
        "### Configuration comparisons and numerical precision",
        1,
    )
    configurations = replace_once(
        configurations,
        "### Physical-model and stellar-model comparisons\n",
        "### Physical-model and stellar-model comparisons\n\n"
        "![Measured physical and stellar configurations: evaluation RMS and squared-residual reduction]"
        f"({ASSET_URL}/restored_configuration_comparison.png)\n",
    )
    configurations = replace_once(
        configurations,
        "### TransformerPayne: compare the covered H orders separately",
        "In plain terms, changing the tested instrumental profile made very little difference with PHOENIX fixed. Adjusting H2O widths helped more than adjusting their centers; allowing both gave the largest improvement among those physical-model experiments. The tested profile, background and self-broadening changes helped only slightly. This suggests that line-shape mismatch deserves attention, but parameter degeneracies prevent identifying a unique database error. The larger improvement from the current stellar correction also shows why persistent residuals cannot all be assigned to telluric lines.\n\n"
        "### TransformerPayne: compare the covered H orders separately",
    )
    old_comparison = (
        "### Comparison and weak-signal preservation\n\n"
        "![Matched model RMS comparison and the statistical composition of the current residual]"
        f"({ASSET_URL}/current_blase_comparison.png)\n\n"
    )
    body = replace_once(
        body,
        old_comparison,
        residual_tables.rstrip() + "\n\n"
        + configurations.rstrip() + "\n\n"
        + "### Current blasé model: weak-signal preservation\n\n",
    )
    body = body.replace("order13", "order 13").replace(
        "orders34/40", "orders 34/40"
    ).replace("Order13", "Order 13").replace("Order34", "Order 34").replace(
        "Order40", "Order 40"
    )
    body = replace_once(
        body,
        "All current figures are generated from the adopted product and matched saved/reconstructed controls. Earlier physical-profile experiments are not combined into this selection.",
        "The spectrum, residual and local diagnostic figures are generated from the adopted product; the configuration figure and tables also retain the saved historical controls. The current Q5 residual was independently reconstructed to a maximum difference of 3.5e-14 in continuum units. Configuration RMS values were recomputed from saved arrays, and the four displayed interval RMS values were cross-checked against the residual figure. Figure generators are `IGRINS/plot_restored_igrins_report.py` and `IGRINS/restore_comparison_tables.py`; their validation records are `results/restored_current_figures_validation.json` and `results/restored_comparison_tables.json`. Earlier physical-profile experiments are not combined into the current selection.",
    )
    # The main comparison and convergence tables must stay visible.
    assert body.count("<details>") == body.count("</details>") == 2
    assert "current_blase_spectral_fit.png" not in body
    assert "current_blase_residuals.png" not in body
    assert "93.130665%" in body and "0.748502%" in body
    (RESULTS / "restored_igrins_issue_body.txt").write_text(body)
    print(f"Restored report: {len(body):,} characters, {len(body.split()):,} words")


if __name__ == "__main__":
    main()
