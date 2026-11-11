"""Compare fitted terrestrial profiles and flag parameter-bound solutions."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from atmosphere_profile import load_profile

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def main():
    reference = load_profile(20)
    p = reference["pressure_bar"]
    column = reference["number_column_cm2"]*reference["vmr"]["H2O"]
    pwv = float(column.sum()*18.01528*1.66053906660e-24*10)
    report = dict(reference_pwv_mm=pwv, exposure=0, models={})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True, constrained_layout=True)
    axes[0].plot(reference["temperature_K"], p, color=".45", lw=2, label="Reanalysis reference")
    axes[1].plot(np.ones_like(p), p, color=".45", lw=2)
    axes[2].plot(reference["vmr"]["H2O"], p, color=".45", lw=2, label=f"Reference: {pwv:.3f} mm")
    variants = [("inv_tp3", "T nodes, 2 exposures", "tab:blue", "-"),
                ("inv_water2", "Water nodes, 2 exposures", "tab:green", "--"),
                ("inv_tp3water2", "T + water nodes, 2 exposures", "tab:orange", "-"),
                ("inv5_tp3", "T nodes, 5 exposures", "tab:red", ":")]
    for tag, label, color, style in variants:
        path = OUT / f"{tag}.json"
        meta = json.loads(path.read_text())
        fitted = meta["fitted_profile"]
        pressure = np.asarray(fitted["pressure_bar"])
        weights = np.asarray(fitted["water_layer_weights"])
        e = (meta["train_exposures"]+[meta["heldout_exposure"]]).index(0)
        water = np.asarray(fitted["water_layer_columns_cm2"])[e]
        vmr = np.asarray(fitted["mass_conserving_water_vmr"])[e]
        amount = float(water.sum()*18.01528*1.66053906660e-24*10)
        axes[0].plot(fitted["temperature_K"], pressure, color=color, ls=style, lw=1.5, label=label)
        axes[1].plot(weights, pressure, color=color, ls=style, lw=1.5)
        axes[2].plot(vmr, pressure, color=color, ls=style, lw=1.5, label=f"{label}: {amount:.3f} mm")
        for bound in meta["bounds_hit"]:
            if bound.startswith("water_shape_node"):
                k = int(bound.removeprefix("water_shape_node").removesuffix("_dex"))
                node = [.1, .35][k]
                value = 10.**meta["parameters"][bound] * weights[-1]
                axes[1].plot(value, node, "D", color=color, ms=7)
        report["models"][tag] = dict(label=label, pwv_e0_mm=amount, bounds_hit=meta["bounds_hit"],
                                     source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                                     temperature_node_corrections_K={key: value for key, value in meta["parameters"].items()
                                                                    if key.startswith("temperature_delta")},
                                     water_shape_node_dex={key: value for key, value in meta["parameters"].items()
                                                           if key.startswith("water_shape_node")})
    axes[0].set(xlabel="Temperature (K)", ylabel="Pressure (bar)", title="Temperature on fixed pressure layers")
    axes[1].set(xlabel="Molecular-column redistribution factor", title="Water shape, total column conserved", xscale="log")
    axes[2].set(xlabel="Mass-consistent H2O VMR", title="Water abundance at exposure 0", xscale="log")
    for ax in axes:
        ax.set_yscale("log")
        ax.set_ylim(.8, .001)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].plot([], [], "D", color=".2", ms=7, label="Water shape node at allowed bound")
    axes[1].legend(fontsize=8, loc="lower left")
    axes[2].legend(fontsize=7, loc="upper left")
    fig.suptitle("Flexible profiles: fitted corrections and boundary sensitivity")
    fig.savefig(OUT / "profile_comparison.png", dpi=160)
    plt.close(fig)
    report["interpretation"] = ("The upper water node hits opposite bounds with and without free temperature. "
                                 "This is a profile-identifiability diagnostic, not a unique atmospheric retrieval. "
                                 "The plotted mass-consistent water VMR is derived from the fitted molecular column; "
                                 "the fit uses the fixed reference background mean mass.")
    (OUT / "profile_comparison.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
