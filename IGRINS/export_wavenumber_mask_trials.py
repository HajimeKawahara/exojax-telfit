"""Export unchanged residual samples split into retained and quarantined sets."""
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


def main():
    source = OUT/"adopted_telluric_fit.npz"
    with np.load(source) as saved:
        original = {key: saved[key] for key in saved.files}
    sources = [Path(__file__), source, OUT/"adopted_telluric_model.json"]
    residual = original["residual_filtered"]
    arrays = {key: original[key] for key in ["orders", "exposures", "wavelength_um"]}
    arrays["wavenumber_cm1"] = 1e4/original["wavelength_um"]
    arrays["original_included_mask"] = original["included_mask"]
    arrays["original_score_mask"] = original["score_mask"]
    cases, rows = {}, []
    for stem, names in [("wavenumber_mask_candidates", ["budget02"]),
                        ("wavenumber_mask_protected", ["protected02", "order34_hotspot"])]:
        paths = [OUT/f"{stem}.npz", OUT/f"{stem}.json"]
        sources += paths
        meta = json.loads(paths[1].read_text())
        with np.load(paths[0]) as masks:
            for name in names:
                index = masks["case_names"].tolist().index(name)
                keep = masks["included_masks"][index]
                score = masks["score_masks"][index]
                quarantine = original["included_mask"] & ~keep
                assert not np.any(keep & quarantine)
                np.testing.assert_array_equal(keep | quarantine, original["included_mask"])
                retained = np.where(keep[:, None], residual, np.nan)
                isolated = np.where(quarantine[:, None], residual, np.nan)
                for values, mask in [(retained, keep), (isolated, quarantine)]:
                    full_mask = np.broadcast_to(mask[:, None], residual.shape)
                    np.testing.assert_array_equal(values[full_mask], residual[full_mask])
                arrays.update({f"keep_included_{name}": keep, f"keep_score_{name}": score,
                               f"quarantine_mask_{name}": quarantine,
                               f"residual_retained_{name}": retained,
                               f"residual_quarantined_{name}": isolated})
                eval_keep = np.broadcast_to(score[:, None], residual[:, 5:].shape)
                eval_removed = np.broadcast_to((original["score_mask"] & ~score)[:, None], residual[:, 5:].shape)
                eval_original = np.broadcast_to(original["score_mask"][:, None], residual[:, 5:].shape)
                values = residual[:, 5:]
                cases[name] = dict(source=stem, removed_included=int(quarantine.sum()),
                    removed_score=int((original["score_mask"] & ~score).sum()),
                    included_removal_percent=float(100*quarantine.sum()/original["included_mask"].sum()),
                    score_removal_percent=float(100*(original["score_mask"] & ~score).sum()/original["score_mask"].sum()),
                    retained_rms_percent=float(100*np.sqrt(np.mean(values[eval_keep]**2))),
                    isolated_squared_residual_percent=float(100*np.sum(values[eval_removed]**2)/np.sum(values[eval_original]**2)),
                    maximum_retained_value_change=0., intervals=meta["cases"][name]["intervals"])
                for interval in cases[name]["intervals"]:
                    rows.append(dict(case=name, **{key: interval[key] for key in
                        ["order", "start_index", "stop_index_exclusive", "nu_lower_cm1", "nu_upper_cm1",
                         "wavelength_lower_um", "wavelength_upper_um", "newly_removed_included", "newly_removed_score"]}))
    arrays["case_names"] = np.array(list(cases))
    product = OUT/"wavenumber_mask_trials.npz"
    np.savez_compressed(product, **arrays)
    with (OUT/"wavenumber_mask_trials.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), status="Experimental masks; not adopted",
        baseline="Current fixed_contrast10, previous IP and degree-five polynomial projection",
        operation="Post-fit masking only: copy retained residuals exactly; expose excluded residuals separately; use NaN for samples outside each set",
        data_integrity="All retained and quarantined included samples are bitwise equal to the source; no source data or adopted product is edited",
        earlier_fit_note="These masks do not retroactively remove excluded points from earlier physical-model or polynomial fits",
        n_original_included=int(original["included_mask"].sum()), n_original_score=int(original["score_mask"].sum()),
        scope="Three orders and nine saved exposures; RMS uses the four evaluation exposures",
        cases=cases, product=product.name, product_sha256=hashlib.sha256(product.read_bytes()).hexdigest(),
        limitations=["Any mask can remove a planet line; small pixel fractions do not guarantee small signal loss.",
                     "Residual reduction here means selecting retained samples, not improving the fit to each observation.",
                     "Template-aware design constraints apply to the design bank, and must be read together with independent stress tests.",
                     "Pixel uncertainty/covariance is unavailable; template-power loss is not a measured detection-significance loss."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"wavenumber_mask_trials.json").write_text(json.dumps(manifest, indent=2)+"\n")
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), constrained_layout=True)
    for b, ax in enumerate(axes):
        nu = arrays["wavenumber_cm1"][b]
        mean = 100*residual[b, 5:].mean(axis=0)
        ax.plot(nu, np.where(original["included_mask"][b], mean, np.nan), color=".6", lw=.8,
                label="Original evaluation mean residual")
        removed = arrays["quarantine_mask_protected02"][b]
        ax.scatter(nu[removed], mean[removed], color="C3", s=14, label="Template-aware trial exclusion")
        ax.axhline(0., color=".7", lw=.5)
        ax.set(xlabel="Wavenumber (cm$^{-1}$)", ylabel="Residual (% of continuum)",
               title=f"Order {original['orders'][b]}")
        ax.legend(fontsize=8)
    fig.savefig(OUT/"wavenumber_mask_trials.png", dpi=160)
    plt.close(fig)
    print("Saved unchanged retained/quarantined samples:", product)
    for name, case in cases.items():
        print(name, case["removed_included"], case["removed_score"], case["retained_rms_percent"])


if __name__ == "__main__":
    main()
