"""Audit hot-water mask power and lossless retained/quarantined exports."""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path

import numpy as np

from exojax.utils.constants import hcperk

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458


def power(signal, original, retained):
    all_values = signal[np.broadcast_to(original[:,None],signal.shape)]
    keep_values = signal[np.broadcast_to(retained[:,None],signal.shape)]
    before, after = float(all_values @ all_values), float(keep_values @ keep_values)
    return dict(original_power=before, retained_power=after,
                retention=after/before if before > 1e-200 else None)


def main():
    path = OUT/"hot_water_mask_stress.json"
    report = json.loads(path.read_text())
    with np.load(path.with_suffix(".npz")) as saved:
        a = {key:saved[key] for key in saved.files}
    with np.load(ROOT/"cache/H2O.npz") as saved:
        raw = {key:saved[key] for key in saved.files}
    with contextlib.redirect_stdout(io.StringIO()):
        import hapi
    q = np.array([[hapi.partitionSum(1,int(iso),float(t),version=2021)
                   for t in a["partition_temperatures_K"]] for iso in a["partition_isotopes"]])
    np.testing.assert_array_equal(q,a["partition_sums"])
    np.testing.assert_array_equal(q,np.array(report["partition_metadata"]["partition_sums"]))
    cached296 = np.array([np.interp(296.,raw["temperatures"],v) for v in raw["partition_sums"]])
    np.testing.assert_array_equal(q[:,0],cached296)
    assert list(a["partition_temperatures_K"]) == [296.,1000.,1500.]
    assert raw["temperatures"].max() == 400.
    qsource = Path(report["partition_metadata"]["hapi_source_path"])
    assert hashlib.sha256(qsource.read_bytes()).hexdigest() == report["partition_metadata"]["hapi_source_sha256"]
    indices, counts = [], {}
    for b,order in enumerate(a["orders"]):
        nu = 1e4/a["wavelength_um"][b]
        selected = np.flatnonzero((raw["nu"] >= nu.min()*np.exp(-260/C_KMS))
                                 & (raw["nu"] <= nu.max()*np.exp(260/C_KMS)))
        np.testing.assert_array_equal(selected,a[f"line_indices_o{int(order)}"])
        counts[str(int(order))] = int(len(selected));indices.append(selected)
    assert counts == report["line_counts"] and sum(counts.values()) == 3105
    assert report["reference_strength_cutoff"] is None
    union = np.unique(np.concatenate(indices))
    iso_index = np.searchsorted(a["partition_isotopes"],raw["iso"])
    np.testing.assert_array_equal(a["partition_isotopes"][iso_index],raw["iso"])
    normalization = {}
    for column,temp in enumerate(a["partition_temperatures_K"][1:],1):
        # Equivalent log form, including partition, Boltzmann and stimulated-emission factors.
        ratio = (np.log(q[:,0]/q[:,column])[iso_index]
                 +hcperk*raw["elower"]*(1/296.-1/temp)
                 +np.log1p(-np.exp(-hcperk*raw["nu"]/temp))
                 -np.log1p(-np.exp(-hcperk*raw["nu"]/296.)))
        strength = raw["sw"]*np.exp(ratio)
        normalization[float(temp)] = float(strength[union].max())
    error = 0.
    for i,case in enumerate(report["cases"]):
        temp = case["spec"]["temperature_K"]
        np.testing.assert_allclose(normalization[temp],report["normalization_strengths"][i],rtol=2e-13,atol=0)
        signal = a["signals"][i]
        for j,name in enumerate(a["case_names"].tolist()):
            kept,score = a["included_masks"][j],a["score_masks"][j]
            expected = case["masks"][name]
            measures = {
                "evaluation_score":power(signal[:,5:],a["score_mask"],score),
                "evaluation_included":power(signal[:,5:],a["included_mask"],kept),
                "training_score":power(signal[:,:5],a["score_mask"],score)}
            for scope, measured in measures.items():
                for key,value in measured.items():
                    error = max(error,abs(value-expected[scope][key]))
                    np.testing.assert_allclose(value,expected[scope][key],atol=1e-13,rtol=0)
            for e in range(5,9):
                measured = power(signal[:,e:e+1],a["score_mask"],score)
                for key,value in measured.items():
                    np.testing.assert_allclose(value,expected["per_evaluation_exposure"][str(a["exposures"][e])][key],atol=1e-13,rtol=0)
            for b,order in enumerate(a["orders"]):
                measured = power(signal[b:b+1,5:],a["score_mask"][b:b+1],score[b:b+1])
                for key,value in measured.items():
                    np.testing.assert_allclose(value,expected["per_evaluation_order"][str(order)][key],atol=1e-13,rtol=0)
    for key,expected in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected,key
    manifest = json.loads((OUT/"wavenumber_mask_trials.json").read_text())
    export_count = 0
    with np.load(OUT/"adopted_telluric_fit.npz") as source, np.load(OUT/manifest["product"]) as exported:
        residual, included = source["residual_filtered"],source["included_mask"]
        all_included = np.broadcast_to(included[:,None],residual.shape)
        for name in exported["case_names"]:
            keep = exported[f"keep_included_{name}"]
            quarantine = exported[f"quarantine_mask_{name}"]
            np.testing.assert_array_equal(keep|quarantine,included)
            assert not np.any(keep&quarantine)
            left,right = exported[f"residual_retained_{name}"],exported[f"residual_quarantined_{name}"]
            keep_full = np.broadcast_to(keep[:,None],residual.shape)
            removed_full = np.broadcast_to(quarantine[:,None],residual.shape)
            assert left[keep_full].tobytes() == residual[keep_full].tobytes()
            assert right[removed_full].tobytes() == residual[removed_full].tobytes()
            assert np.isnan(left[~keep_full]).all() and np.isnan(right[~removed_full]).all()
            reconstructed = np.where(keep_full,left,right)
            assert reconstructed[all_included].tobytes() == residual[all_included].tobytes()
            export_count += int(all_included.sum())
    for key,expected in manifest["source_sha256"].items():
        assert hashlib.sha256((ROOT/key).read_bytes()).hexdigest() == expected,key
    assert hashlib.sha256((OUT/manifest["product"]).read_bytes()).hexdigest() == manifest["product_sha256"]
    sources = [Path(__file__),path,path.with_suffix(".npz"),ROOT/"cache/H2O.npz",
               OUT/"wavenumber_mask_trials.json",OUT/manifest["product"],OUT/"adopted_telluric_fit.npz"]
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),passed=True,
        partition_296_and_high_temperature_values_verified=True,partition_cache_extrapolation=False,
        line_counts_verified=counts,temperature_strength_formula_and_normalization_verified=True,
        maximum_power_metric_discrepancy=error,export_bitwise_reconstruction_samples=export_count,
        source_and_product_hashes_verified=True,
        limitations=["No radiative-transfer opacity is recomputed; the line-weight formula and saved Gaussian-pattern powers are audited.",
                     "Using every cached line does not establish high-temperature completeness of HITRAN or a physical planet spectrum.",
                     "Power retention is conditional on artificial line patterns and fixed masks, not a guarantee for every species or individual line."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"hot_water_mask_stress_independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("Hot-water/export audit passed",counts,"max metric",error,"export samples",export_count)


if __name__ == "__main__":
    main()
