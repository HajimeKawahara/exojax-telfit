"""Stress frozen masks with all cached H2O lines reweighted to hot temperatures.

These optically thin Gaussian line patterns are not planetary radiative-transfer
models, and completeness of the available HITRAN lines at high T is not assumed.
"""
from __future__ import annotations

import contextlib
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
import os
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np

from check_wavenumber_mask_signal import build_template_bank
from exojax.database.core.line_strength import line_strength_numpy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
C_KMS = 299792.458
TEMPERATURES = [1000., 1500.]
PADDING_KMS = 260.
CHUNK_LINES = 128


def load_line_data():
    """Use local TIPS-2021 tables, never extrapolate the150-400K cache."""
    with np.load(ROOT/"cache/H2O.npz") as saved:
        raw = {key:saved[key] for key in saved.files}
    with contextlib.redirect_stdout(io.StringIO()):
        import hapi
    qtemperatures = [296.]+TEMPERATURES
    partitions = np.array([[hapi.partitionSum(1, int(iso), temperature, version=2021)
                            for temperature in qtemperatures]
                           for iso in raw["partition_isotopes"]], dtype=float)
    original = np.array([np.interp(296., raw["temperatures"], row) for row in raw["partition_sums"]])
    np.testing.assert_allclose(partitions[:,0], original, atol=0., rtol=1e-13)
    isotope_index = np.searchsorted(raw["partition_isotopes"], raw["iso"])
    strengths = np.array([line_strength_numpy(temperature, raw["sw"], raw["nu"], raw["elower"],
                              (partitions[:, j+1]/partitions[:,0])[isotope_index])
                          for j, temperature in enumerate(TEMPERATURES)])
    assert np.all(np.isfinite(strengths)) and np.all(strengths >= 0)
    hapi_source = Path(inspect.getmodule(hapi.partitionSum).__file__)
    meta = dict(cache_temperature_range_K=[float(raw["temperatures"].min()),float(raw["temperatures"].max())],
        high_temperature_source="Installed HAPI offline partitionSum(version=2021), same TIPS edition as existing cache",
        no_cache_extrapolation=True, downloaded_data=False,
        temperatures_K=qtemperatures, isotopes=raw["partition_isotopes"].tolist(),
        partition_sums=partitions.tolist(), reference296_max_difference=float(abs(partitions[:,0]-original).max()),
        hapi_source_path=str(hapi_source), hapi_source_sha256=hashlib.sha256(hapi_source.read_bytes()).hexdigest())
    return raw, strengths, meta


def make_hot_signal(data, raw, strengths, spec, padding=PADDING_KMS):
    """Sum all cached transitions in padded windows without a strength cutoff."""
    line_indices = []
    for wavelength in data["wavelength_um"]:
        nu = 1e4/wavelength
        line_indices.append(np.flatnonzero((raw["nu"] >= nu.min()*np.exp(-padding/C_KMS))
                            & (raw["nu"] <= nu.max()*np.exp(padding/C_KMS))))
    normalization = float(strengths[np.unique(np.concatenate(line_indices))].max())
    signal = np.zeros_like(data["component"])
    shift = spec["kp_kms"]*np.sin(2*np.pi*data["phase"])+spec["systemic_kms"]
    sigma = spec["fwhm_kms"]/np.sqrt(8*np.log(2))
    for b, indices in enumerate(line_indices):
        wavelength = data["wavelength_um"][b]
        reference = np.exp(np.mean(np.log(wavelength)))
        velocity = C_KMS*np.log(wavelength/reference)
        centers = C_KMS*np.log((1e4/raw["nu"][indices])/reference)
        weights = strengths[indices]/normalization
        forest = np.zeros_like(data["component"][b])
        for start in range(0,len(indices),CHUNK_LINES):
            sl = slice(start,start+CHUNK_LINES)
            offset = velocity[None,:,None]-centers[None,None,sl]-shift[:,None,None]
            forest += np.sum(weights[None,None,sl]*np.exp(-.5*(offset/sigma)**2), axis=-1)
        signal[b] = -spec["nominal_depth"]*data["attenuation"][b]*forest
    return signal, line_indices, normalization


def retained_power(signal, original, retained):
    before = np.broadcast_to(original[:,None],signal.shape)
    after = np.broadcast_to(retained[:,None],signal.shape)
    original_power = float(np.sum(signal[before]**2))
    remaining = float(np.sum(signal[after]**2))
    return dict(original_power=original_power, retained_power=remaining,
                retention=remaining/original_power if original_power > 1e-200 else None)


def main():
    data, _, _ = build_template_bank()
    raw, hot_strengths, partition_meta = load_line_data()
    masks = {}
    for filename, names in [("wavenumber_mask_candidates.npz",["none","budget02"]),
                            ("wavenumber_mask_protected.npz",["protected02","order34_hotspot"])]:
        with np.load(OUT/filename) as saved:
            for name in names:
                i = saved["case_names"].tolist().index(name)
                masks[name] = dict(included=saved["included_masks"][i],score=saved["score_masks"][i])
            np.testing.assert_array_equal(saved["wavelength_um"],data["wavelength_um"])
    specs = [dict(name=f"hotwater_T{t:g}_w{w:g}_k{k:g}_v{v:+g}",temperature_K=t,
                  fwhm_kms=w,kp_kms=k,systemic_kms=v,nominal_depth=1e-4)
             for t in TEMPERATURES for w in [6.,10.] for k in [100.,200.] for v in [-20.,0.,20.]]
    signals, records, normalizations = [], [], []
    line_indices = None
    for spec in specs:
        temperature_index = TEMPERATURES.index(spec["temperature_K"])
        signal, line_indices, normalization = make_hot_signal(data,raw,hot_strengths[temperature_index],spec)
        signals.append(signal);normalizations.append(normalization)
        item = dict(spec=spec,masks={},evaluation_order_original_power_fraction={})
        total = retained_power(signal[:,5:],data["score_mask"],data["score_mask"])["original_power"]
        for b,order in enumerate(data["orders"]):
            p = retained_power(signal[b:b+1,5:],data["score_mask"][b:b+1],data["score_mask"][b:b+1])["original_power"]
            item["evaluation_order_original_power_fraction"][str(int(order))] = p/total
        for name,keep in masks.items():
            result = dict(evaluation_score=retained_power(signal[:,5:],data["score_mask"],keep["score"]),
                evaluation_included=retained_power(signal[:,5:],data["included_mask"],keep["included"]),
                training_score=retained_power(signal[:,:5],data["score_mask"],keep["score"]),
                per_evaluation_exposure={},per_evaluation_order={})
            for e in range(5,9):
                result["per_evaluation_exposure"][str(int(data["exposures"][e]))] = retained_power(
                    signal[:,e:e+1],data["score_mask"],keep["score"])
            for b,order in enumerate(data["orders"]):
                result["per_evaluation_order"][str(int(order))] = retained_power(
                    signal[b:b+1,5:],data["score_mask"][b:b+1],keep["score"][b:b+1])
            item["masks"][name] = result
        records.append(item)
        print(spec["name"],"protected",item["masks"]["protected02"]["evaluation_score"]["retention"],flush=True)
    # Doubling padding for the widest/hottest,high-Kp case probes omitted far tails.
    test_spec = specs[-1]
    wide,_,wide_normalization = make_hot_signal(data,raw,hot_strengths[-1],test_spec,padding=2*PADDING_KMS)
    matching = signals[-1]*normalizations[-1]/wide_normalization
    tail_error = float(np.max(abs(wide-matching))/max(float(np.max(abs(wide))),1e-200))
    assert tail_error < 1e-10,tail_error
    summary = {}
    for name in masks:
        pooled = [row["masks"][name]["evaluation_score"]["retention"] for row in records]
        included = [row["masks"][name]["evaluation_included"]["retention"] for row in records]
        epochs = [(entry["retention"],row["spec"]["name"],epoch) for row in records
                  for epoch,entry in row["masks"][name]["per_evaluation_exposure"].items()]
        worst = int(np.argmin(pooled));epoch_worst=min(epochs)
        summary[name] = dict(minimum_evaluation_score_retention=float(min(pooled)),
            maximum_evaluation_score_retention=float(max(pooled)),
            minimum_evaluation_included_retention=float(min(included)),
            worst_template=records[worst]["spec"],minimum_epoch_score_retention=float(epoch_worst[0]),
            worst_epoch_template=epoch_worst[1],worst_epoch=epoch_worst[2])
    sources = [Path(__file__),ROOT/"check_wavenumber_mask_signal.py",ROOT/"cache/H2O.npz",ROOT/"cache/H2O.json",
        OUT/"adopted_telluric_fit.npz",OUT/"quadratic_strength_fit.npz",OUT/"line_shape_fit.json",
        ROOT/"data/20201214_info.csv",OUT/"wavenumber_mask_candidates.npz",OUT/"wavenumber_mask_protected.npz"]
    report = dict(created_utc=datetime.now(timezone.utc).isoformat(),partition_metadata=partition_meta,
        reference_strength_cutoff=None,all_cached_lines_used_inside_padded_windows=True,
        padding_kms=PADDING_KMS,line_counts={str(int(o)):int(len(i)) for o,i in zip(data["orders"],line_indices)},
        high_temperature_strength_normalization="One maximum line strength per temperature over the union of padded windows; common normalization preserves relative order and isotope weighting",
        normalization_strengths=normalizations,padding_doubling_max_relative_signal_difference=tail_error,
        cases=records,summary=summary,
        limitations=["These are optically thin Gaussian H2O patterns at observed resolution, not planetary radiative-transfer spectra or a retrieval.",
            "The existing HITRAN cache is used without a296K line-strength cutoff, but high-temperature line-list completeness is not established.",
            "The150-400K cached partition table is not extrapolated: locally installed offline TIPS-2021 supplies1000/1500K values and reproduces every cached296K value.",
            "Existing natural-abundance reference strengths are retained; isotope abundance is not multiplied a second time.",
            "Model attenuation is approximated by the original saved stellar/telluric spectrum after IP convolution.",
            "Only fixed-mask raw squared-signal power is measured. Q5 changes, refits, real noise covariance and detection significance are not measured.",
            "Masks are frozen; neither these templates nor results alter their boundaries.",
            "Good aggregate retention does not protect every individual planetary line or species."],
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    (OUT/"hot_water_mask_stress.json").write_text(json.dumps(report,indent=2)+"\n")
    np.savez_compressed(OUT/"hot_water_mask_stress.npz",signals=np.array(signals),
        template_names=np.array([s["name"] for s in specs]),wavelength_um=data["wavelength_um"],
        exposures=data["exposures"],orders=data["orders"],included_mask=data["included_mask"],score_mask=data["score_mask"],
        case_names=np.array(list(masks)),included_masks=np.array([m["included"] for m in masks.values()]),
        score_masks=np.array([m["score"] for m in masks.values()]),
        **{f"line_indices_o{int(o)}":i for o,i in zip(data["orders"],line_indices)},
        partition_temperatures_K=np.array(partition_meta["temperatures_K"]),
        partition_isotopes=np.array(partition_meta["isotopes"]),partition_sums=np.array(partition_meta["partition_sums"]))
    lines = ["Frozen-mask stress with all cached hot-weighted H2O lines", "",
             "Case | Minimum eval score power retained (%) | Minimum eval included power retained (%) | Worst exposure retained (%)"]
    for name,row in summary.items():
        lines.append(f"{name} | {100*row['minimum_evaluation_score_retention']:.7f} | {100*row['minimum_evaluation_included_retention']:.7f} | {100*row['minimum_epoch_score_retention']:.7f}")
    lines += ["",f"Lines per padded window: {report['line_counts']}",
              f"Padding-doubling relative error: {tail_error:.3g}","","Limitations:"]+report["limitations"]+[""]
    (OUT/"HOT_WATER_MASK_STRESS_REPORT.txt").write_text("\n".join(lines))
    print("\n".join(lines[:8]),flush=True)


if __name__ == "__main__":
    main()
