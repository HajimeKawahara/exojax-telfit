"""Audit and report the fiducial fit across all stored IGRINS orders."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from zipfile import BadZipFile

os.environ.setdefault("MPLCONFIGDIR", "/tmp/igrins-matplotlib")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPOSURES = np.array([0, 19, 39, 59, 78, 10, 30, 50, 70])
STAGES = ("baseline", "stellar", "refit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(values):
    values = np.asarray(values)
    return float(100 * np.sqrt(np.mean(values ** 2))) if values.size else None


def q5(raw, wave, included):
    """Project six smooth modes with an independent QR factorization."""
    x = 2 * (np.log(wave) - np.log(wave).min()) / np.ptp(np.log(wave)) - 1
    basis = np.polynomial.chebyshev.chebvander(x, 5)
    orthogonal, triangular = np.linalg.qr(basis[included], mode="reduced")
    coefficients = np.linalg.solve(triangular, orthogonal.T @ raw[:, included].T)
    smooth = (basis @ coefficients).T
    return raw - smooth, smooth


def selected_metrics(residual, mask):
    return dict(wavelength_pixels=int(mask.sum()), evaluation_samples=int(4 * mask.sum()),
                training_rms_percent=rms(residual[:5, mask]),
                evaluation_rms_percent=rms(residual[5:, mask]))


def continuum_settings(path, metadata):
    """Read the actual continuum scope and degree from the physical fits."""
    prefix = metadata.get("physical_prefix")
    if prefix is None:
        candidates = [Path(name).name for name in metadata.get("source_sha256", {})
                      if name.endswith("_physical_e10.json")]
        if candidates:
            prefix = candidates[0][:-len("10.json")]
    scopes, degrees = {}, {}
    for exposure in EXPOSURES[5:]:
        source = path.parent / f"{prefix}{exposure}.json" if prefix is not None else None
        if source is not None and source.exists():
            physical = json.loads(source.read_text())
            arguments = physical.get("provenance", {}).get("command_arguments", {})
            scopes[str(exposure)] = physical.get("continuum_scope", arguments.get("continuum_scope", "calibration"))
            degrees[str(exposure)] = physical.get("continuum_degree", arguments.get("continuum_degree", 3))
    return scopes, degrees


def read_product(path, order, reference_wave=None):
    """Validate stored arrays and reconstruct the reported RMS independently."""
    metadata = json.loads(path.with_suffix(".json").read_text())
    product_hash = digest(path)
    assert metadata["product_sha256"] == product_hash, "Product hash does not match metadata"
    with np.load(path) as saved:
        arrays = {key: saved[key] for key in saved.files}
    np.testing.assert_array_equal(arrays["orders"], [order])
    np.testing.assert_array_equal(arrays["exposures"], EXPOSURES)
    wave = arrays["wavelength_um"][0]
    included = arrays["included_mask"][0]
    score = arrays["score_mask"][0]
    assert wave.shape == included.shape == score.shape == (1848,)
    assert included.all(), "The fiducial all-order run must retain all stored pixels"
    assert np.all(np.diff(wave) < 0), "Expected descending wavelength, ascending wavenumber"
    if reference_wave is not None:
        np.testing.assert_allclose(wave, reference_wave[::-1], rtol=0, atol=1e-12)
    pixel = np.arange(len(wave))
    expected_score = ((pixel // 80) % 2 == 1) & (pixel % 80 >= 5) & (pixel % 80 < 75)
    np.testing.assert_array_equal(score, expected_score)
    for name, values in arrays.items():
        if np.issubdtype(values.dtype, np.number):
            assert np.isfinite(values).all(), f"Nonfinite values in {name}"
    y, continuum = arrays["y_over_c"][0], arrays["continuum"][0]
    assert y.shape == continuum.shape == (9, 1848)
    assert (continuum > 0).all(), "Nonpositive fitted continuum"
    residuals, smooth, metrics, errors = {}, {}, {}, []
    for stage in STAGES:
        component = arrays[f"component_{stage}"][0]
        assert component.shape == y.shape and component.min() >= -1e-12
        residuals[stage], smooth[stage] = q5(y - component, wave, included)
        error = float(np.max(abs(residuals[stage] - arrays[f"residual_{stage}"][0])))
        assert error < 1e-10, f"Saved Q5 residual mismatch: {stage}"
        errors.append(error)
        values = residuals[stage]
        metrics[stage] = dict(training_all_rms_percent=rms(values[:5, included]),
                              evaluation_all_rms_percent=rms(values[5:, included]),
                              evaluation_score_rms_percent=rms(values[5:, score]),
                              evaluation_per_frame_rms_percent=[rms(row[included]) for row in values[5:]])
        for scope, value in [("training", metrics[stage]["training_all_rms_percent"]),
                             ("evaluation_included", metrics[stage]["evaluation_all_rms_percent"]),
                             ("evaluation_score", metrics[stage]["evaluation_score_rms_percent"])]:
            np.testing.assert_allclose(value, metadata["metrics"][stage][scope]["rms_percent"],
                                       rtol=0, atol=1e-9)
    np.testing.assert_allclose(arrays["component"][0], arrays["component_refit"][0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(arrays["residual_filtered"][0], residuals["refit"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(arrays["residual_unfiltered"][0], y - arrays["component"][0], rtol=0, atol=1e-12)
    minimum_t = np.min(arrays["telluric_only_baseline"][0, :5], axis=0)
    star_depth = np.max(1 - arrays["stellar_only_baseline"][0, :5], axis=0)
    np.testing.assert_array_equal(arrays["calibration_mask"][0], minimum_t > .95)
    star, telluric = arrays["stellar_only"][0], arrays["telluric_only"][0]
    assert star.min() >= -1e-12 and telluric.min() >= -1e-12 and telluric.max() <= 1 + 1e-10
    q, z = arrays["q"], arrays["z"]
    assert q.min(initial=0) >= -1e-12
    assert np.all(q <= arrays["stellar_upper"] + 1e-9)
    np.testing.assert_array_equal(q[~arrays["stellar_free_mask"]], np.ones((~arrays["stellar_free_mask"]).sum()))
    records = metadata["line_records"]
    factors = 1 + np.array([r["uncertainty_upper"] for r in records]) * z
    assert factors.min(initial=0) >= -1e-10
    species = dict(Counter(r["species"] for r in records))
    assert metadata["free_strength_count"] == len(z) == len(records)
    assert metadata["free_strength_species"] == species
    fits = [fit for group in metadata["fits"].values() for fit in group]
    success = bool(fits) and all(fit.get("success", False) for fit in fits)
    warnings = []
    scopes, degrees = continuum_settings(path, metadata)
    continuum_relative_minimum = float(np.min(continuum / np.median(continuum, axis=1)[:, None]))
    if any(scope != "calibration" for scope in scopes.values()):
        warnings.append("Held-out continuum scope differs from the fiducial calibration blocks")
    if any(degree != 3 for degree in degrees.values()):
        warnings.append(f"Continuum degree differs from cubic fiducial: {sorted(set(degrees.values()))}")
    if continuum_relative_minimum < .01:
        warnings.append("Low throughput: continuum falls below 1% of its frame median")
    if not success:
        warnings.append("One or more optimizers did not report success")
    if not metadata.get("numerically_stabilized", False):
        warnings.append("Strict repeated-refit prediction threshold not met")
    if star.max() > 1 + 1e-6:
        warnings.append("Stellar-only model exceeds unity")
    if not arrays["stellar_free_mask"].any():
        warnings.append("No stellar amplitudes are identifiable/free")
    if not len(z):
        warnings.append("No reported-error telluric strengths are free")
    if not (minimum_t > .95).any():
        warnings.append("No baseline telluric-clear stellar calibration pixels")
    negative = int(np.count_nonzero(y <= 0))
    if negative:
        warnings.append("Nonpositive observed samples retained")
    depth_bins = {}
    boundaries = [-np.inf, .15, .50, .80, .95, np.inf]
    for lower, upper, label in zip(boundaries[:-1], boundaries[1:],
                                   ["T<=0.15", "0.15<T<=0.50", "0.50<T<=0.80", "0.80<T<=0.95", "T>0.95"]):
        mask = (minimum_t > lower) & (minimum_t <= upper)
        depth_bins[label] = selected_metrics(residuals["refit"], mask)
    stellar_regions = {f"depth>{threshold}": selected_metrics(residuals["refit"], star_depth > threshold)
                       for threshold in [.02, .10]}
    raw = y - arrays["component_baseline"][0]
    metrics["baseline_unfiltered"] = dict(training_all_rms_percent=rms(raw[:5]),
                                           evaluation_all_rms_percent=rms(raw[5:]),
                                           evaluation_score_rms_percent=rms(raw[5:, score]))
    row = dict(cube_index=order, status="complete" if success else "fit_not_converged",
               wavelength_min_um=float(wave.min()), wavelength_max_um=float(wave.max()),
               included_wavelengths=int(included.sum()), evaluation_samples=int(4 * included.sum()),
               score_wavelengths=int(score.sum()), nonpositive_observed_samples=negative,
               stellar_calibration_pixels=int((minimum_t > .95).sum()),
               free_stellar_amplitudes=int(arrays["stellar_free_mask"].sum()),
               free_telluric_strengths=len(z), free_strength_species=species,
               optimizer_success=success, numerically_stabilized=bool(metadata.get("numerically_stabilized", False)),
               physical_continuum_scope=scopes,
               physical_continuum_scope_deviation=any(scope != "calibration" for scope in scopes.values()),
               physical_continuum_degree=degrees,
               physical_continuum_degree_deviation=any(degree != 3 for degree in degrees.values()),
               continuum_minimum=float(continuum.min()), continuum_maximum=float(continuum.max()),
               continuum_min_to_frame_median=continuum_relative_minimum,
               q5_reconstruction_max_error=max(errors), metrics=metrics,
               transmission_bins=depth_bins, stellar_depth_regions=stellar_regions,
               stellar_maximum=float(star.max()), warnings=warnings,
               product_path=str(path), product_sha256=product_hash,
               metadata_sha256=digest(path.with_suffix(".json")),
               product_origin=metadata.get("reference_reused", metadata.get("reused_reference", False)))
    return row, dict(wave=wave, nu=1e4 / wave, observed=y, continuum=continuum,
                     component=arrays["component"][0],
                     residual_unfiltered=arrays["residual_unfiltered"][0],
                     residual_filtered=arrays["residual_filtered"][0],
                     model=arrays["component_refit"][0] + smooth["refit"],
                     star=star, telluric=telluric, residual=residuals["refit"],
                     residuals=residuals, raw_baseline=raw, included=included,
                     score=score, strong_telluric=minimum_t <= .5)


def combine_products(rows, products, out, prefix):
    """Stack all audited successful orders atomically, without changing flux units."""
    target = out / f"{prefix}_fiducial_fit.npz"
    blocked = [row["cube_index"] for row in rows
               if row["cube_index"] not in products or row["status"] != "complete"
               or not row.get("optimizer_success", False)
               or row.get("runner_stage") not in (None, "complete")]
    result = dict(requested=True, created=False, path=str(target),
                  required_order_count=44, blocked_orders=blocked,
                  existing_file_unchanged=target.exists())
    if len(rows) != 44 or set(products) != set(range(44)) or blocked:
        result["reason"] = "Every stored order must have an independently audited, successful completed product; missing, failed or running orders prevent combination."
        return result
    changed = []
    for row in rows:
        path = Path(row["product_path"])
        if digest(path) != row["product_sha256"] or digest(path.with_suffix(".json")) != row["metadata_sha256"]:
            changed.append(row["cube_index"])
    if changed:
        result.update(blocked_orders=changed, reason="One or more input products changed after their independent audit.")
        return result
    mapping = dict(wavelength_um="wave", included_mask="included", score_mask="score",
                   y_over_c="observed", continuum="continuum", component="component",
                   stellar_only="star", telluric_only="telluric",
                   residual_unfiltered="residual_unfiltered", residual_filtered="residual_filtered")
    combined = {name: np.stack([products[order][key] for order in range(44)])
                for name, key in mapping.items()}
    combined.update(orders=np.arange(44), exposures=EXPOSURES.copy())
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{prefix}_fiducial_fit.", suffix=".npz",
                                         dir=out, delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **combined)
        with np.load(temporary) as saved:
            assert set(saved.files) == set(combined)
            for name, values in combined.items():
                np.testing.assert_array_equal(saved[name], values)
        os.replace(temporary, target)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    result.update(created=True, reason="All 44 products passed the independent audit and optimizer-success gate.",
                  existing_file_unchanged=False, sha256=digest(target),
                  shapes={name: list(values.shape) for name, values in combined.items()},
                  source_sha256={str(row["cube_index"]): row["product_sha256"] for row in rows},
                  all_orders_numerically_stabilized=all(row["numerically_stabilized"] for row in rows),
                  physical_continuum_degree={str(row["cube_index"]): row["physical_continuum_degree"] for row in rows},
                  physical_continuum_scope={str(row["cube_index"]): row["physical_continuum_scope"] for row in rows},
                  units="Same original continuum units and descending-wavelength orientation as the per-order products; component excludes the additive Q5 display correction.",
                  variable_length_parameters="Stellar q and telluric z remain in the individual order products.")
    return result


def bounds(values):
    values = np.concatenate([np.asarray(value).ravel() for value in values])
    lo, hi = float(values.min()), float(values.max())
    pad = max(.06 * (hi - lo), .01)
    return lo - pad, hi + pad


def shade(ax, nu, mask):
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    edges = np.r_[nu[0] - (nu[1] - nu[0]) / 2, (nu[1:] + nu[:-1]) / 2,
                  nu[-1] + (nu[-1] - nu[-2]) / 2]
    for group in groups:
        if len(group):
            ax.axvspan(edges[group[0]], edges[group[-1] + 1], color="#ffe5aa", alpha=.5, zorder=0)


def spectrum_page(row, data):
    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True,
                             gridspec_kw={"height_ratios": [2.4, 1.1, 1.2]})
    fig.subplots_adjust(left=.075, right=.98, top=.86, bottom=.13, hspace=.22)
    index = row["cube_index"]
    label = f"Stored cube index {index}" + (" (user Order 7)" if index == 5 else "")
    metrics = row["metrics"]["refit"]
    fig.suptitle(f"WASP-77Ab / IGRINS: {label}", fontsize=15, weight="bold", y=.975)
    fig.text(.5, .935, f"{row['wavelength_min_um']:.8f}–{row['wavelength_max_um']:.8f} microns | "
             f"all {row['included_wavelengths']} stored pixels | matched-frame RMS {metrics['evaluation_all_rms_percent']:.4f}%", ha="center")
    nu, frame = data["nu"], 6
    top_values = [data[key][frame] for key in ["observed", "model", "star", "telluric"]]
    for values, color, style, width, label in zip(top_values, ["black", "#c53932", "#238b45", "#d58b19"],
                                                 ["-", "-", "--", ":"], [.65, 1., 1., .9],
                                                 ["Observed, frame 30", "Fiducial model + Q5", "Stellar model alone", "Telluric alone"]):
        axes[0].plot(nu, values, color=color, ls=style, lw=width, label=label)
    axes[0].set(ylabel="Flux / continuum", ylim=bounds(top_values))
    axes[0].legend(fontsize=9, ncol=2, loc="lower left")
    values = data["residual"][frame] * 100
    axes[1].plot(nu, values, color=".65", lw=.65)
    axes[1].scatter(nu[data["score"]], values[data["score"]], s=5, color="#2166ac", zorder=3)
    axes[1].set(ylabel="Frame 30 residual (%)", ylim=bounds([values]))
    for exposure, residual, color in zip(EXPOSURES[5:], data["residual"][5:],
                                         ["#1b9e77", "#7570b3", "#d95f02", "#666666"]):
        axes[2].plot(nu, residual * 100, color=color, lw=.6, alpha=.75, label=f"Frame {exposure}")
    axes[2].plot(nu, data["residual"][5:].mean(axis=0) * 100, color="black", lw=1.1, label="Mean")
    axes[2].set(ylabel="Evaluation residual (%)", xlabel=r"Wavenumber (cm$^{-1}$)",
                ylim=bounds([data["residual"][5:] * 100]))
    axes[2].legend(fontsize=8, ncol=5, loc="lower left")
    for ax in axes:
        shade(ax, nu, data["strong_telluric"])
        ax.set_xlim(nu.min(), nu.max())
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[1:]:
        ax.axhline(0, color="black", lw=.5)
    flags = "; ".join(row["warnings"][:2])
    if len(row["warnings"]) > 2:
        flags += f"; {len(row['warnings']) - 2} additional flags in summary JSON"
    fig.text(.5, .035, "Blue points: fixed score subset; gray-only points also included. Yellow: strong reference telluric absorption, included.\n"
             "All-pixel degree-five residual projection; evaluation nuisance parameters are conditional. No enuma input.\n"
             + (f"Flags: {flags}" if flags else "Numerical audit passed; no diagnostic flags."),
             ha="center", fontsize=8)
    return fig


def status_page(row):
    fig, ax = plt.subplots(figsize=(14, 9))
    ax.axis("off")
    ax.text(.05, .9, f"Stored cube index {row['cube_index']}: {row['status']}",
            transform=ax.transAxes, fontsize=18, weight="bold")
    ax.text(.05, .78, row.get("message", "No completed fit product is available."),
            transform=ax.transAxes, fontsize=11, wrap=True)
    return fig


def run(args):
    out = args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    reference = None
    info_path = getattr(args, "info", ROOT / "data/20201214_info.csv")
    noise = None
    if info_path.exists():
        with info_path.open() as stream:
            info = list(csv.DictReader(stream))
        snr = np.array([float(info[index]["Med. SNR"]) for index in EXPOSURES[5:]])
        if np.isfinite(snr).all() and (snr > 0).all():
            noise = float(np.sqrt(np.mean((100 / snr) ** 2)))
    if args.cube.exists():
        with args.cube.open("rb") as stream:
            reference, _ = pickle.load(stream)
        assert reference.shape == (44, 1848)
    status_path = out / f"{args.prefix}_run_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    rows, products, sums = [], {}, {}
    for order in range(44):
        path = out / f"{args.prefix}_o{order:02d}_stellar_fit.npz"
        row = dict(cube_index=order, status="missing", message="Completed NPZ/JSON pair is not available.")
        progress = status.get("orders", {}).get(str(order), {})
        if progress:
            stage = progress.get("stage", "missing")
            row.update(status="missing" if stage == "complete" else stage,
                       message=progress.get("message", f"Runner stage: {stage}; no completed product pair yet."))
        if reference is not None:
            row.update(wavelength_min_um=float(reference[order].min()), wavelength_max_um=float(reference[order].max()))
        if path.exists() and path.with_suffix(".json").exists():
            try:
                row, data = read_product(path, order, None if reference is None else reference[order])
                products[order] = data
                for stage in (*STAGES, "baseline_unfiltered"):
                    residual = data["raw_baseline"] if stage == "baseline_unfiltered" else data["residuals"][stage]
                    accum = sums.setdefault(stage, dict(training_sse=0., training_samples=0,
                                                        evaluation_sse=0., evaluation_samples=0,
                                                        score_sse=0., score_samples=0))
                    for scope, values in [("training", residual[:5]), ("evaluation", residual[5:]),
                                          ("score", residual[5:, data["score"]])]:
                        accum[f"{scope}_sse"] += float(np.sum(values ** 2))
                        accum[f"{scope}_samples"] += values.size
            except (AssertionError, ValueError, KeyError, OSError, BadZipFile, json.JSONDecodeError) as error:
                row.update(status="invalid_product", message=f"{type(error).__name__}: {error}")
        row["metadata_noise_proxy_percent"] = noise
        row["runner_stage"] = progress.get("stage")
        rows.append(row)
    pooled = {stage: {f"{scope}_rms_percent": 100 * np.sqrt(values[f"{scope}_sse"] / values[f"{scope}_samples"])
                       for scope in ["training", "evaluation", "score"]} for stage, values in sums.items()}
    counts = dict(Counter(row["status"] for row in rows))
    summary = dict(created_utc=datetime.now(timezone.utc).isoformat(), requested_stored_orders=44,
                   completed_numeric_audits=len(products), status_counts=counts,
                   all_orders_have_products=len(products) == 44,
                   all_optimizers_successful=len(products) == 44 and all(row.get("optimizer_success", False) for row in rows),
                   training_frames=EXPOSURES[:5].tolist(), evaluation_frames=EXPOSURES[5:].tolist(),
                   pooled_metrics=pooled, pooled_accumulators=sums,
                   pooled_scope="All numerically audited available products; missing/invalid products are not silently counted. Inspect optimizer statuses separately.",
                   order_numbering="Released cube indices 0–43; only cube index 5 has the user-supplied Order 7 label.",
                   rms_units="Percent of the original fitted continuum, pooling individual frame residuals, without dividing by sqrt(number of frames)",
                   photon_noise_note="No order-specific pixel variances are provided; global metadata S/N is not a deep-line noise limit.",
                   metadata_noise_proxy_percent=noise,
                   caveats=["Shared corrections use five training frames; evaluation nuisance parameters and Q5 are fitted conditionally.",
                            "Strong absorption and finite nonpositive observed samples remain included.",
                            "No all-order planet-signal injection test has been performed.",
                            "An optimizer success flag does not imply the strict repeated-refit threshold was met."],
                   rows=rows, script_sha256=digest(__file__))
    if status_path.exists():
        summary["runner_status"] = dict(path=str(status_path), sha256=digest(status_path),
                                         updated_utc=status.get("updated_utc"))
    summary["combined_product"] = (combine_products(rows, products, out, args.prefix)
                                   if getattr(args, "combine", False)
                                   else dict(requested=False, created=False,
                                             reason="Pass --combine to write one artifact after all 44 orders pass the audit."))
    summary_path = out / f"{args.prefix}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    columns = ["cube_index", "status", "wavelength_min_um", "wavelength_max_um", "training_rms_percent",
               "evaluation_all_rms_percent", "evaluation_score_rms_percent", "baseline_q5_evaluation_rms_percent",
               "stellar_q5_evaluation_rms_percent", "stellar_calibration_pixels", "free_stellar_amplitudes",
               "free_telluric_strengths", "numerically_stabilized", "nonpositive_observed_samples",
               "physical_continuum_scope", "physical_continuum_scope_deviation", "continuum_minimum",
               "physical_continuum_degree", "physical_continuum_degree_deviation",
               "continuum_min_to_frame_median", "warnings"]
    columns.insert(-1, "metadata_noise_proxy_percent")
    with (out / f"{args.prefix}_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key) for key in columns}
            if "metrics" in row:
                m = row["metrics"]
                flat.update(training_rms_percent=m["refit"]["training_all_rms_percent"],
                            evaluation_all_rms_percent=m["refit"]["evaluation_all_rms_percent"],
                            evaluation_score_rms_percent=m["refit"]["evaluation_score_rms_percent"],
                            baseline_q5_evaluation_rms_percent=m["baseline"]["evaluation_all_rms_percent"],
                            stellar_q5_evaluation_rms_percent=m["stellar"]["evaluation_all_rms_percent"])
            flat["warnings"] = "; ".join(row.get("warnings", [])) or row.get("message", "")
            flat["physical_continuum_scope"] = json.dumps(row.get("physical_continuum_scope", {}))
            flat["physical_continuum_degree"] = json.dumps(row.get("physical_continuum_degree", {}))
            writer.writerow(flat)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2.2, 1]})
    for stage, color, label in [("baseline", ".6", "PHOENIX + Q5"),
                               ("stellar", "#d58b19", "Stellar calibrated + Q5"),
                               ("refit", "#2166ac", "Fiducial final + Q5")]:
        axes[0].plot(range(44), [row["metrics"][stage]["evaluation_all_rms_percent"] if "metrics" in row else np.nan
                                for row in rows], ".-", color=color, lw=1, ms=5, label=label)
    plotted_rms = [row["metrics"][stage]["evaluation_all_rms_percent"]
                   for row in rows if "metrics" in row for stage in STAGES]
    positive_rms = [value for value in plotted_rms if value > 0]
    logarithmic = bool(positive_rms and max(positive_rms) / min(positive_rms) > 10)
    if logarithmic:
        axes[0].set_yscale("log")
    else:
        axes[0].set_ylim(0, None)
    axes[0].set_ylabel("Four-frame all-pixel RMS (%)" + ("; logarithmic scale" if logarithmic else ""))
    if noise is not None:
        axes[0].axhline(noise, color=".35", ls=":", lw=1, label="Global metadata S/N proxy")
    axes[0].legend(ncol=3, fontsize=10)
    for row in rows:
        i = row["cube_index"]
        color = "#238b45" if row["status"] == "complete" else "#d58b19" if row["status"] == "fit_not_converged" else "#c53932"
        axes[1].bar(i, row.get("stellar_calibration_pixels", 0), color=color)
        if i not in products:
            axes[0].axvspan(i-.3, i+.3, color="#c53932", alpha=.12)
    axes[1].set(ylabel="Telluric-clear\nstellar-calibration pixels", xlabel="Stored cube index (not a physical echelle-order label)",
                xticks=np.arange(44))
    for ax in axes:
        ax.grid(alpha=.15)
        ax.set_xlim(-.5, 43.5)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"IGRINS fiducial model: {len(products)}/44 stored orders with audited products")
    fig.tight_layout()
    fig.savefig(out / f"{args.prefix}_rms_overview.png", dpi=160)
    plt.close(fig)
    if not args.no_spectra:
        with PdfPages(out / f"{args.prefix}_spectral_fits.pdf") as pdf:
            for row in rows:
                fig = spectrum_page(row, products[row["cube_index"]]) if row["cube_index"] in products else status_page(row)
                pdf.savefig(fig)
                if args.png and row["cube_index"] in products:
                    fig.savefig(out / f"{args.prefix}_o{row['cube_index']:02d}_spectral_fit.png", dpi=150)
                plt.close(fig)
    report = ["IGRINS fiducial model across all 44 stored orders", "",
              f"Audited products: {len(products)}/44. Status counts: {json.dumps(counts)}.",
              "Cube indices identify released rows; physical echelle-order labels are not inferred.",
              "All finite stored pixels are retained, including deep absorption and nonpositive measured flux.",
              "Shared training frames: 0,19,39,59,78. Conditional evaluation frames: 10,30,50,70.",
              "RMS is percent of the same fitted continuum; all-pixel Q5 is reconstructed independently.", "",
              f"Global metadata S/N proxy: {noise:.6f}% (not order-specific photon variance)." if noise is not None else "Global metadata S/N proxy unavailable.", "",
              "Index  Wavelength (micron)       Train RMS    All RMS   Score RMS  Status"]
    for row in rows:
        wave = f"{row['wavelength_min_um']:.6f}-{row['wavelength_max_um']:.6f}" if "wavelength_min_um" in row else "not available"
        if "metrics" in row:
            m = row["metrics"]["refit"]
            numbers = f"{m['training_all_rms_percent']:10.5f} {m['evaluation_all_rms_percent']:10.5f} {m['evaluation_score_rms_percent']:10.5f}"
        else:
            numbers = f"{'n/a':>10} {'n/a':>10} {'n/a':>10}"
        report.append(f"{row['cube_index']:5d}  {wave:24s} {numbers}  {row['status']}")
    exceptions = [row for row in rows if row.get("physical_continuum_scope_deviation")
                  or row.get("physical_continuum_degree_deviation")]
    if exceptions:
        report += ["", "Explicit continuum exceptions; all stored samples remain included:"]
        for row in exceptions:
            report.append(f"Cube index {row['cube_index']}: physical degree={row['physical_continuum_degree']}; "
                          f"continuum scope={row['physical_continuum_scope']}; "
                          f"minimum fitted continuum={row['continuum_minimum']:.8g}.")
        report.append("These smooth-continuum exceptions are separate from the degree-five residual projection used for every order.")
    report += ["", "Pooled metrics over available audited products only:", json.dumps(pooled, indent=2), "",
               "Incomplete or invalid orders are listed above, not masked from a claimed all-order result.",
               "Optimizer success and numerical stabilization are separate flags in the JSON/CSV.",
               "Depth-bin and stellar-depth subset statistics are retained per order in the JSON.",
               "No per-pixel photon variances or all-order planet-injection validation are available.",
               "No enuma data are used by this report.", "",
               "Combined full-order product:", json.dumps(summary["combined_product"], indent=2)]
    (out / f"{args.prefix}_report.txt").write_text("\n".join(report) + "\n")
    print(json.dumps(dict(audited_orders=len(products), status_counts=counts, pooled_metrics=pooled), indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--cube", type=Path, default=ROOT / "data/cube14v3.pic")
    parser.add_argument("--info", type=Path, default=ROOT / "data/20201214_info.csv")
    parser.add_argument("--prefix", default="allorders")
    parser.add_argument("--no-spectra", action="store_true", help="Skip the multipage spectrum PDF for quick progress summaries")
    parser.add_argument("--png", action="store_true", help="Also save each completed order's spectrum as a PNG")
    parser.add_argument("--combine", action="store_true", help="Write one spectral NPZ only after all 44 orders have independently audited successful products")
    run(parser.parse_args())
