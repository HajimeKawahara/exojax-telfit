"""Provisionally adopt the validated, uninjected blase stellar/telluric fit."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from fit_blase_stellar import Problem, additional_metrics
from fit_line_strength import metrics

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
CASE = "blase_stellar_refit_stabilized"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = OUT / "adopted_telluric_model.json"
    previous = json.loads(manifest_path.read_text())
    history = OUT / "model_history/fixed_contrast10"
    history.mkdir(parents=True, exist_ok=True)
    if previous["case"] == "fixed_contrast10":
        for source, target in [(manifest_path, history/"model.json"),
                               (ROOT/"CURRENT_MODEL.txt", history/"CURRENT_MODEL.txt")]:
            if not target.exists():
                shutil.copy2(source, target)
    elif previous["case"] != CASE:
        raise ValueError(f"Unexpected current model: {previous['case']}")
    old = json.loads((history/"model.json").read_text())
    if previous["case"] == "fixed_contrast10":
        assert previous["product_sha256"] == old["product_sha256"]
    assert digest(ROOT/old["product"]) == old["product_sha256"]
    experiment = json.loads((OUT/"blase_stellar_fit.json").read_text())
    validation = json.loads((OUT/"blase_planet.json").read_text())
    for name in ["blase_stellar_fit", "blase_planet"]:
        metadata = json.loads((OUT/f"{name}.json").read_text())
        for path, expected in metadata["source_sha256"].items():
            assert digest(ROOT/path) == expected, path
    problem = Problem()
    context = problem.context
    with np.load(OUT/"blase_planet.npz") as saved:
        q, z = saved["baseline_q_stellar"], saved["baseline_z_stellar"]
        component = saved["baseline_prediction_stellar"]
        for source, key in [("orders", "orders"), ("exposures", "exposures"),
                            ("wavelength_um", "wave"), ("included_mask", "allowed"),
                            ("score_mask", "score")]:
            np.testing.assert_array_equal(saved[source], context[key])
    np.testing.assert_allclose(problem.predict(q, z), component, rtol=0, atol=1e-10)
    stellar = problem.stellar(q)
    measured, residual = metrics(context, component)
    measured["diagnostic"] = additional_metrics(context, residual, problem.stellar(problem.initial_stellar))
    with np.load(ROOT/old["product"]) as saved:
        arrays = {key: saved[key].copy() for key in saved.files}
    physical_parameters = arrays["parameters"].copy()
    physical_parameters[arrays["line_parameter_indices"]] = z
    arrays.update(parameters=np.r_[physical_parameters, q],
                  stellar_parameter_indices=np.arange(len(physical_parameters), len(physical_parameters)+len(q)),
                  q=q, z=z, component=component, stellar_only=stellar,
                  residual_filtered=residual, residual_unfiltered=context["y_over_c"]-component,
                  calibration_mask=problem.calibration_mask,
                  calibration_telluric=problem.calibration_telluric,
                  stellar_free_mask=np.concatenate([b["free"] for b in problem.blocks]),
                  stellar_rest_centers_angstrom=np.concatenate([b["clone"].centers_angstrom for b in problem.blocks]),
                  stellar_feature_orders=np.concatenate([np.full(b["clone"].n_lines, b["order"]) for b in problem.blocks]))
    assert np.isfinite(component).all() and component.min() >= 0
    # Check the filter independently, without the fitting projection callback.
    for b, wave in enumerate(arrays["wavelength_um"]):
        mask = arrays["included_mask"][b]
        x = 2*(np.log(wave)-np.log(wave).min())/np.ptp(np.log(wave))-1
        basis = np.polynomial.chebyshev.chebvander(x, 5)
        orthogonal, triangular = np.linalg.qr(basis[mask], mode="reduced")
        coeff = np.linalg.solve(triangular, orthogonal.T@arrays["residual_unfiltered"][b][:, mask].T)
        independent = arrays["residual_unfiltered"][b]-(basis@coeff).T
        np.testing.assert_allclose(independent, residual[b], rtol=0, atol=1e-11)
    product = OUT / "adopted_blase_telluric_fit.npz"
    temporary = product.with_name(product.stem+".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(product)
    sources = [Path(__file__), OUT/"blase_stellar_fit.json", OUT/"blase_stellar_fit.npz",
               OUT/"blase_planet.json", OUT/"blase_planet.npz", OUT/"blase_independent_audit.json",
               history/"model.json", ROOT/"blase_stellar.py", ROOT/"fit_blase_stellar.py"]
    manifest = dict(schema_version=2, status="provisionally adopted by explicit user instruction",
                    created_utc=datetime.now(timezone.utc).isoformat(), case=CASE,
                    product=str(product.relative_to(ROOT)), product_sha256=digest(product),
                    physical_source=old["physical_source"], ip=old["ip"], filter=old["filter"],
                    strength_prior=old["strength_prior"], scope=old["scope"],
                    stellar=dict(model="PHOENIX times the actual blase clone ratio in the stellar rest frame",
                                 observed_free_parameters="215 shared feature amplitudes; centers and clone widths fixed",
                                 candidate_features=len(q), free_features=int(arrays["stellar_free_mask"].sum()),
                                 plan=experiment["plan"], clone_metadata=experiment["clone_metadata"]),
                    selection="Uninjected stellar baseline from the completed paired planet-response check, after numerical stabilization; no injected fit is adopted",
                    metrics=measured,
                    planet_validation=dict(source="results/blase_planet.json", audit=validation["audit"],
                                           cases=[dict(kind=row["kind"], fwhm_kms=row["fwhm_kms"], kp_kms=row["kp_kms"],
                                                       depth=row["depth"], paired_gain_difference=row["paired_gain_difference"])
                                                  for row in validation["cases"]]),
                    previous_model=dict(manifest=str((history/"model.json").relative_to(ROOT)),
                                        product=old["product"], product_sha256=old["product_sha256"],
                                        restore_command="python IGRINS/adopt_telluric_model.py"),
                    definitions=dict(component="Stellar times telluric spectrum after the original IP, in original continuum units",
                                     residual_filtered="Original included-mask Q5 applied to y_over_c-component",
                                     q="434 clone amplitude ratios in order13/34/40 blocks; 215 selected features free and others equal1",
                                     z="269 original HITRAN intensity correction parameters, with unchanged weak-line priors",
                                     parameters="Legacy physical vector with updated z, followed by q; use the stored parameter-index arrays",
                                     calibration_telluric="Frozen pre-calibration telluric transmission used to select stellar training pixels, not the final refitted transmission"),
                    limitations=experiment["limitations"]+[
                        "Artificial moving-line controls are conditional observed-resolution tests, not a guarantee for all planetary spectra.",
                        "Stationary injected lines lose an additional6.22 percentage points of projection response; some strong stellar lines are overfilled.",
                        "Keep the original masks. Provisional adoption does not validate previously excluded stellar regions.",
                        "The eight unreported-intensity lines tested separately are not part of this model."],
                    source_sha256={str(p.relative_to(ROOT)): digest(p) for p in sources})
    temporary_manifest = manifest_path.with_suffix(".tmp.json")
    temporary_manifest.write_text(json.dumps(manifest, indent=2)+"\n")
    temporary_manifest.replace(manifest_path)
    (ROOT/"CURRENT_MODEL.txt").write_text(
        "Current provisional telluric model\n"
        "==================================\n\n"
        "Authoritative manifest: results/adopted_telluric_model.json\n"
        "Current predictions and filtered residuals: results/adopted_blase_telluric_fit.npz\n"
        "Case: blase_stellar_refit_stabilized (explicit provisional user adoption).\n"
        "PHOENIX with shared blase stellar-amplitude corrections, followed by telluric-strength refitting.\n"
        "Use the stabilized UNINJECTED baseline from results/blase_planet.npz.\n"
        "Keep original masks, degree-five spatial filtering, old IP and weak-line telluric priors.\n"
        "No temporal residual subtraction. No unknown-intensity extension.\n\n"
        f"Evaluation score RMS: {measured['evaluation_score']['rms_percent']:.9f}%\n"
        "Some strong stellar regions are overfilled; stationary signals remain degenerate.\n\n"
        "Recreate this selection: python IGRINS/adopt_blase_model.py\n"
        "Restore previous PHOENIX selection: python IGRINS/adopt_telluric_model.py\n"
        "Previous manifest: results/model_history/fixed_contrast10/model.json\n"
        "Legacy results/adopted_telluric_fit.npz is the FROZEN PHOENIX baseline for historical experiments.\n"
        "New consumers must resolve the current product through the authoritative manifest.\n"
        "A new experiment does not automatically change the selected model.\n"
        "All files in IGRINS remain outside Git tracking.\n")
    print(f"Provisionally adopted {CASE}: evaluation RMS {measured['evaluation_score']['rms_percent']:.9f}%")


if __name__ == "__main__":
    main()
