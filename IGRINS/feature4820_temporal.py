"""Describe the adopted-model residual near 4820 cm-1 without changing the fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def gaussian_description(nu, values):
    """Measure a local positive feature with a Gaussian plus a linear baseline."""
    x = nu - 4819.95
    def model(p):
        return p[0] + p[1] * x + p[2] * np.exp(-.5 * ((x-p[3])/p[4])**2)
    fit = least_squares(lambda p: model(p)-values, [0., 0., .01, 0., .06],
                        bounds=([-.1, -1., 0., -.10, .015], [.1, 1., .1, .10, .15]),
                        x_scale=[.01, .01, .01, .05, .05],
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    p = fit.x
    return dict(center_cm1=float(4819.95+p[3]), amplitude_percent=float(100*p[2]),
                sigma_cm1=float(p[4]), fwhm_cm1=float(2.354820045*p[4]),
                width_at_bound=bool(np.isclose(p[4], .015) or np.isclose(p[4], .15)),
                remaining_rms_percent=float(100*np.sqrt(np.mean(fit.fun**2))),
                baseline_intercept_percent=float(100*p[0]),
                baseline_slope_percent_per_cm1=float(100*p[1]))


def main():
    paths = [OUT / "adopted_telluric_fit.npz", ROOT / "cache/transformerpayne_fit_context.npz",
             ROOT / "data/20201214_info.csv"]
    with np.load(paths[0]) as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(paths[1]) as saved:
        np.testing.assert_array_equal(data["exposures"], saved["exposures"])
        np.testing.assert_array_equal(data["wavelength_um"], saved["wavelength_um"])
        star = saved["phoenix_stellar"][0]
    info = pd.read_csv(paths[2]).iloc[data["exposures"]]
    nu = 1e4 / data["wavelength_um"][0]
    residual = data["residual_filtered"][0]
    raw = data["residual_unfiltered"][0]
    average = residual.mean(axis=0)
    report = dict(adopted_model_unchanged=True, gaussian_fits_are_descriptive_only=True,
                  residual_sign="observed minus model, divided by original continuum",
                  exposures=data["exposures"].tolist(), intervals=[])
    for lo, hi in [(4819.25, 4819.55), (4819.75, 4820.12), (4820.1, 4820.4),
                   (4818.5, 4821.5)]:
        mask = (nu >= lo) & (nu <= hi)
        row = dict(lower_cm1=lo, upper_cm1=hi, pixels=int(mask.sum()),
                   rms_percent=float(100*np.sqrt(np.mean(residual[:, mask]**2))))
        row["common_rms_percent"] = float(100*np.sqrt(np.mean(average[mask]**2)))
        row["variable_rms_percent"] = float(100*np.sqrt(np.mean((residual[:, mask]-average[mask])**2)))
        row["common_variance_fraction"] = row["common_rms_percent"]**2 / row["rms_percent"]**2
        for label, subset in [("all", mask), ("included", mask & data["included_mask"][0]),
                              ("score", mask & data["score_mask"][0])]:
            indices = np.flatnonzero(subset)
            if not len(indices):
                row[label] = None
                continue
            pair = {}
            for kind, j in [("positive", indices[np.argmax(average[indices])]),
                            ("negative", indices[np.argmin(average[indices])])]:
                pair[kind] = dict(index=int(j), nu_cm1=float(nu[j]),
                    mean_residual_percent=float(100*average[j]),
                    mean_raw_residual_percent=float(100*raw[:, j].mean()),
                    residual_percent_by_frame=(100*residual[:, j]).tolist(),
                    mean_stellar_depth_percent=float(100*np.mean(1-star[:, j])),
                    included=bool(data["included_mask"][0, j]),
                    scored=bool(data["score_mask"][0, j]))
            row[label] = pair
        report["intervals"].append(row)
    local = (nu >= 4819.65) & (nu <= 4820.14)
    report["feature_window_cm1"] = [4819.65, 4820.14]
    report["residual_feature_mean"] = gaussian_description(nu[local], average[local])
    report["stellar_feature_mean"] = gaussian_description(nu[local], (1-star).mean(axis=0)[local])
    report["feature_descriptions"] = [dict(exposure=int(exposure),
        residual=gaussian_description(nu[local], residual[e, local]),
        stellar=gaussian_description(nu[local], 1-star[e, local]))
        for e, exposure in enumerate(data["exposures"])]
    report["local_shape_correlation"] = float(np.corrcoef(average[local], (1-star).mean(axis=0)[local])[0, 1])
    report["info_rv_span_kms"] = float(np.ptp(info["RV [km/s]"]))
    report["corresponding_shift_cm1"] = float(4820*report["info_rv_span_kms"]/299792.458)
    report["mean_pixel_step_cm1"] = float(np.mean(np.diff(nu)[local[:-1]]))
    report["limitations"] = [
        "A positive observed-minus-model feature can mean overpredicted absorption; co-location alone does not identify the absorber.",
        "The local PHOENIX depth and residual are similar, but their smooth baselines and convolution are not separable additive physical components.",
        "Info RV spans less than one pixel and far less than the instrumental profile; this night alone has limited stellar/telluric motion discrimination.",
        "Gaussian centroids are descriptive measurements on noisy residuals with adjacent features; they are not independent calibrated velocities.",
        "Some individual-frame Gaussian widths reach a bound, so their apparent center scatter cannot be treated as measured line motion.",
        "No masks, physical model parameters, planetary model, or adopted products are changed."]
    report["source_sha256"] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT / "feature4820_temporal.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ["residual_feature_mean", "stellar_feature_mean",
          "local_shape_correlation", "info_rv_span_kms", "corresponding_shift_cm1", "mean_pixel_step_cm1"]}, indent=2))


if __name__ == "__main__":
    main()
