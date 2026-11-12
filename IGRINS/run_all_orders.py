"""Apply the Order 7 fiducial fitting procedure to all released cube orders."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import pickle
from queue import Queue
import shutil
import subprocess
import sys
from threading import Lock
import time
import zipfile
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TRAIN = [0, 19, 39, 59, 78]
EVALUATION = [10, 30, 50, 70]
SPECIES = ["H2O", "CH4", "N2O", "CO", "CO2"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stamp():
    return datetime.now(timezone.utc).isoformat()


def invalid_cache_returns_false(check):
    """Treat incomplete cache files as work to recompute on resume."""
    @wraps(check)
    def validate(*args, **kwargs):
        try:
            return check(*args, **kwargs)
        except (OSError, ValueError, KeyError, TypeError, AttributeError,
                EOFError, zipfile.BadZipFile, zlib.error):
            return False
    return validate


@invalid_cache_returns_false
def valid_physical(tag, order, exposure, continuum_scope="calibration", continuum_degree=3):
    """Reuse only complete, successful products in the requested configuration."""
    path = OUT / f"{tag}.json"
    if not path.exists() or not path.with_suffix(".npz").exists():
        return False
    meta = json.loads(path.read_text())
    if meta.get("continuum_scope", "calibration") != continuum_scope:
        return False
    if meta.get("continuum_degree", 3) != continuum_degree:
        return False
    if not (meta["training_fit"]["success"] and meta["heldout_calibration_fit"]["success"]):
        return False
    arguments = meta["provenance"]["command_arguments"]
    expected = dict(orders=[order], pixels=[0, 1848], train=TRAIN, holdout=exposure,
                    atmosphere="profile", layers=20, dv=.75, shift=True,
                    ip="mixture", ip_epoch=True, species=SPECIES, fix_star=True)
    if any(arguments.get(key) != value for key, value in expected.items()):
        return False
    with np.load(path.with_suffix(".npz")) as data:
        return (data["orders"].tolist() == [order]
                and data["exposures"].tolist() == TRAIN+[exposure]
                and data["included_mask"].all()
                and np.isfinite(data["model"]).all()
                and np.all(data["continuum"] > 0))


@invalid_cache_returns_false
def valid_stellar(tag, order, continuum_scope="calibration", continuum_degree=3,
                  allow_reference=True):
    path = OUT / f"{tag}_stellar_fit.json"
    if not path.exists() or not path.with_suffix(".npz").exists():
        return False
    meta = json.loads(path.read_text())
    if meta.get("reference_reused") and not allow_reference:
        return False
    prefix = meta.get("physical_prefix", "order7_physical_e" if meta.get("reference_reused")
                      else f"{tag}_physical_e")
    for exposure in EVALUATION:
        physical_path = OUT/f"{prefix}{exposure}.json"
        if not physical_path.exists():
            return False
        physical_metadata = json.loads(physical_path.read_text())
        if physical_metadata.get("continuum_scope", "calibration") != continuum_scope:
            return False
        if physical_metadata.get("continuum_degree", 3) != continuum_degree:
            return False
    for name, expected_hash in meta.get("source_sha256", {}).items():
        if name.startswith("results/"):
            source = ROOT/name
            if not source.exists() or sha256(source) != expected_hash:
                return False
    if meta["product_sha256"] != sha256(path.with_suffix(".npz")):
        return False
    if not all(row["success"] for values in meta["fits"].values() for row in values):
        return False
    with np.load(path.with_suffix(".npz")) as data:
        return (data["orders"].tolist() == [order]
                and data["exposures"].tolist() == TRAIN+EVALUATION
                and data["included_mask"].shape == (1, 1848)
                and data["included_mask"].all()
                and np.isfinite(data["residual_filtered"]).all())


def reuse_reference():
    """Keep the previously fitted Order 7 result byte-identical in the comparison."""
    source = OUT / "order7_stellar_fit.npz"
    metadata = source.with_suffix(".json")
    if not source.exists() or not metadata.exists():
        return False
    reference = json.loads(metadata.read_text())
    assert reference["product_sha256"] == sha256(source)
    with np.load(source) as data:
        assert data["orders"].tolist() == [5]
        assert data["included_mask"].all()
    destination = OUT / "allorders_o05_stellar_fit.npz"
    temporary = destination.with_suffix(".tmp.npz")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    reference["reference_reused"] = True
    reference["reference_source"] = "results/order7_stellar_fit.npz"
    reference["reference_note"] = "Frozen original fiducial result; not refitted in this batch."
    temporary = destination.with_suffix(".tmp.json")
    temporary.write_text(json.dumps(reference, indent=2)+"\n")
    temporary.replace(destination.with_suffix(".json"))
    return valid_stellar("allorders_o05", 5)


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "data/cube14v3.pic").open("rb") as stream:
        wavelength, flux = pickle.load(stream)
    orders = list(range(len(wavelength))) if args.orders is None else args.orders
    if len(orders) != len(set(orders)) or any(order < 0 or order >= len(wavelength) for order in orders):
        raise ValueError("Use unique released cube indices in 0..43.")
    if set(args.degree_four_continuum_orders) & set(args.degree_five_continuum_orders):
        raise ValueError("An order cannot have two continuum degrees")
    cpus = sorted(os.sched_getaffinity(0))
    if args.jobs < 1 or args.cpus_per_job < 1 or args.jobs*args.cpus_per_job > len(cpus):
        raise ValueError("Requested CPU allocation exceeds this process's allowed CPUs.")
    if not np.isfinite(flux[orders]).all():
        raise ValueError("Non-finite data require an explicit policy; no silent masking.")
    path = args.status_file
    path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(path.read_text()) if path.exists() else dict(created_utc=stamp(), orders={})
    state["config"] = dict(requested_cube_indices=orders, training_frames=TRAIN,
                          evaluation_frames=EVALUATION, retained_pixels_per_order=1848,
                          model="Order 7 fiducial procedure, fitted independently per order",
                          atmosphere_layers=20, species=SPECIES, dv_kms=.75,
                          stellar="PHOENIX times blasé amplitude ratio; fixed reference kinematics",
                          ip="Symmetric two-Gaussian mixture, epoch widths",
                          line_strengths="HITRAN finite-uncertainty bins with weak-line suppression",
                          smooth="Degree-five residual projection on all pixels",
                          negative_observed_flux="Retained in the least-squares likelihood",
                          new_absorption_or_residual_masks=False, enuma_used=False,
                          jobs=args.jobs, cpus_per_job=args.cpus_per_job,
                          initial_config=str(args.initial), null_refits=args.null_refits)
    state["config"]["all_pixel_continuum_orders"] = args.all_pixel_continuum_orders
    state["config"]["degree_five_continuum_orders"] = args.degree_five_continuum_orders
    state["config"]["degree_four_continuum_orders"] = args.degree_four_continuum_orders
    lock = Lock()

    def update(order, **changes):
        with lock:
            row = state["orders"].setdefault(str(order), {})
            row.update(changes, updated_utc=stamp())
            state["updated_utc"] = stamp()
            temporary = path.with_suffix(".tmp.json")
            temporary.write_text(json.dumps(state, indent=2)+"\n")
            temporary.replace(path)
        print(json.dumps(dict(order=order, **changes)), flush=True)

    for order in orders:
        update(order, wavelength_um=[float(wavelength[order].min()), float(wavelength[order].max())],
               nonpositive_observed_samples=int(np.count_nonzero(flux[order, TRAIN+EVALUATION] <= 0)),
               stage="queued")
    if 5 in orders and args.reuse_order7 and reuse_reference():
        update(5, stage="complete", reused_reference=True,
               product="allorders_o05_stellar_fit.npz")
    allocations = Queue()
    for index in range(args.jobs):
        allocations.put(cpus[index*args.cpus_per_job:(index+1)*args.cpus_per_job])

    def run(order):
        tag = f"allorders_o{order:02d}"
        continuum_scope = "all" if order in args.all_pixel_continuum_orders else "calibration"
        continuum_degree = 5 if order in args.degree_five_continuum_orders else 3
        if order in args.degree_four_continuum_orders:
            continuum_degree = 4
        if valid_stellar(tag, order, continuum_scope, continuum_degree,
                         allow_reference=args.reuse_order7):
            update(order, stage="complete", reused_existing=True, product=f"{tag}_stellar_fit.npz")
            return True
        allocation = allocations.get()
        started = time.monotonic()
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                   JAX_ENABLE_X64="true", JAX_PLATFORMS="cpu",
                   MPLCONFIGDIR=f"/tmp/igrins-matplotlib-o{order:02d}")

        def command(arguments, logname):
            full = ["taskset", "-c", ",".join(map(str, allocation)), sys.executable, *map(str, arguments)]
            update(order, command=full, log=logname)
            with (OUT/logname).open("w") as output:
                return subprocess.run(full, env=env, stdout=output, stderr=subprocess.STDOUT).returncode

        try:
            common = [ROOT/"fit_joint.py", "--orders", order, "--pixels", 0, 1848,
                      "--train", *TRAIN, "--atmosphere", "profile", "--layers", 20,
                      "--dv", .75, "--shift", "--ip", "mixture", "--ip-epoch",
                      "--species", *SPECIES, "--fix-star"]
            if continuum_scope == "all":
                common += ["--continuum-scope", "all"]
            if continuum_degree != 3:
                common += ["--continuum-degree", continuum_degree]
            for exposure in EVALUATION:
                physical_tag = f"{tag}_physical_e{exposure}"
                if valid_physical(physical_tag, order, exposure, continuum_scope, continuum_degree):
                    continue
                update(order, stage="physical", exposure=exposure)
                initial = args.initial if exposure == EVALUATION[0] else OUT/f"{tag}_physical_e10.json"
                for attempt in range(1, 4):
                    arguments = [*common, "--holdout", exposure, "--initial", initial,
                                 "--tag", physical_tag, "--max-nfev", args.physical_max_nfev*attempt]
                    if exposure != EVALUATION[0]:
                        arguments += ["--freeze-shared"]
                    code = command(arguments, f"{physical_tag}_attempt{attempt}.log")
                    if code == 0 and valid_physical(physical_tag, order, exposure, continuum_scope, continuum_degree):
                        break
                    saved = OUT/f"{physical_tag}.json"
                    if saved.exists() and exposure == EVALUATION[0]:
                        initial = saved
                else:
                    raise RuntimeError(f"Physical fit failed for frame {exposure}; see recorded logs.")
            if args.stage == "physical":
                update(order, stage="physical_complete", seconds=time.monotonic()-started)
                return True
            for attempt in range(1, args.stellar_attempts+1):
                update(order, stage="stellar", attempt=attempt)
                arguments = [ROOT/"fit_order_stellar.py", "--order", order, "--tag", tag,
                             "--strength-solver", "lbfgsb", "--resume",
                             "--max-nfev", args.stellar_max_nfev, "--null-refits", args.null_refits]
                code = command(arguments, f"{tag}_stellar_attempt{attempt}.log")
                if code == 0 and valid_stellar(tag, order, continuum_scope, continuum_degree,
                                                allow_reference=args.reuse_order7):
                    update(order, stage="complete", seconds=time.monotonic()-started,
                           product=f"{tag}_stellar_fit.npz")
                    return True
            raise RuntimeError("Stellar/strength fit did not complete; see recorded logs.")
        except Exception as error:
            update(order, stage="failed", message=str(error), seconds=time.monotonic()-started)
            return False
        finally:
            allocations.put(allocation)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(run, order): order for order in orders}
        results = [future.result() for future in as_completed(futures)]
    print(json.dumps(dict(finished_utc=stamp(), requested=len(orders), successful=sum(results),
                          failed=len(results)-sum(results), status_file=str(path))), flush=True)
    if not all(results):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int)
    parser.add_argument("--status-file", type=Path, default=OUT/"allorders_run_status.json")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--cpus-per-job", type=int, default=8)
    parser.add_argument("--initial", type=Path, default=ROOT.parent/"config/order7_initial.json")
    parser.add_argument("--physical-max-nfev", type=int, default=150)
    parser.add_argument("--stellar-max-nfev", type=int, default=200)
    parser.add_argument("--stellar-attempts", type=int, default=6)
    parser.add_argument("--null-refits", type=int, default=8)
    parser.add_argument("--all-pixel-continuum-orders", nargs="*", type=int, default=[],
                        help="Explicit order indices whose held-out cubic continuum uses all pixels")
    parser.add_argument("--degree-five-continuum-orders", nargs="*", type=int, default=[],
                        help="Explicit order indices using a degree-five multiplicative continuum")
    parser.add_argument("--degree-four-continuum-orders", nargs="*", type=int, default=[],
                        help="Explicit order indices using a degree-four multiplicative continuum")
    parser.add_argument("--reuse-order7", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage", choices=["all", "physical"], default="all")
    main(parser.parse_args())
