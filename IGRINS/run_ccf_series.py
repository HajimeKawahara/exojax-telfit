"""Apply completed shared fits to all observed frames without planet injection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from queue import Queue
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def shared_fit_ready(stem):
    """Wait until the metadata and completed NPZ describe the same product."""
    try:
        metadata = json.loads(stem.with_suffix(".json").read_text())
        product_hash = hashlib.sha256(stem.with_suffix(".npz").read_bytes()).hexdigest()
        return metadata["product_sha256"] == product_hash
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(args):
    orders = list(range(44)) if args.orders is None else args.orders
    if len(orders) != len(set(orders)) or any(order not in range(44) for order in orders):
        raise ValueError("Use unique released cube indices in 0..43")
    cpus = sorted(os.sched_getaffinity(0))
    if min(args.jobs, args.cpus_per_job) < 1 or args.jobs*args.cpus_per_job > len(cpus):
        raise ValueError("Requested CPU allocation exceeds the available CPUs")
    OUT.mkdir(parents=True, exist_ok=True)
    state = dict(started_utc=stamp(), artificial_injection=False,
                 requested_cube_indices=orders,
                 orders={str(order): dict(stage="waiting_for_shared_fit") for order in orders})
    status_path = OUT / "allorders_ccf_run_status.json"
    allocations = Queue()
    for index in range(args.jobs):
        allocations.put(cpus[index*args.cpus_per_job:(index+1)*args.cpus_per_job])

    def save():
        state["updated_utc"] = stamp()
        temporary = status_path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(state, indent=2)+"\n")
        temporary.replace(status_path)

    def apply(order):
        allocation = allocations.get()
        tag = f"allorders_o{order:02d}"
        command = ["taskset", "-c", ",".join(map(str, allocation)), sys.executable,
                   str(ROOT/"apply_fiducial_series.py"), "--order", str(order)]
        if args.overwrite:
            command.append("--overwrite")
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
                   JAX_PLATFORMS="cpu", JAX_ENABLE_X64="true",
                   MPLCONFIGDIR=f"/tmp/igrins-ccf-matplotlib-o{order:02d}")
        try:
            with (OUT/f"{tag}_ccf_application.log").open("w") as stream:
                code = subprocess.run(command, env=env, stdout=stream,
                                      stderr=subprocess.STDOUT).returncode
            return dict(stage="complete" if code == 0 else "failed", returncode=code,
                        log=f"{tag}_ccf_application.log", finished_utc=stamp())
        finally:
            allocations.put(allocation)

    deadline = time.monotonic()+args.wait_for_fits
    pending = set(orders)
    running = {}
    save()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        while pending or running:
            for future, order in list(running.items()):
                if future.done():
                    try:
                        state["orders"][str(order)].update(future.result())
                    except Exception as error:
                        state["orders"][str(order)].update(stage="failed", message=str(error))
                    del running[future]
                    print(json.dumps(dict(order=order, **state["orders"][str(order)])), flush=True)
            for order in sorted(pending):
                if len(running) >= args.jobs:
                    break
                stem = OUT/f"allorders_o{order:02d}_stellar_fit"
                if shared_fit_ready(stem):
                    pending.remove(order)
                    state["orders"][str(order)].update(stage="applying", started_utc=stamp())
                    running[pool.submit(apply, order)] = order
            if pending and not running and time.monotonic() >= deadline:
                for order in pending:
                    state["orders"][str(order)].update(stage="missing_shared_fit")
                pending.clear()
            save()
            if pending or running:
                time.sleep(5)
    state["finished_utc"] = stamp()
    save()
    if any(row["stage"] != "complete" for row in state["orders"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--cpus-per-job", type=int, default=8)
    parser.add_argument("--wait-for-fits", type=float, default=0,
                        help="Maximum seconds to wait for missing shared fits")
    parser.add_argument("--overwrite", action="store_true")
    main(parser.parse_args())
