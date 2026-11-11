"""Cache public HITRAN SDVoigt and first-order line-mixing parameters."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import hapi
import numpy as np

ROOT = Path(__file__).resolve().parent
PARAMETERS = ["par_line", "gamma_SDV_0_air_296", "n_SDV_air_296",
              "gamma_SDV_2_air_296", "n_gamma_SDV_2_air_296",
              "delta_SDV_0_air_296", "deltap_SDV_air_296",
              "Y_SDV_air_296", "n_Y_SDV_air_296", "SD_air", "deltap_air",
              "y_air"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", nargs="+", type=int, default=[13,34,40])
    parser.add_argument("--species", nargs="+", default=["H2O","CO2"])
    parser.add_argument("--padding", type=float, default=9.)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()
    with np.load(ROOT / "results/focus_ip.npz") as data:
        orders, wavelengths = data["orders"], data["wavelength_um"]
    for species, molecule in [("H2O", 1), ("CO2", 2)]:
        if species not in args.species:
            continue
        isotopes = [str(value[0]) for key,value in hapi.ISO.items() if key[0] == molecule]
        for order, wave in zip(orders, wavelengths):
            if order not in args.orders:
                continue
            path = ROOT / "cache" / f"advanced_{species}_o{order}{args.suffix}.csv"
            numin = float(np.floor(np.min(1e4/wave)-args.padding))
            numax = float(np.ceil(np.max(1e4/wave)+args.padding))
            url = ("https://hitran.org/lbl/api?iso_ids_list=" + ",".join(isotopes)
                   + f"&numin={numin}&numax={numax}&head=False&fixwidth=0&sep=[comma]&request_params="
                   + ",".join(PARAMETERS))
            if not path.exists():
                with urlopen(url, timeout=120) as response:
                    raw = b"".join(iter(lambda:response.read(1024*1024), b""))
                lines = raw.decode().splitlines()
                if not lines or any(len(line.split(",")) != len(PARAMETERS) for line in lines):
                    raise ValueError("Unexpected advanced-parameter response")
                path.write_bytes(raw)
                path.with_suffix(".json").write_text(json.dumps(dict(
                    url=url, parameters=PARAMETERS, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    note="HITRANonline live API, no pinned edition; # denotes an unavailable parameter"), indent=2)+"\n")
            print(species, int(order), len(path.read_text().splitlines()), flush=True)


if __name__ == "__main__":
    main()
