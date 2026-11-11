"""Download restricted HITRANonline line lists and cache TIPS partition sums."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen

import hapi
import numpy as np


CACHE = Path(__file__).resolve().parent / "cache"
MOLECULES = {"H2O": 1, "CO2": 2, "CH4": 6, "N2O": 4, "CO": 5}
FIELDS = {
    "nu": (3, 15), "sw": (15, 25), "a": (25, 35),
    "gamma_air": (35, 40), "gamma_self": (40, 45),
    "elower": (45, 55), "n_air": (55, 59), "delta_air": (59, 67),
}


def prepare(name: str, molecule: int) -> None:
    """Keep raw server bytes and numerical arrays with download provenance."""
    CACHE.mkdir(exist_ok=True)
    isotopes = [value[0] for key, value in hapi.ISO.items() if key[0] == molecule]
    url = ("https://hitran.org/lbl/api?iso_ids_list="
           + ",".join(map(str, isotopes)) + "&numin=4050&numax=6900")
    raw_path = CACHE / f"{name}.par"
    if not raw_path.exists():
        with urlopen(url, timeout=180) as response:
            # The endpoint can report CRLF sizes while sending LF records.
            raw = b"".join(iter(lambda: response.read(1024 * 1024), b""))
        lines = raw.decode("ascii").splitlines()
        if not lines or any(len(line) != 160 for line in lines):
            raise ValueError(f"Invalid HITRAN fixed-width response for {name}")
        raw_path.write_bytes(raw)
        provenance = {
            "url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "edition": "HITRAN2024 (live HITRANonline; endpoint has no version pin)",
            "edition_reference": "https://hitran.org/docs/hitran-papers/",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "wavenumber_range_cm-1": [4050, 6900],
            "partition_sums": "HAPI 1.2.2.3 partitionSum default TIPS-2021",
        }
        (CACHE / f"{name}.json").write_text(json.dumps(provenance, indent=2) + "\n")
    lines = raw_path.read_text().splitlines()
    data = {key: np.array([float(line[start:end]) for line in lines])
            for key, (start, end) in FIELDS.items()}
    # HITRAN uses hexadecimal-like A/B labels for isotopes beyond ten.
    data["iso"] = np.array([int(line[2]) if line[2].isdigit() and line[2] != "0"
                            else 10 if line[2] == "0" else ord(line[2]) - ord("A") + 11
                            for line in lines])
    data["line_mass"] = np.array([hapi.molecularMass(molecule, iso)
                                  for iso in data["iso"]])
    temperatures = np.arange(150.0, 401.0)
    partition_isotopes = np.unique(data["iso"])
    partition_sums = np.array([hapi.partitionSum(molecule, int(iso), temperatures.tolist())
                              for iso in partition_isotopes])
    np.savez_compressed(CACHE / f"{name}.npz", **data,
                        temperatures=temperatures,
                        partition_isotopes=partition_isotopes,
                        partition_sums=partition_sums)
    print(f"{name}: {len(lines)} lines, isotopes {partition_isotopes.tolist()}", flush=True)


if __name__ == "__main__":
    for name, molecule in MOLECULES.items():
        prepare(name, molecule)
