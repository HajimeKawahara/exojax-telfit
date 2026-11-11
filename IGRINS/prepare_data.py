"""Download and verify the two public inputs for the Order 7 analysis."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import tempfile

import requests


ROOT = Path(__file__).resolve().parent
BASE = "https://zenodo.org/records/10382053/files/"
FILES = {
    "cube14v3.pic": {
        "bytes": 77591894,
        "sha256": "999623a7ba4460aa72ba9bbe284de92e7b5a811ebe09e51d70667084ae592b26",
    },
    "20201214_info.csv": {
        "bytes": 7261,
        "sha256": "68f5a80f930e5f1f99cd1861678754f824bf665972d675f617a7d66c4fdad11e",
    },
}


def matches(path: Path, expected: dict) -> bool:
    """Verify size and SHA256 without loading the observational cube into RAM."""
    if not path.is_file() or path.stat().st_size != expected["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == expected["sha256"]


def prepare(data_dir: Path, verify_only: bool = False) -> None:
    """Skip valid inputs and replace other files only after a verified download."""
    for name, expected in FILES.items():
        destination = data_dir / name
        if matches(destination, expected):
            print(f"Verified existing file: {destination}", flush=True)
            continue
        if verify_only:
            raise ValueError(f"Missing or invalid input: {destination}")
        data_dir.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=data_dir, prefix=name + ".",
                                             suffix=".part", delete=False) as stream:
                temporary = Path(stream.name)
                with requests.get(BASE + name + "?download=1", stream=True,
                                  timeout=(30, 180)) as response:
                    response.raise_for_status()
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if not matches(temporary, expected):
                raise ValueError(f"Downloaded file failed size/SHA256 verification: {name}")
            temporary.replace(destination)
            print(f"Downloaded and verified: {destination}", flush=True)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    prepare(arguments.data_dir, arguments.verify_only)
