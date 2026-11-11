"""Fetch public sibling nights for a wavelength-motion diagnostic."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
FILES = {"cube06v3.pic": "fa338b902cfa5127fa4707187c2d00f0",
         "cube21v3.pic": "53f3c67601475f0f5ffc8d1b9277c304"}


def main():
    records = []
    for name, expected in FILES.items():
        path = ROOT/"data"/name
        url = f"https://zenodo.org/records/10382053/files/{name}?download=1"
        if not path.exists():
            partial = path.with_suffix(path.suffix+".partial")
            with urlopen(url, timeout=180) as source, partial.open("wb") as stream:
                for chunk in iter(lambda: source.read(1024*1024), b""):
                    stream.write(chunk)
            if hashlib.md5(partial.read_bytes()).hexdigest() != expected:
                raise ValueError(f"Download MD5 mismatch for {name}")
            partial.replace(path)
        raw = path.read_bytes()
        md5 = hashlib.md5(raw).hexdigest()
        if md5 != expected:
            raise ValueError(f"Existing MD5 mismatch for {name}; file not changed")
        records.append(dict(file=name, url=url, bytes=len(raw), md5=md5,
                            sha256=hashlib.sha256(raw).hexdigest(),
                            verified_utc=datetime.now(timezone.utc).isoformat()))
        print(name, len(raw), md5, flush=True)
    (ROOT/"results/crossnight_downloads.json").write_text(json.dumps(records, indent=2)+"\n")


if __name__ == "__main__":
    main()
