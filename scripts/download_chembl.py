#!/usr/bin/env python
"""Download and unpack a ChEMBL SQLite release.

The script prefers robust command line downloaders when available because the
SQLite archive is several GB. It records SHA256 locally even when upstream only
publishes MD5/checksum files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases"


def sha256_file(path: Path, block_size: int = 1024 * 1024 * 32) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_small(url: str, output: Path) -> bool:
    try:
        with urlopen(url, timeout=30) as response:
            output.write_bytes(response.read())
        return True
    except (HTTPError, URLError, TimeoutError) as exc:
        print(f"[warn] optional download failed: {url} ({exc})", file=sys.stderr)
        return False


def run_downloader(url: str, output_dir: Path, filename: str, connections: int) -> None:
    output = output_dir / filename
    if output.exists() and output.stat().st_size > 0:
        print(f"[download] found existing {output}, resume/checksum will reuse it")

    aria2c = shutil.which("aria2c")
    wget = shutil.which("wget")
    curl = shutil.which("curl")

    if aria2c:
        cmd = [
            aria2c,
            "--continue=true",
            f"--max-connection-per-server={connections}",
            f"--split={connections}",
            "--min-split-size=16M",
            "--retry-wait=10",
            "--max-tries=0",
            "--summary-interval=60",
            "--dir",
            str(output_dir),
            "--out",
            filename,
            url,
        ]
    elif wget:
        cmd = [wget, "-c", "--tries=0", "--timeout=30", "-O", str(output), url]
    elif curl:
        cmd = [curl, "-L", "-C", "-", "--retry", "999", "--retry-delay", "10", "-o", str(output), url]
    else:
        raise RuntimeError("Need one of aria2c, wget, or curl to download ChEMBL reliably.")

    print("[download]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def unpack_sqlite(archive: Path, output_dir: Path) -> Path:
    candidates = sorted(output_dir.rglob("*.db")) + sorted(output_dir.rglob("*.sqlite"))
    if candidates:
        print(f"[extract] existing sqlite detected: {candidates[0]}")
        return candidates[0]

    print(f"[extract] unpacking {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(output_dir)

    candidates = sorted(output_dir.rglob("*.db")) + sorted(output_dir.rglob("*.sqlite"))
    if not candidates:
        raise FileNotFoundError(f"No SQLite database found after extracting {archive}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="chembl_37")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-dir", default="data/raw/chembl_37")
    parser.add_argument("--connections", type=int, default=16)
    parser.add_argument("--no-extract", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    release_url = f"{args.base_url.rstrip('/')}/{args.release}"
    sqlite_name = f"{args.release}_sqlite.tar.gz"
    sqlite_url = f"{release_url}/{sqlite_name}"
    archive = output_dir / sqlite_name

    for optional_name in ("checksums.txt", "README", f"{args.release}_release_notes.txt"):
        fetch_small(f"{release_url}/{optional_name}", output_dir / optional_name)

    run_downloader(sqlite_url, output_dir, sqlite_name, args.connections)
    archive_sha256 = sha256_file(archive)

    sqlite_path = None
    if not args.no_extract:
        sqlite_path = unpack_sqlite(archive, output_dir)

    metadata = {
        "release": args.release,
        "url": sqlite_url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive),
        "archive_size_bytes": archive.stat().st_size,
        "archive_sha256": archive_sha256,
        "sqlite_path": str(sqlite_path) if sqlite_path else None,
    }
    metadata_path = output_dir / "download_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
