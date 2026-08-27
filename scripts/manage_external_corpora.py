#!/usr/bin/env python3
"""Fetch and verify immutable external-corpus benchmark assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "benchmarks/shootout/corpora/vsbench-workload.json"


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("assets"), dict):
        raise ValueError(f"unsupported external-corpus manifest: {path}")
    return payload


def _data_root(manifest: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    env = os.environ.get(manifest["data_root_env"])
    if env:
        return Path(env).expanduser().resolve()
    return (REPO_ROOT / manifest["default_data_root"]).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_one(key: str, asset: dict[str, Any], root: Path) -> tuple[bool, str]:
    path = root / asset["local_path"]
    if not path.is_file():
        return False, f"missing {path}"
    if path.stat().st_size != asset["size_bytes"]:
        return False, f"size mismatch {path}"
    actual = _sha256(path)
    if actual != asset["sha256"]:
        return False, f"SHA-256 mismatch {path}: {actual}"
    return True, f"ok {key} {path}"


def _fetch_one(key: str, asset: dict[str, Any], root: Path) -> None:
    valid, message = _verify_one(key, asset, root)
    if valid:
        print(message)
        return
    target = root / asset["local_path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        asset["url"],
        headers={"User-Agent": "vibeSpatial-external-corpus/1"},
    )
    digest = hashlib.sha256()
    size = 0
    with (
        urllib.request.urlopen(request) as response,
        tempfile.NamedTemporaryFile(
            dir=target.parent,
            prefix=f".{target.name}.",
            delete=False,
        ) as output,
    ):
        temporary = Path(output.name)
        try:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    if size != asset["size_bytes"] or digest.hexdigest() != asset["sha256"]:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"download identity mismatch for {key}: size={size}, sha256={digest.hexdigest()}"
        )
    temporary.replace(target)
    print(f"fetched {key} {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "fetch", "verify"))
    parser.add_argument("assets", nargs="*", help="asset keys; defaults to all")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path)
    args = parser.parse_args(argv)

    manifest = _load_manifest(args.manifest)
    root = _data_root(manifest, args.data_root)
    unknown = sorted(set(args.assets) - set(manifest["assets"]))
    if unknown:
        parser.error(f"unknown assets: {', '.join(unknown)}")
    keys = args.assets or list(manifest["assets"])

    if args.command == "list":
        for key in keys:
            asset = manifest["assets"][key]
            print(f"{key}\t{asset['size_bytes']}\t{asset['license']}\t{asset['shape']}")
        return 0
    if args.command == "fetch":
        for key in keys:
            _fetch_one(key, manifest["assets"][key], root)
        return 0

    failed = False
    for key in keys:
        valid, message = _verify_one(key, manifest["assets"][key], root)
        print(message)
        failed = failed or not valid
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
