from __future__ import annotations

import argparse
import json

from vibespatial.benchmark_fixtures import ensure_fixture, list_fixture_specs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate GeoParquet benchmark fixtures in .benchmark_fixtures/.")
    parser.add_argument("--fixture", action="append", default=[], help="Fixture name to generate. May be repeated.")
    parser.add_argument("--all-default", action="store_true", help="Generate all default fixtures.")
    parser.add_argument("--force", action="store_true", help="Rebuild fixtures even when they already exist.")
    parser.add_argument("--list", action="store_true", help="List known fixture names and exit.")
    args = parser.parse_args(argv)

    specs = list_fixture_specs()
    if args.list:
        print(json.dumps([spec.name for spec in specs], indent=2))
        return 0

    requested = args.fixture or ([] if not args.all_default else [spec.name for spec in specs])
    if not requested:
        parser.error("select --fixture NAME or --all-default")

    payload = []
    for name in requested:
        path = ensure_fixture(name, force=args.force)
        payload.append({"fixture": name, "path": str(path)})
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
