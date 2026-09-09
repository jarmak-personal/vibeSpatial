"""Reproduce physical-VRAM pressure and high-degree PBF relation canaries.

Run this script in a fresh process with no other GPU workload. The pressure
mode reserves real VRAM outside RMM before its pool grows, then validates the
complete result after releasing the reservation. Junction mode measures the
endpoint graph independently of protobuf bytes and geometry row counts.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from time import perf_counter


def pressure(args):
    import cupy as cp
    import rmm.statistics as rstats
    from benchmark_osm_pbf import _fingerprint, _sha

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.io.osm_pbf_native import read_osm_pbf_native
    from vibespatial.runtime.dispatch import clear_dispatch_events, get_dispatch_events

    baseline = json.loads(args.baseline.read_text())
    if _sha(args.path) != baseline["identity"]["source_sha256"]:
        raise ValueError("PBF differs from the immutable comparator dataset")
    get_cuda_runtime()._ensure_context()
    cp.cuda.get_current_stream().synchronize()
    free, _ = cp.cuda.runtime.memGetInfo()
    reserve = max(0, free-(args.headroom_mib << 20))
    pointer = cp.cuda.runtime.malloc(reserve) if reserve else 0
    try:
        if reserve:
            cp.cuda.runtime.memset(pointer, 0, reserve)
        cp.cuda.get_current_stream().synchronize()
        before = cp.cuda.runtime.memGetInfo()[0]
        clear_dispatch_events()
        rstats.push_statistics()
        start = perf_counter()
        try:
            result = read_osm_pbf_native(args.path, layer=args.layer)
            cp.cuda.get_current_stream().synchronize()
            elapsed = perf_counter()-start
        finally:
            counts = rstats.pop_statistics()
        packet = {
            "layer": args.layer, "physical_free_before_bytes": before,
            "reserved_bytes": reserve, "seconds": elapsed,
            "rows": result.geometry.row_count, "rmm_live_bytes": counts.current_bytes,
            "rmm_peak_bytes": counts.peak_bytes,
            "managed_events": [event.to_dict() for event in get_dispatch_events()
                               if "cuda_managed" in event.implementation],
        }
    finally:
        if pointer:
            cp.cuda.runtime.free(pointer)
    packet["fingerprint"] = _fingerprint(result)
    if packet["fingerprint"] != baseline["layers"][args.layer]["fingerprint"]:
        raise ValueError("Pressure result differs from the full GDAL oracle")
    packet["source_sha256"] = baseline["identity"]["source_sha256"]
    return packet


def junction(args):
    import cupy as cp

    from vibespatial.cuda._runtime import get_cuda_runtime
    from vibespatial.io.osm_pbf_rings import assemble_rings

    get_cuda_runtime()._ensure_context()
    packets = []
    for count in args.members:
        members = cp.zeros((count, 10), dtype=cp.int64)
        members[:, 1] = members[:, 2] = 1
        members[:, 5] = 2
        x = cp.empty(2*count, dtype=cp.float64)
        y = cp.empty_like(x)
        x[::2], y[::2] = -70, 40
        x[1::2], y[1::2] = -69+cp.arange(count)/count, 41
        offsets = cp.arange(count+1, dtype=cp.int64)*2
        relation_offsets = cp.asarray([0, count], dtype=cp.int64)
        samples = []
        for iteration in range(args.repeat+1):
            cp.cuda.get_current_stream().synchronize()
            start = perf_counter()
            ox, oy, rings, rows = assemble_rings(members, offsets, relation_offsets, x, y, 1)
            cp.cuda.get_current_stream().synchronize()
            elapsed = perf_counter()-start
            if ox.size or oy.size or rows.size or rings.size != 1:
                raise ValueError("Open star junction unexpectedly assembled a polygon")
            if iteration:
                samples.append(elapsed)
            del ox, oy, rings, rows
        packets.append({"members": count, "endpoints": 2*count,
                        "seconds": statistics.median(samples), "samples": samples})
    return {"shape": "one relation, all open ways sharing a single junction", "results": packets}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    physical = subparsers.add_parser("pressure")
    physical.add_argument("path", type=Path)
    physical.add_argument("--baseline", type=Path, required=True)
    physical.add_argument("--layer", default="multipolygons",
                          choices=("points", "lines", "multilinestrings", "multipolygons", "other_relations"))
    physical.add_argument("--headroom-mib", type=int, default=1536)
    graph = subparsers.add_parser("junction")
    graph.add_argument("--members", type=int, nargs="+", default=[10_000, 100_000, 1_000_000])
    graph.add_argument("--repeat", type=int, default=3)
    for child in (physical, graph):
        child.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "pressure" and args.headroom_mib < 32:
        parser.error("headroom must be at least 32 MiB")
    if args.mode == "junction" and (args.repeat < 1 or min(args.members) < 1):
        parser.error("repeat and member counts must be positive")
    packet = pressure(args) if args.mode == "pressure" else junction(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(packet, indent=2)+"\n")
    print(json.dumps(packet, indent=2), flush=True)


if __name__ == "__main__":
    main()
