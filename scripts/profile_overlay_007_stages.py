from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp

from vibespatial.api import read_file
from vibespatial.overlay.contract import contract_overlay_microcells
from vibespatial.overlay.contraction_reconstruct import (
    _coalesce_selected_microcells,
    reconstruct_overlay_from_microcells,
)
from vibespatial.overlay.microcells import (
    build_aligned_overlay_workload,
    build_and_label_overlay_microcells,
    build_overlay_microcell_bands,
    label_overlay_microcells,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "tests" / "upstream" / "geopandas" / "tests" / "data"
LEFT_NYBB = DATA_ROOT / "nybb_16a.zip"
RIGHT_NYBB = DATA_ROOT / "overlay" / "nybb_qgis" / "polydf2.shp"


def _selected_count(labels, operation: str) -> int:
    match operation:
        case "intersection":
            mask = labels.left_inside & labels.right_inside
        case "union":
            mask = labels.left_inside | labels.right_inside
        case "difference":
            mask = labels.left_inside & ~labels.right_inside
        case "symmetric_difference":
            mask = labels.left_inside ^ labels.right_inside
        case "identity":
            mask = labels.left_inside
        case _:
            raise ValueError(f"unsupported operation: {operation}")
    return int(cp.count_nonzero(mask).item())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operation",
        default="intersection",
        choices=("intersection", "union", "difference", "symmetric_difference", "identity"),
    )
    parser.add_argument("--with-contract", action="store_true")
    parser.add_argument("--skip-reconstruct", action="store_true")
    parser.add_argument("--selected-build", action="store_true")
    args = parser.parse_args()

    left = read_file(f"zip://{LEFT_NYBB}").geometry.values.to_owned()
    right = read_file(str(RIGHT_NYBB)).geometry.values.to_owned()

    t0 = time.perf_counter()
    aligned = build_aligned_overlay_workload(left, right)
    t1 = time.perf_counter()
    print(
        json.dumps({"stage": "align", "elapsed_s": t1 - t0, "aligned_rows": aligned.row_count}),
        file=sys.stderr,
        flush=True,
    )
    if args.selected_build:
        labels = build_and_label_overlay_microcells(
            aligned.left_aligned,
            aligned.right_aligned,
            selection_operation=args.operation,
        )
        cp.cuda.Stream.null.synchronize()
        t2 = time.perf_counter()
        bands = labels.bands
        t3 = t2
        print(
            json.dumps(
                {
                    "stage": "selected_build_labels",
                    "elapsed_s": t2 - t1,
                    "bands": bands.count,
                }
            ),
            file=sys.stderr,
            flush=True,
        )
    else:
        bands = build_overlay_microcell_bands(aligned.left_aligned, aligned.right_aligned)
        t2 = time.perf_counter()
        print(
            json.dumps({"stage": "bands", "elapsed_s": t2 - t1, "bands": bands.count}),
            file=sys.stderr,
            flush=True,
        )
        labels = label_overlay_microcells(bands, aligned.left_aligned, aligned.right_aligned)
        cp.cuda.Stream.null.synchronize()
        t3 = time.perf_counter()
    selected_bands = _selected_count(labels, args.operation)
    coalesced = _coalesce_selected_microcells(
        labels,
        cp.flatnonzero(
            {
                "intersection": labels.left_inside & labels.right_inside,
                "union": labels.left_inside | labels.right_inside,
                "difference": labels.left_inside & ~labels.right_inside,
                "symmetric_difference": labels.left_inside ^ labels.right_inside,
                "identity": labels.left_inside,
            }[args.operation].astype(cp.bool_, copy=False)
        ).astype(cp.int64, copy=False),
    )
    print(
        json.dumps(
            {
                "stage": "labels",
                "elapsed_s": t3 - t2,
                "labels": labels.count,
                "selected_bands": selected_bands,
                "coalesced_selected": int(cp.asarray(coalesced["row_indices"]).size),
            }
        ),
        file=sys.stderr,
        flush=True,
    )
    components = None
    t4 = t3
    if args.with_contract:
        components = contract_overlay_microcells(labels)
        t4 = time.perf_counter()
        print(
            json.dumps(
                {"stage": "contract", "elapsed_s": t4 - t3, "components": components.component_count}
            ),
            file=sys.stderr,
            flush=True,
        )
    if args.skip_reconstruct:
        result_row_count = 0
        t5 = t4
    else:
        result = reconstruct_overlay_from_microcells(
            labels,
            args.operation,
            components=components,
            row_count=aligned.row_count,
        )
        cp.cuda.Stream.null.synchronize()
        t5 = time.perf_counter()
        result_row_count = result.row_count
        print(
            json.dumps({"stage": "reconstruct", "elapsed_s": t5 - t4, "result_rows": result_row_count}),
            file=sys.stderr,
            flush=True,
        )

    payload = {
        "operation": args.operation,
        "stages": {
            "align_s": t1 - t0,
            "bands_s": t2 - t1,
            "label_s": t3 - t2,
            "contract_s": t4 - t3,
            "reconstruct_s": t5 - t4,
            "total_s": t5 - t0,
        },
        "counts": {
            "aligned_rows": aligned.row_count,
            "bands": bands.count,
            "components": None if components is None else components.component_count,
            "selected_bands": selected_bands,
            "result_rows": result_row_count,
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
