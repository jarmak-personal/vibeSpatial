"""Complex-geometry shape canaries with reusable same-WKB Shapely evidence."""

from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path

from benchmark_osm_nearest import canonical, compare, sha, write_json
from experiment_generalized_strtree import GeneralizedSTRtree
from experiment_nearest_hierarchy import timed
from experiment_nearest_strategies import experiment_environment
from experiment_segment_bvh import SegmentBVHRefiner


def fixture(case):
    import numpy as np
    import shapely

    def polygon(vertices,centre,radius=10.,holes=False):
        theta=np.linspace(0,2*np.pi,vertices,endpoint=False)
        shell=np.stack((np.cos(theta),np.sin(theta)),axis=1)*radius+centre
        inner=[]
        if holes:
            angle=np.linspace(0,2*np.pi,64,endpoint=False)
            circle=np.stack((np.cos(angle),np.sin(angle)),axis=1)*.4
            inner=[circle+centre+[x,y] for x in (-3.,-1.,1.,3.) for y in (-3.,-1.,1.,3.)]
        return shapely.Polygon(shell,holes=inner)

    # Host fixture generation is outside every measured boundary.
    if case == "single":
        return [polygon(16384,[25.,0.])],[polygon(16384,[0.,0.])]
    if case == "skew":
        centres=np.stack((np.arange(128)*100.,np.full(128,1000.)),axis=1)
        centres[0]=0.
        return ([polygon(16384 if i==0 else 16,c+[25.,0.]) for i,c in enumerate(centres)],
                [polygon(16384 if i==0 else 16,c) for i,c in enumerate(centres)])
    if case == "holes":
        centres=np.stack((np.arange(256)*100.,np.zeros(256)),axis=1)
        return ([polygon(128,c+[25.,0.],holes=True) for c in centres],
                [polygon(128,c,holes=True) for c in centres])
    rng=np.random.default_rng(792)
    return ([polygon(256,c,radius=100.) for c in rng.uniform(-10,10,(128,2))],
            [polygon(256,c,radius=100.) for c in rng.uniform(-10,10,(128,2))])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("single", "skew", "holes", "overlap"), required=True)
    parser.add_argument("--tile-segments", type=int, default=0)
    parser.add_argument("--seed-first", action="store_true")
    parser.add_argument("--public-sindex", action="store_true")
    parser.add_argument("--reuse-comparator", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output path")
    import cupy as cp
    import numpy as np
    import shapely

    import vibespatial as vs

    vs.set_execution_mode("gpu")
    t, q = fixture(args.case)
    twkb, qwkb = shapely.to_wkb(t), shapely.to_wkb(q)
    h = hashlib.sha256()
    for value in (*twkb, *qwkb):
        h.update(len(value).to_bytes(8, "little"))
        h.update(value)
    env = experiment_environment()
    identity = dict(
        case=args.case,
        fixture_sha256=h.hexdigest(),
        generator_sha256=hashlib.sha256(inspect.getsource(fixture).encode()).hexdigest(),
        environment={
            k: env[k] for k in ("python", "host", "platform", "packages", "geos", "lock_sha256")
        },
        boundary="One first and one fresh-input trial. Same WKB ingress/build/first/reused host output with synchronization; fixture/import/oracle work excluded.",
    )
    sources = [
        Path(__file__),
        Path("scripts/experiment_generalized_strtree.py"),
        Path("scripts/experiment_segment_bvh.py"),
        Path("scripts/experimental_segment_bvh_kernels.py"),
        Path("src/vibespatial/spatial/segment_distance_kernels.py"),
        Path("src/vibespatial/spatial/segment_primitives.py"),
    ]
    sources.extend(Path("src/vibespatial/kernels/spatial").glob("*.py"))
    sources.extend(Path(p) for p in ("src/vibespatial/api/sindex.py", "src/vibespatial/api/_native_metadata.py", "src/vibespatial/cuda/_runtime.py", "src/vibespatial/cuda/device_functions/segment_distance.py",
                "src/vibespatial/cuda/device_functions/point_segment_distance.py",
                "src/vibespatial/cuda/cccl_primitives.py",
                "src/vibespatial/runtime/cccl_warmup_specs.py",
                "src/vibespatial/predicates/binary.py",
                "src/vibespatial/predicates/polygon.py",
                "src/vibespatial/predicates/point_relations.py",
                "src/vibespatial/spatial/spatial_index_device.py", "src/vibespatial/spatial/nearest.py", "src/vibespatial/spatial/point_distance_kernels.py", "scripts/experimental_strtree_kernels.py", "scripts/experimental_strtree_bounds_kernels.py", "scripts/experimental_strtree_bounds.py"))
    report = dict(
        status="running",
        public_sindex=args.public_sindex,
        identity=identity,
        tile_segments=args.tile_segments,
        seed_first=args.seed_first,
        environment=env,
        source_sha256={str(p): sha(p) for p in sources},
        cases={},
    )
    for p in sources:
        args.output.with_name(args.output.stem + "." + p.name).write_text(p.read_text())
    expected = None
    if args.reuse_comparator:
        previous = json.loads(args.reuse_comparator.read_text())
        assert previous["identity"] == identity
        assert previous["status"] == "passed"
        op = Path(previous["oracle_path"])
        assert sha(op) == previous["oracle_sha256"]
        with np.load(op) as o:
            expected = o["indices"], o["distances"]
        report.update(
            oracle_path=str(op), oracle_sha256=sha(op), reused_comparator=str(args.reuse_comparator)
        )
        report["cases"]["shapely"] = previous["cases"]["shapely"]
    for backend in ("gpu",) if expected is not None else ("shapely", "gpu"):
        gpu = backend == "gpu"
        trials = []
        report["cases"][backend] = trials
        for _ in range(2):
            record = {}
            trials.append(record)
            ctor = vs.GeoSeries.from_wkb if gpu else shapely.from_wkb
            (tree, query), record["ingress_s"] = timed(lambda ctor=ctor: (ctor(twkb), ctor(qwkb)), gpu=gpu)
            factory = functools.partial(SegmentBVHRefiner, tile_segments=args.tile_segments)
            index, record["build_s"] = timed(
                lambda tree=tree, factory=factory, gpu=gpu: (
                    (tree.sindex if args.public_sindex else GeneralizedSTRtree(tree, refiner_factory=factory, seed_first=args.seed_first))
                    if gpu
                    else shapely.STRtree(tree)
                ),
                gpu=gpu,
            )
            for stage in ("first_query_s", "warm_query_s"):
                report["active_stage"] = f"{backend}/{stage}"
                write_json(args.output, report)
                result, record[stage] = timed(
                    lambda index=index, query=query, gpu=gpu: index.nearest(query, return_all=True, return_distance=True) if gpu and args.public_sindex else index.query_nearest(query, all_matches=True, return_distance=True),
                    gpu=gpu,
                )
                if expected is None:
                    expected = canonical(*result)
                    op = args.output.with_suffix(".oracle.npz")
                    np.savez(op, indices=expected[0], distances=expected[1])
                    report.update(oracle_path=str(op), oracle_sha256=sha(op))
                record[stage + "_parity"] = compare(result, expected)
                assert record[stage + "_parity"]["passed"], record[stage + "_parity"]
            if gpu and args.public_sindex:
                record["backends"] = index.backend_info
            if gpu and not args.public_sindex:
                record.update(
                    work=index.last_work,
                    refiner_work=cp.asnumpy(index.refiner.work).tolist(),
                    tree_refiner_bytes=index.refiner.tree.bytes,
                    query_refiner_bytes=index.refiner.query.bytes,
                )
            write_json(args.output, report)
        print(backend, trials[-1], flush=True)
    report["status"] = "passed"
    report.pop("active_stage", None)
    write_json(args.output, report)


if __name__ == "__main__":
    main()
