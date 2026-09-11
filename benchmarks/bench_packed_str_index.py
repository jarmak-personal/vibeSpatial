"""Public nearest canary; fixtures and ingress are outside reported timings."""
from __future__ import annotations

import time

TIER = 1
REFERENCE_SCALE = "100000"


def run_benchmark():
    import cupy as cp
    import numpy as np
    import shapely

    import vibespatial as vs
    from vibespatial.runtime._runtime import set_requested_mode

    rows, vertices = 100000, 4
    x = np.arange(rows, dtype=float) * 30
    theta = np.linspace(0, 2*np.pi, vertices, endpoint=False)
    ring = np.stack((np.cos(theta), np.sin(theta)), axis=1) * 10
    tree = shapely.polygons(shapely.linearrings(ring[None,:,:]+np.stack((x,np.zeros(rows)),axis=1)[:,None,:]))
    query = shapely.transform(tree, lambda xy: xy + [25.,0.])
    timings = []
    with set_requested_mode("gpu"):
        left, right = vs.GeoSeries.from_wkb(shapely.to_wkb(query)), vs.GeoSeries.from_wkb(shapely.to_wkb(tree))
        index = right.sindex
        for _ in range(2):
            cp.cuda.get_current_stream().synchronize()
            started = time.perf_counter()
            result = index.nearest(left, return_distance=True)
            cp.cuda.get_current_stream().synchronize()
            timings.append(time.perf_counter()-started)
    return {"kernel":"packed_str_index","tier":TIER,"rows":rows,"vertices":vertices,
            "first_query_seconds":timings[0],"reused_query_seconds":timings[1],"output_pairs":len(result[1])}


if __name__ == "__main__":
    print(run_benchmark())
