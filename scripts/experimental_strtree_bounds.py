"""Shared device bounds packing for isolated STR experiments.

Geometry-agnostic construction; callers own leaf semantics and refinement.
No public dispatch changes. The segment and all-family experiments share this
builder and the same packed bounds kernels.
"""
from __future__ import annotations

import math


class PackedBoundsBuilder:
    def build_recursive(self, bounds_coords):
        cp, fanout = self.cp, self.fanout
        levels = [math.ceil(self.n / self.width)]
        while levels[-1] > 1:
            levels.append(math.ceil(levels[-1] / fanout))
        self.first_leaf, self.total = levels[0], sum(levels)
        self.bounds = cp.empty((4, self.total), dtype=cp.float64)
        self.children = cp.full((self.total - levels[0]) * fanout, -1, dtype=cp.int32)
        self.run("packed_leaves", levels[0], bounds_coords + (self.bounds,),
                 (self.n, 0, levels[0], self.total, self.width))
        start = 0
        for count, parent_count in zip(levels[:-1], levels[1:], strict=True):
            b = self.bounds[:, start:start + count]
            order = self.pack(.5 * (b[0] + b[2]), .5 * (b[1] + b[3]), width=fanout)
            next_start = start + count
            first_child = (next_start - levels[0]) * fanout
            self.children[first_child:first_child + count] = order + start
            self.run("str_parents", parent_count, (self.bounds, self.children),
                     (next_start, parent_count, self.total, levels[0]))
            start = next_start

    def pack(self, x, y, *, width=None):
        from vibespatial.cuda.cccl_primitives import sort_pairs

        cp, n = self.cp, len(x)
        width = self.width if width is None else width
        if self.packing == "morton":
            return sort_pairs(self.morton(x, y), cp.arange(n, dtype=cp.int32)).values
        # Equal-population x slices, then y within each slice. The recursive
        # builder also calls this for parent boxes; leaf-only STR does not.
        order_x = sort_pairs(x, cp.arange(n, dtype=cp.int32)).values
        slices = math.ceil(math.sqrt(math.ceil(n / width)))
        slice_size = math.ceil(n / (slices * width)) * width
        # Sort original float64 y values, then stably group by integer slice.
        # This avoids coordinate quantization and host geometry loops.
        order_y = sort_pairs(y[order_x], cp.arange(n, dtype=cp.int32)).values
        slice_ids = (order_y // slice_size).astype(cp.int32)
        return order_x[order_y[cp.argsort(slice_ids, kind="stable")]]

    def morton(self, x, y):
        cp = self.cp

        def spread(v):
            v = v & cp.uint32(0x0000ffff)
            v = (v | (v << 8)) & cp.uint32(0x00ff00ff)
            v = (v | (v << 4)) & cp.uint32(0x0f0f0f0f)
            v = (v | (v << 2)) & cp.uint32(0x33333333)
            return (v | (v << 1)) & cp.uint32(0x55555555)

        ix = ((x - x.min()) / cp.maximum(x.max() - x.min(), 1.) * 65535.).astype(cp.uint32)
        iy = ((y - y.min()) / cp.maximum(y.max() - y.min(), 1.) * 65535.).astype(cp.uint32)
        return spread(ix) | (spread(iy) << 1)

    def run(self, name, n, arrays, scalars):
        kernel = self.kernels[name]
        grid, block = self.runtime.launch_config(kernel, n)
        self.runtime.launch(kernel, grid=grid, block=block,
                            params=(tuple(a.data.ptr for a in arrays) + scalars,
                                    (self.ptr,) * len(arrays) + (self.i32,) * len(scalars)))

