from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point

from vibespatial.api.geometry_array import from_shapely


def _geometry_array():
    return from_shapely([Point(0, 0), None, Point(2, 2)])


def test_geometry_array_readonly_blocks_mutation_and_copy_clears_flag() -> None:
    values = _geometry_array()
    values._readonly = True

    assert values[:]._readonly is True
    assert values.copy()._readonly is False

    with pytest.raises(ValueError, match="Cannot modify read-only array"):
        values[0] = values[2]


def test_geometry_array_readonly_numpy_view_is_not_writeable() -> None:
    values = _geometry_array()
    values._readonly = True

    copied = np.array(values)
    viewed = np.asarray(values)

    assert copied.flags.writeable
    assert not viewed.flags.writeable


def test_geometry_array_fillna_copy_false_respects_readonly() -> None:
    values = _geometry_array()
    values._readonly = True

    with pytest.raises(ValueError, match="Cannot modify read-only array"):
        values.fillna(values[0], copy=False)

    filled = values.fillna(values[0], copy=True)
    assert filled[1].equals(values[0])
    assert values[1] is None
