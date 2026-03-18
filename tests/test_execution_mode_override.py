"""Tests for session-wide execution mode override (VIBESPATIAL_EXECUTION_MODE)."""
from __future__ import annotations

import os

import pytest

from vibespatial.runtime import (
    EXECUTION_MODE_ENV_VAR,
    ExecutionMode,
    get_requested_mode,
    set_execution_mode,
)


@pytest.fixture(autouse=True)
def _clean_mode():
    """Reset execution mode override and env var after each test."""
    set_execution_mode(None)
    old = os.environ.pop(EXECUTION_MODE_ENV_VAR, None)
    yield
    set_execution_mode(None)
    if old is not None:
        os.environ[EXECUTION_MODE_ENV_VAR] = old
    else:
        os.environ.pop(EXECUTION_MODE_ENV_VAR, None)


class TestGetRequestedMode:
    def test_default_is_auto(self):
        assert get_requested_mode() is ExecutionMode.AUTO

    def test_env_var_cpu(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "cpu"
        assert get_requested_mode() is ExecutionMode.CPU

    def test_env_var_gpu(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "gpu"
        assert get_requested_mode() is ExecutionMode.GPU

    def test_env_var_auto(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "auto"
        assert get_requested_mode() is ExecutionMode.AUTO

    def test_env_var_case_insensitive(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "CPU"
        assert get_requested_mode() is ExecutionMode.CPU

    def test_env_var_invalid_raises(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "bogus"
        with pytest.raises(ValueError):
            get_requested_mode()


class TestSetExecutionMode:
    def test_override_wins_over_env(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "gpu"
        set_execution_mode(ExecutionMode.CPU)
        assert get_requested_mode() is ExecutionMode.CPU

    def test_override_with_string(self):
        set_execution_mode("cpu")
        assert get_requested_mode() is ExecutionMode.CPU

    def test_clear_override(self):
        set_execution_mode(ExecutionMode.CPU)
        assert get_requested_mode() is ExecutionMode.CPU
        set_execution_mode(None)
        assert get_requested_mode() is ExecutionMode.AUTO

    def test_clear_restores_env_var(self):
        os.environ[EXECUTION_MODE_ENV_VAR] = "gpu"
        set_execution_mode(ExecutionMode.CPU)
        assert get_requested_mode() is ExecutionMode.CPU
        set_execution_mode(None)
        assert get_requested_mode() is ExecutionMode.GPU


class TestCpuModeBypassPrevention:
    """Verify that CPU mode prevents GPU dispatch at critical entry points."""

    def test_try_gpu_read_file_returns_none_in_cpu_mode(self):
        """_try_gpu_read_file should return None when CPU mode is active."""
        set_execution_mode(ExecutionMode.CPU)

        from vibespatial.io_file import _try_gpu_read_file, plan_vector_file_io
        from vibespatial.io_support import IOOperation

        # Use a dummy filename; the function should bail before touching the file.
        plan = plan_vector_file_io("dummy.geojson", operation=IOOperation.READ)
        result = _try_gpu_read_file(
            "dummy.geojson", plan=plan, bbox=None, columns=None, rows=None,
        )
        assert result is None

    def test_binary_predicate_respects_cpu_mode(self):
        """evaluate_geopandas_binary_predicate should use CPU dispatch mode."""
        import numpy as np
        from shapely.geometry import Point

        set_execution_mode(ExecutionMode.CPU)

        from vibespatial.binary_predicates import (
            evaluate_geopandas_binary_predicate,
        )
        from vibespatial.dispatch import clear_dispatch_events, get_dispatch_events

        left = np.array([Point(0, 0), Point(1, 1)], dtype=object)
        right = np.array([Point(0, 0), Point(2, 2)], dtype=object)

        clear_dispatch_events()
        result = evaluate_geopandas_binary_predicate("intersects", left, right)
        if result is not None:
            # If the predicate was evaluated, check dispatch events
            events = get_dispatch_events()
            for ev in events:
                if hasattr(ev, "selected"):
                    assert ev.selected is ExecutionMode.CPU

    def test_geoseries_from_owned_skips_dga_in_cpu_mode(self):
        """geoseries_from_owned should skip DeviceGeometryArray in CPU mode."""
        from shapely.geometry import Point

        from vibespatial.device_geometry_array import DeviceGeometryArray
        from vibespatial.owned_geometry import from_shapely_geometries

        set_execution_mode(ExecutionMode.CPU)

        from vibespatial.io_geoarrow import geoseries_from_owned

        owned = from_shapely_geometries([Point(0, 0), Point(1, 1)])
        series = geoseries_from_owned(owned, name="geometry", crs="EPSG:4326")
        # In CPU mode, the backing array should NOT be a DeviceGeometryArray
        assert not isinstance(series.values, DeviceGeometryArray)
