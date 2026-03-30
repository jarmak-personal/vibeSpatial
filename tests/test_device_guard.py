"""Tests for the runtime device-residency guard, including scalar exemption."""
from __future__ import annotations

import pytest

cp = pytest.importorskip("cupy")

from vibespatial.testing import DeviceResidencyViolation, device_residency_guard


class TestScalarExemption:
    """Scalar (size <= 1) transfers are exempt; bulk transfers still raise."""

    def test_bulk_get_raises(self):
        arr = cp.arange(10)
        with device_residency_guard("test"):
            with pytest.raises(DeviceResidencyViolation):
                arr.get()

    def test_scalar_get_allowed(self):
        scalar = cp.array(42)
        with device_residency_guard("test"):
            result = scalar.get()
        assert int(result) == 42

    def test_single_element_get_allowed(self):
        arr = cp.arange(10)
        with device_residency_guard("test"):
            result = int(arr[-1].get())
        assert result == 9

    def test_bulk_asnumpy_raises(self):
        arr = cp.arange(10)
        with device_residency_guard("test"):
            with pytest.raises(DeviceResidencyViolation):
                cp.asnumpy(arr)

    def test_scalar_asnumpy_allowed(self):
        scalar = cp.array(7.0)
        with device_residency_guard("test"):
            result = cp.asnumpy(scalar)
        assert float(result) == 7.0

    def test_bulk_np_asarray_raises(self):
        import numpy as np

        arr = cp.arange(10)
        with device_residency_guard("test"):
            with pytest.raises(DeviceResidencyViolation):
                np.asarray(arr)

    def test_scalar_np_asarray_allowed(self):
        """np.asarray on a scalar cupy array passes the guard.

        CuPy itself may reject the implicit conversion (TypeError), so
        we only verify the guard does not raise DeviceResidencyViolation.
        """
        import numpy as np

        scalar = cp.array(3)
        with device_residency_guard("test"):
            try:
                np.asarray(scalar)
            except TypeError:
                pass  # CuPy rejects implicit conversion — guard still passed

    def test_indexing_last_element_allowed(self):
        """The motivating use-case: int(d_offsets[-1]) for allocation sizing."""
        offsets = cp.array([0, 5, 12, 20], dtype=cp.int32)
        with device_residency_guard("test"):
            total = int(offsets[-1])
        assert total == 20
