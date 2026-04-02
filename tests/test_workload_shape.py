from __future__ import annotations

import pytest

from vibespatial.runtime.crossover import WorkloadShape, detect_workload_shape


class TestWorkloadShapeEnum:
    """WorkloadShape StrEnum values and membership."""

    def test_values(self) -> None:
        assert WorkloadShape.PAIRWISE == "pairwise"
        assert WorkloadShape.BROADCAST_RIGHT == "broadcast_right"
        assert WorkloadShape.SCALAR_RIGHT == "scalar_right"

    def test_member_count(self) -> None:
        assert len(WorkloadShape) == 3

    def test_str_identity(self) -> None:
        # StrEnum instances compare equal to their string value.
        assert WorkloadShape.PAIRWISE == "pairwise"
        assert isinstance(WorkloadShape.PAIRWISE, str)


class TestDetectWorkloadShape:
    """detect_workload_shape() classification and edge cases."""

    @pytest.mark.parametrize(
        ("left", "right", "expected"),
        [
            pytest.param(100, 100, WorkloadShape.PAIRWISE, id="equal-lengths"),
            pytest.param(100, 1, WorkloadShape.BROADCAST_RIGHT, id="broadcast-right"),
            pytest.param(100, None, WorkloadShape.SCALAR_RIGHT, id="scalar-right"),
            pytest.param(1, 1, WorkloadShape.PAIRWISE, id="both-one"),
            pytest.param(0, 0, WorkloadShape.PAIRWISE, id="both-zero"),
            # (1, 100) intentionally omitted: BROADCAST_LEFT is not
            # supported, so this raises ValueError (tested below).
        ],
    )
    def test_classification(
        self,
        left: int,
        right: int | None,
        expected: WorkloadShape,
    ) -> None:
        assert detect_workload_shape(left, right) is expected

    @pytest.mark.parametrize(
        ("left", "right"),
        [
            pytest.param(100, 50, id="mismatched-lengths"),
            pytest.param(50, 100, id="mismatched-reversed"),
            pytest.param(2, 3, id="small-mismatch"),
            pytest.param(1, 100, id="broadcast-left-unsupported"),
        ],
    )
    def test_incompatible_raises(self, left: int, right: int) -> None:
        with pytest.raises(ValueError, match="Incompatible lengths"):
            detect_workload_shape(left, right)

    def test_error_message_includes_counts(self) -> None:
        with pytest.raises(ValueError, match=r"left=100.*right=50"):
            detect_workload_shape(100, 50)

    def test_error_message_suggests_sjoin(self) -> None:
        with pytest.raises(ValueError, match="sjoin"):
            detect_workload_shape(100, 50)


class TestImportFromCrossover:
    """WorkloadShape and detect_workload_shape importable from vibespatial.runtime.crossover."""

    def test_import_enum(self) -> None:
        from vibespatial.runtime.crossover import WorkloadShape as WS

        assert WS.PAIRWISE == "pairwise"

    def test_import_function(self) -> None:
        from vibespatial.runtime.crossover import detect_workload_shape as dws

        assert dws(10, 10) is WorkloadShape.PAIRWISE
