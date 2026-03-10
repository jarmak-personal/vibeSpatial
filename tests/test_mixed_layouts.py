from __future__ import annotations

from vibespatial.testing.mixed_layouts import (
    DATASET_MIXES,
    GeometryFamily,
    benchmark_mixed_layout,
    build_payload_model,
    generate_family_codes,
)


def test_generate_family_codes_respects_scale() -> None:
    tags = generate_family_codes(100, DATASET_MIXES["mixed"], seed=4)
    assert tags.size == 100
    assert set(tags.tolist()) == {
        int(GeometryFamily.POINT),
        int(GeometryFamily.LINE),
        int(GeometryFamily.POLYGON),
    }


def test_payload_model_promotes_to_larger_polygon_like_payloads() -> None:
    model = build_payload_model(seed=1, sample_size=64)
    assert model.promoted_point_bytes >= model.point_bytes
    assert model.promoted_polygon_bytes != model.polygon_bytes
    assert model.promoted_line_bytes > 0


def test_benchmark_mixed_layout_returns_recommendation() -> None:
    model = build_payload_model(seed=2, sample_size=64)
    result = benchmark_mixed_layout("mixed", 10_000, model, seed=3)
    assert result.scale == 10_000
    assert result.family_counts["point"] + result.family_counts["line"] + result.family_counts["polygon"] == 10_000
    assert result.tagged_warp_purity < 1.0
    assert result.sort_partition_warp_purity == 1.0
    assert result.recommended_default in {
        "tagged-union storage with direct execution",
        "tagged-union storage with optional late partitioning",
        "dense tagged storage with sort-partition execution",
    }
