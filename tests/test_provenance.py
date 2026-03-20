"""Tests for the provenance tagging and rewrite system."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from shapely.geometry import Point

from vibespatial.api.geometry_array import GeometryArray
from vibespatial.runtime.provenance import (
    ProvenanceTag,
    RewriteRule,
    _is_point_only,
    _r1_preconditions_met,
    _r6_preconditions_met,
    attempt_provenance_rewrite,
    clear_rewrite_events,
    get_rewrite_candidates,
    get_rewrite_events,
    record_rewrite_event,
    register_rewrite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def point_array():
    """Small array of 10 random points for testing."""
    rng = np.random.default_rng(42)
    points = np.array(
        [Point(x, y) for x, y in zip(rng.uniform(0, 100, 10), rng.uniform(0, 100, 10))],
        dtype=object,
    )
    return GeometryArray(points)


@pytest.fixture()
def other_point_array():
    """Another small array of 10 random points."""
    rng = np.random.default_rng(99)
    points = np.array(
        [Point(x, y) for x, y in zip(rng.uniform(0, 100, 10), rng.uniform(0, 100, 10))],
        dtype=object,
    )
    return GeometryArray(points)


@pytest.fixture()
def polygon_array():
    """Small array of polygons (buffered points)."""
    rng = np.random.default_rng(42)
    points = [Point(x, y) for x, y in zip(rng.uniform(0, 100, 10), rng.uniform(0, 100, 10))]
    polys = np.array([p.buffer(5) for p in points], dtype=object)
    return GeometryArray(polys)


@pytest.fixture(autouse=True)
def _clear_events():
    """Clear rewrite events before and after each test."""
    clear_rewrite_events()
    yield
    clear_rewrite_events()


# ---------------------------------------------------------------------------
# Unit tests: data model
# ---------------------------------------------------------------------------


class TestProvenanceTag:
    def test_frozen(self):
        tag = ProvenanceTag(
            operation="buffer",
            params=(("distance", 10.0),),
            source_geom_types=frozenset({"point"}),
            reason="test",
        )
        with pytest.raises(AttributeError):
            tag.operation = "other"  # type: ignore[misc]

    def test_get_param(self):
        tag = ProvenanceTag(
            operation="buffer",
            params=(("distance", 10.0), ("cap_style", "round")),
            source_geom_types=None,
            reason="test",
        )
        assert tag.get_param("distance") == 10.0
        assert tag.get_param("cap_style") == "round"
        assert tag.get_param("missing") is None
        assert tag.get_param("missing", "default") == "default"

    def test_source_array_excluded_from_repr(self):
        tag = ProvenanceTag(
            operation="buffer",
            params=(("distance", 10.0),),
            source_geom_types=None,
            source_array="fake_ref",
            reason="test",
        )
        r = repr(tag)
        assert "fake_ref" not in r

    def test_source_array_excluded_from_comparison(self):
        tag1 = ProvenanceTag(
            operation="buffer",
            params=(("distance", 10.0),),
            source_geom_types=None,
            source_array="ref_a",
            reason="test",
        )
        tag2 = ProvenanceTag(
            operation="buffer",
            params=(("distance", 10.0),),
            source_geom_types=None,
            source_array="ref_b",
            reason="test",
        )
        assert tag1 == tag2


# ---------------------------------------------------------------------------
# Unit tests: registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_lookup(self):
        rule = RewriteRule(
            name="test_rule",
            input_pattern="test_op",
            consumer_operation="test_consumer",
            preconditions=("p1",),
            reason="test",
        )
        register_rewrite(rule)
        candidates = get_rewrite_candidates("test_op", "test_consumer")
        assert rule in candidates

    def test_no_candidates_for_unknown_pair(self):
        candidates = get_rewrite_candidates("nonexistent_op", "nonexistent_consumer")
        assert candidates == ()

    def test_duplicate_registration_idempotent(self):
        rule = RewriteRule(
            name="idempotent_test",
            input_pattern="idem_op",
            consumer_operation="idem_consumer",
            preconditions=(),
            reason="test",
        )
        register_rewrite(rule)
        register_rewrite(rule)
        candidates = get_rewrite_candidates("idem_op", "idem_consumer")
        assert candidates.count(rule) == 1

    def test_builtin_rules_registered(self):
        """R1, R5, R6 should be registered at import time."""
        r1 = get_rewrite_candidates("buffer", "intersects")
        assert any(r.name == "R1_buffer_intersects_to_dwithin" for r in r1)

        r5 = get_rewrite_candidates("buffer", "__self__")
        assert any(r.name == "R5_buffer_zero_identity" for r in r5)

        r6 = get_rewrite_candidates("buffer", "buffer")
        assert any(r.name == "R6_buffer_chain_merge" for r in r6)


# ---------------------------------------------------------------------------
# Unit tests: event logging
# ---------------------------------------------------------------------------


class TestRewriteEvents:
    def test_record_and_retrieve(self):
        record_rewrite_event(
            rule_name="test_rule",
            surface="test.surface",
            original_operation="op_a",
            rewritten_operation="op_b",
            reason="testing",
        )
        events = get_rewrite_events()
        assert len(events) == 1
        assert events[0].rule_name == "test_rule"
        assert events[0].original_operation == "op_a"
        assert events[0].rewritten_operation == "op_b"

    def test_clear(self):
        record_rewrite_event(
            rule_name="r",
            surface="s",
            original_operation="a",
            rewritten_operation="b",
            reason="t",
        )
        assert len(get_rewrite_events()) == 1
        clear_rewrite_events()
        assert len(get_rewrite_events()) == 0

    def test_get_with_clear(self):
        record_rewrite_event(
            rule_name="r",
            surface="s",
            original_operation="a",
            rewritten_operation="b",
            reason="t",
        )
        events = get_rewrite_events(clear=True)
        assert len(events) == 1
        assert len(get_rewrite_events()) == 0

    def test_to_dict(self):
        event = record_rewrite_event(
            rule_name="r",
            surface="s",
            original_operation="a",
            rewritten_operation="b",
            reason="t",
            detail="d",
            elapsed_seconds=0.042,
        )
        d = event.to_dict()
        assert d["rule_name"] == "r"
        assert d["detail"] == "d"
        assert d["elapsed_seconds"] == 0.042


# ---------------------------------------------------------------------------
# Unit tests: precondition checkers
# ---------------------------------------------------------------------------


class TestPreconditions:
    def _make_tag(self, geom_types, distance=10.0, cap="round", join="round", single=False, source=True):
        return ProvenanceTag(
            operation="buffer",
            params=(
                ("distance", distance),
                ("cap_style", cap),
                ("join_style", join),
                ("single_sided", single),
            ),
            source_geom_types=geom_types,
            source_array="fake" if source else None,
            reason="test",
        )

    def test_r1_accepts_point_round_positive(self):
        tag = self._make_tag(frozenset({"point"}))
        assert _r1_preconditions_met(tag)

    def test_r1_accepts_multipoint(self):
        tag = self._make_tag(frozenset({"multipoint"}))
        assert _r1_preconditions_met(tag)

    def test_r1_rejects_polygon(self):
        tag = self._make_tag(frozenset({"polygon"}))
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_mixed(self):
        tag = self._make_tag(frozenset({"point", "polygon"}))
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_square_cap(self):
        tag = self._make_tag(frozenset({"point"}), cap="square")
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_negative_distance(self):
        tag = self._make_tag(frozenset({"point"}), distance=-5.0)
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_zero_distance(self):
        tag = self._make_tag(frozenset({"point"}), distance=0)
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_single_sided(self):
        tag = self._make_tag(frozenset({"point"}), single=True)
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_no_source(self):
        tag = self._make_tag(frozenset({"point"}), source=False)
        assert not _r1_preconditions_met(tag)

    def test_r1_rejects_none_geom_types(self):
        tag = self._make_tag(None)
        assert not _r1_preconditions_met(tag)

    def test_r6_accepts_matching_style(self):
        tag = self._make_tag(frozenset({"point"}), distance=5.0)
        assert _r6_preconditions_met(tag, 10.0, "round", "round")

    def test_r6_rejects_mismatched_cap(self):
        tag = self._make_tag(frozenset({"point"}), distance=5.0)
        assert not _r6_preconditions_met(tag, 10.0, "square", "round")

    def test_r6_rejects_negative_new_distance(self):
        tag = self._make_tag(frozenset({"point"}), distance=5.0)
        assert not _r6_preconditions_met(tag, -1.0, "round", "round")

    def test_r6_rejects_polygon(self):
        tag = self._make_tag(frozenset({"polygon"}), distance=5.0)
        assert not _r6_preconditions_met(tag, 10.0, "round", "round")

    def test_is_point_only(self):
        assert _is_point_only(frozenset({"point"}))
        assert _is_point_only(frozenset({"multipoint"}))
        assert _is_point_only(frozenset({"point", "multipoint"}))
        assert not _is_point_only(frozenset({"polygon"}))
        assert not _is_point_only(frozenset({"point", "linestring"}))
        assert not _is_point_only(frozenset())
        assert not _is_point_only(None)


# ---------------------------------------------------------------------------
# Integration tests: tag setting
# ---------------------------------------------------------------------------


class TestTagSetting:
    def test_buffer_sets_provenance(self, point_array):
        result = point_array.buffer(10.0)
        assert result._provenance is not None
        assert result._provenance.operation == "buffer"
        assert result._provenance.get_param("distance") == 10.0
        assert result._provenance.source_array is point_array

    def test_buffer_tag_has_geom_types(self, point_array):
        result = point_array.buffer(10.0)
        assert result._provenance is not None
        assert result._provenance.source_geom_types is not None
        assert "point" in result._provenance.source_geom_types

    def test_buffer_tag_preserves_params(self, point_array):
        result = point_array.buffer(10.0, cap_style="square", join_style="mitre")
        tag = result._provenance
        assert tag is not None
        assert tag.get_param("cap_style") == "square"
        assert tag.get_param("join_style") == "mitre"

    def test_no_provenance_on_plain_array(self, point_array):
        assert point_array._provenance is None

    def test_provenance_survives_geoseries_wrapping(self, point_array):
        from vibespatial.api.geoseries import GeoSeries

        buffered = point_array.buffer(10.0)
        gs = GeoSeries(buffered)
        assert gs.values._provenance is not None
        assert gs.values._provenance.operation == "buffer"


# ---------------------------------------------------------------------------
# Integration tests: R5 (buffer(0) identity)
# ---------------------------------------------------------------------------


class TestR5BufferZero:
    def test_buffer_zero_returns_same_geometries(self, point_array):
        result = point_array.buffer(0)
        np.testing.assert_array_equal(
            shapely.get_coordinates(result._data),
            shapely.get_coordinates(point_array._data),
        )

    def test_buffer_zero_logs_rewrite_event(self, point_array):
        point_array.buffer(0)
        events = get_rewrite_events()
        assert len(events) == 1
        assert events[0].rule_name == "R5_buffer_zero_identity"
        assert events[0].rewritten_operation == "identity"

    def test_buffer_zero_no_provenance_tag(self, point_array):
        """buffer(0) result should NOT have a buffer provenance tag."""
        result = point_array.buffer(0)
        assert result._provenance is None


# ---------------------------------------------------------------------------
# Integration tests: R1 (buffer(r).intersects -> dwithin(r))
# ---------------------------------------------------------------------------


class TestR1BufferIntersects:
    def test_equivalence(self, point_array, other_point_array):
        """buffer(r).intersects(Y) should produce the same result as dwithin(Y, r)."""
        distance = 30.0
        buffered = point_array.buffer(distance)

        # With rewrite (provenance-based)
        result_rewrite = buffered.intersects(other_point_array)

        # Without rewrite (direct dwithin)
        result_dwithin = point_array.dwithin(other_point_array, distance)

        np.testing.assert_array_equal(result_rewrite, result_dwithin)

    def test_logs_rewrite_event(self, point_array, other_point_array):
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)

    def test_no_rewrite_for_polygon_input(self, polygon_array, other_point_array):
        """Polygons should NOT trigger R1 -- falls through to normal intersects."""
        buffered = polygon_array.buffer(10.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert not any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)

    def test_multiple_consumers_both_rewrite(self, point_array, other_point_array):
        """Two calls on the same buffered result should both trigger R1."""
        rng = np.random.default_rng(123)
        third = GeometryArray(
            np.array(
                [Point(x, y) for x, y in zip(rng.uniform(0, 100, 10), rng.uniform(0, 100, 10))],
                dtype=object,
            )
        )
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()

        buffered.intersects(other_point_array)
        buffered.intersects(third)

        events = get_rewrite_events()
        r1_events = [e for e in events if e.rule_name == "R1_buffer_intersects_to_dwithin"]
        assert len(r1_events) == 2

    @pytest.mark.parametrize("distance", [0.1, 1.0, 10.0, 50.0, 100.0])
    def test_r1_equivalence_parametrized(self, distance):
        rng = np.random.default_rng(42)
        n = 100
        left = GeometryArray(
            np.array(
                [Point(x, y) for x, y in zip(rng.uniform(0, 100, n), rng.uniform(0, 100, n))],
                dtype=object,
            )
        )
        right = GeometryArray(
            np.array(
                [Point(x, y) for x, y in zip(rng.uniform(0, 100, n), rng.uniform(0, 100, n))],
                dtype=object,
            )
        )
        result_rewrite = left.buffer(distance).intersects(right)
        result_dwithin = left.dwithin(right, distance)
        np.testing.assert_array_equal(result_rewrite, result_dwithin)


# ---------------------------------------------------------------------------
# Integration tests: R6 (buffer chain merge)
# ---------------------------------------------------------------------------


class TestR6BufferChain:
    def test_chain_merge_equivalence(self, point_array):
        """buffer(a).buffer(b) should equal buffer(a+b) for points."""
        a, b = 5.0, 10.0
        chained = point_array.buffer(a).buffer(b)
        direct = point_array.buffer(a + b)

        # Compare areas -- should be very close
        chained_areas = shapely.area(chained._data)
        direct_areas = shapely.area(direct._data)
        np.testing.assert_allclose(chained_areas, direct_areas, rtol=1e-10)

    def test_chain_merge_logs_event(self, point_array):
        point_array.buffer(5.0).buffer(10.0)
        events = get_rewrite_events()
        assert any(e.rule_name == "R6_buffer_chain_merge" for e in events)

    def test_no_chain_merge_for_polygons(self, polygon_array):
        """Polygon input should NOT trigger R6."""
        clear_rewrite_events()
        polygon_array.buffer(5.0).buffer(10.0)
        events = get_rewrite_events()
        assert not any(e.rule_name == "R6_buffer_chain_merge" for e in events)

    def test_no_chain_merge_different_cap(self, point_array):
        """Different cap styles should NOT trigger R6."""
        clear_rewrite_events()
        point_array.buffer(5.0, cap_style="square").buffer(10.0, cap_style="round")
        events = get_rewrite_events()
        assert not any(e.rule_name == "R6_buffer_chain_merge" for e in events)


# ---------------------------------------------------------------------------
# Integration tests: rewrite fallthrough
# ---------------------------------------------------------------------------


class TestRewriteFallthrough:
    def test_non_buffer_op_no_rewrite(self, point_array, other_point_array):
        """Operations without provenance should work normally."""
        clear_rewrite_events()
        result = point_array.intersects(other_point_array)
        assert result is not None
        events = get_rewrite_events()
        assert len(events) == 0

    def test_attempt_provenance_rewrite_returns_none_for_no_tag(self, point_array, other_point_array):
        result = attempt_provenance_rewrite("intersects", point_array, other_point_array)
        assert result is None


# ---------------------------------------------------------------------------
# Unit tests: elapsed_seconds on RewriteEvent
# ---------------------------------------------------------------------------


class TestRewriteTiming:
    def test_rewrite_event_has_elapsed_seconds(self, point_array, other_point_array):
        """R1 rewrite event should carry positive timing."""
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        r1 = [e for e in events if e.rule_name == "R1_buffer_intersects_to_dwithin"]
        assert len(r1) == 1
        assert r1[0].elapsed_seconds > 0

    def test_r5_event_has_elapsed_seconds(self, point_array):
        """R5 rewrite event should carry non-negative timing."""
        clear_rewrite_events()
        point_array.buffer(0)

        events = get_rewrite_events()
        r5 = [e for e in events if e.rule_name == "R5_buffer_zero_identity"]
        assert len(r5) == 1
        assert r5[0].elapsed_seconds >= 0

    def test_r6_event_has_elapsed_seconds(self, point_array):
        """R6 rewrite event should carry positive timing."""
        clear_rewrite_events()
        point_array.buffer(5.0).buffer(10.0)

        events = get_rewrite_events()
        r6 = [e for e in events if e.rule_name == "R6_buffer_chain_merge"]
        assert len(r6) == 1
        assert r6[0].elapsed_seconds > 0

    def test_rewrite_event_to_dict_includes_elapsed(self, point_array, other_point_array):
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        d = events[0].to_dict()
        assert "elapsed_seconds" in d
        assert d["elapsed_seconds"] > 0


# ---------------------------------------------------------------------------
# Integration tests: A/B disable switch
# ---------------------------------------------------------------------------


class TestDisableSwitch:
    @pytest.fixture(autouse=True)
    def _restore_switch(self):
        """Ensure the switch is restored after each test."""
        from vibespatial.runtime.provenance import set_provenance_rewrites

        yield
        set_provenance_rewrites(None)

    def test_default_is_enabled(self):
        from vibespatial.runtime.provenance import provenance_rewrites_enabled

        assert provenance_rewrites_enabled()

    def test_disable_via_python_api(self, point_array, other_point_array):
        from vibespatial.runtime.provenance import set_provenance_rewrites

        set_provenance_rewrites(False)
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert not any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)

    def test_reenable_after_disable(self, point_array, other_point_array):
        from vibespatial.runtime.provenance import set_provenance_rewrites

        set_provenance_rewrites(False)
        set_provenance_rewrites(None)  # clear override -> back to default (enabled)

        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)

    def test_disable_via_env_var(self, monkeypatch, point_array, other_point_array):
        monkeypatch.setenv("VIBESPATIAL_PROVENANCE_REWRITES", "off")

        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert not any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)

    def test_r5_respects_disable(self, point_array):
        """When disabled, buffer(0) should run the full buffer (producing valid polygons, not identity)."""
        from vibespatial.runtime.provenance import set_provenance_rewrites

        set_provenance_rewrites(False)
        clear_rewrite_events()
        result = point_array.buffer(0)

        events = get_rewrite_events()
        assert not any(e.rule_name == "R5_buffer_zero_identity" for e in events)
        # Result should still be valid geometry (shapely buffer(0) returns empty geoms or points)
        assert len(result) == len(point_array)

    def test_r1_respects_disable(self, point_array, other_point_array):
        """When disabled, buffer().intersects() runs the normal predicate path."""
        from vibespatial.runtime.provenance import set_provenance_rewrites

        set_provenance_rewrites(False)
        buffered = point_array.buffer(30.0)
        clear_rewrite_events()
        result = buffered.intersects(other_point_array)

        events = get_rewrite_events()
        assert not any(e.rule_name == "R1_buffer_intersects_to_dwithin" for e in events)
        # Result should still be a valid boolean array
        assert result.dtype == bool
        assert len(result) == len(point_array)


# ---------------------------------------------------------------------------
# Integration tests: R7 (simplify(0) identity)
# ---------------------------------------------------------------------------


class TestR7SimplifyZero:
    def test_simplify_zero_returns_same_geometries(self, polygon_array):
        result = polygon_array.simplify(0)
        np.testing.assert_array_equal(
            shapely.get_coordinates(result._data),
            shapely.get_coordinates(polygon_array._data),
        )

    def test_simplify_zero_logs_rewrite_event(self, polygon_array):
        clear_rewrite_events()
        polygon_array.simplify(0)
        events = get_rewrite_events()
        assert len(events) == 1
        assert events[0].rule_name == "R7_simplify_zero_identity"
        assert events[0].rewritten_operation == "identity"

    def test_simplify_zero_has_elapsed_seconds(self, polygon_array):
        clear_rewrite_events()
        polygon_array.simplify(0)
        events = get_rewrite_events()
        assert events[0].elapsed_seconds >= 0

    def test_simplify_nonzero_no_rewrite(self, polygon_array):
        clear_rewrite_events()
        polygon_array.simplify(1.0)
        events = get_rewrite_events()
        assert not any(e.rule_name == "R7_simplify_zero_identity" for e in events)

    def test_simplify_zero_respects_disable(self, polygon_array):
        from vibespatial.runtime.provenance import set_provenance_rewrites

        set_provenance_rewrites(False)
        try:
            clear_rewrite_events()
            result = polygon_array.simplify(0)
            events = get_rewrite_events()
            assert not any(e.rule_name == "R7_simplify_zero_identity" for e in events)
            assert len(result) == len(polygon_array)
        finally:
            set_provenance_rewrites(None)

    def test_r7_rule_registered(self):
        candidates = get_rewrite_candidates("simplify", "__self__")
        assert any(r.name == "R7_simplify_zero_identity" for r in candidates)


# ---------------------------------------------------------------------------
# Integration tests: R2 (sjoin buffer+intersects -> sjoin dwithin)
# ---------------------------------------------------------------------------


class TestR2SjoinBufferIntersects:
    def test_equivalence(self):
        """sjoin(buffer(r, X), Y, 'intersects') should match sjoin(X, Y, 'dwithin', r)."""
        import vibespatial.api as gpd

        rng = np.random.default_rng(42)
        n = 200
        left = gpd.GeoDataFrame(
            {"val": rng.integers(0, 100, n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        right = gpd.GeoDataFrame(
            {"zone": rng.integers(0, 10, n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )

        # With rewrite (provenance-based)
        buffered_left = left.copy()
        buffered_left["geometry"] = left.geometry.buffer(5.0)
        clear_rewrite_events()
        result_rewrite = gpd.sjoin(buffered_left, right, predicate="intersects")

        events = get_rewrite_events()
        assert any(e.rule_name == "R2_sjoin_buffer_intersects_to_dwithin" for e in events)

        # Direct dwithin (ground truth)
        result_dwithin = gpd.sjoin(left, right, predicate="dwithin", distance=5.0)

        # Both should find the same pairs (index sets should match)
        rewrite_pairs = set(zip(result_rewrite.index, result_rewrite["index_right"]))
        dwithin_pairs = set(zip(result_dwithin.index, result_dwithin["index_right"]))
        assert rewrite_pairs == dwithin_pairs

    def test_logs_rewrite_event(self):
        import vibespatial.api as gpd

        rng = np.random.default_rng(42)
        n = 50
        left = gpd.GeoDataFrame(
            {"val": np.arange(n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        right = gpd.GeoDataFrame(
            {"zone": np.arange(n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        buffered_left = left.copy()
        buffered_left["geometry"] = left.geometry.buffer(5.0)
        clear_rewrite_events()
        gpd.sjoin(buffered_left, right, predicate="intersects")

        events = get_rewrite_events()
        r2 = [e for e in events if e.rule_name == "R2_sjoin_buffer_intersects_to_dwithin"]
        assert len(r2) == 1
        assert "buffer_distance=5.0" in r2[0].detail

    def test_no_rewrite_for_polygon_input(self):
        """Polygon geometries should NOT trigger R2."""
        import vibespatial.api as gpd

        rng = np.random.default_rng(42)
        n = 50
        polys = shapely.buffer(shapely.points(rng.uniform(0, 100, (n, 2))), 3.0)
        left = gpd.GeoDataFrame(
            {"val": np.arange(n)},
            geometry=gpd.GeoSeries(polys),
        )
        right = gpd.GeoDataFrame(
            {"zone": np.arange(n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        buffered_left = left.copy()
        buffered_left["geometry"] = left.geometry.buffer(5.0)
        clear_rewrite_events()
        gpd.sjoin(buffered_left, right, predicate="intersects")

        events = get_rewrite_events()
        assert not any(e.rule_name == "R2_sjoin_buffer_intersects_to_dwithin" for e in events)

    def test_no_rewrite_when_disabled(self):
        import vibespatial.api as gpd
        from vibespatial.runtime.provenance import set_provenance_rewrites

        rng = np.random.default_rng(42)
        n = 50
        left = gpd.GeoDataFrame(
            {"val": np.arange(n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        right = gpd.GeoDataFrame(
            {"zone": np.arange(n)},
            geometry=gpd.GeoSeries(shapely.points(rng.uniform(0, 100, (n, 2)))),
        )
        buffered_left = left.copy()
        buffered_left["geometry"] = left.geometry.buffer(5.0)

        set_provenance_rewrites(False)
        try:
            clear_rewrite_events()
            gpd.sjoin(buffered_left, right, predicate="intersects")
            events = get_rewrite_events()
            assert not any(e.rule_name == "R2_sjoin_buffer_intersects_to_dwithin" for e in events)
        finally:
            set_provenance_rewrites(None)

    def test_r2_rule_registered(self):
        candidates = get_rewrite_candidates("buffer", "sjoin")
        assert any(r.name == "R2_sjoin_buffer_intersects_to_dwithin" for r in candidates)
