"""GPU-accelerated PostGIS read/write via ADBC (Arrow Database Connectivity).

Pipeline:
  read:  ADBC → Arrow table → split WKB column → GPU decode → GeoDataFrame
  write: OwnedGeometryArray → GPU WKB encode → Arrow table → ADBC bulk ingest

ADBC is an OPTIONAL dependency. All public functions return ``None`` when
ADBC is not installed, allowing the caller to fall back to the existing
Shapely-based path transparently.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from vibespatial.runtime import ExecutionMode
from vibespatial.runtime.dispatch import record_dispatch_event
from vibespatial.runtime.fallbacks import record_fallback_event

if TYPE_CHECKING:
    import pyarrow as pa

    from vibespatial.api import GeoDataFrame
    from vibespatial.geometry.owned import OwnedGeometryArray

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SIMPLE_TABLE_RE = re.compile(
    r"^\s*(?:(?P<schema>[a-zA-Z_]\w*)\.)?(?P<table>[a-zA-Z_]\w*)\s*$"
)
"""Matches a bare ``schema.table`` or ``table`` identifier (no SQL keywords)."""


def _is_simple_table_name(sql: str) -> bool:
    """Return True if *sql* is a bare table (or schema.table) identifier."""
    return _SIMPLE_TABLE_RE.match(sql) is not None


def _wrap_sql_for_wkb(sql: str, geom_col: str) -> str:
    """Wrap user SQL so the geometry column is returned as WKB binary.

    Strategy: always wrap as a sub-query.  The geometry column is replaced
    by ``ST_AsBinary("<geom_col>")`` and aliased to ``__vibes_wkb``.  The
    original geometry column is kept in ``*`` but we drop it post-fetch.
    """
    return (
        f'SELECT *, ST_AsBinary("{geom_col}") AS __vibes_wkb '
        f"FROM ({sql}) AS __vibes_sub"
    )


def _get_connection_uri(con: object) -> str | None:
    """Extract a PostgreSQL URI from a SQLAlchemy engine/connection, or passthrough a string.

    Returns ``None`` when the object cannot be converted, signalling the
    caller to fall back to the Shapely path.
    """
    if isinstance(con, str):
        return con
    # SQLAlchemy Engine
    if hasattr(con, "url"):
        url = str(con.url)
        # SQLAlchemy uses ``postgresql://`` which ADBC also accepts.
        return url
    # SQLAlchemy Connection (has .engine.url)
    if hasattr(con, "engine") and hasattr(con.engine, "url"):
        return str(con.engine.url)
    return None


def _detect_crs_from_table(
    con_uri: str,
    sql: str,
    geom_col: str,
) -> str | None:
    """Try to determine the SRID from PostGIS metadata.

    Only works for simple ``table`` or ``schema.table`` queries. Returns
    an ``"EPSG:<srid>"`` string, or ``None`` on failure.
    """
    match = _SIMPLE_TABLE_RE.match(sql)
    if match is None:
        return None

    schema = match.group("schema") or "public"
    table = match.group("table")

    try:
        import adbc_driver_postgresql.dbapi as pg_dbapi

        with pg_dbapi.connect(con_uri) as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT Find_SRID('{schema}', '{table}', '{geom_col}')"
            )
            row = cur.fetchone()
            if row is not None and row[0] not in (None, 0):
                return f"EPSG:{row[0]}"
    except Exception:
        pass

    return None


def _arrow_table_to_geodataframe(
    table: pa.Table,
    geom_col: str,
    crs: str | int | None,
) -> GeoDataFrame:
    """Convert a PyArrow table with a WKB binary geometry column into a GeoDataFrame.

    The WKB column is GPU-decoded via ``decode_wkb_arrow_array_owned``,
    then wrapped into a device-backed GeoSeries. Attribute columns stay
    on host as a pandas DataFrame.
    """
    import pyarrow as pa

    from vibespatial.io.geoarrow import geoseries_from_owned
    from vibespatial.io.wkb import decode_wkb_arrow_array_owned

    # --- extract WKB column -------------------------------------------------
    col_names = table.column_names
    if geom_col not in col_names:
        raise ValueError(
            f"Geometry column {geom_col!r} not found in query result. "
            f"Available columns: {col_names}"
        )

    wkb_column = table.column(geom_col).combine_chunks()

    # If the column is string/utf8 (hex WKB), convert to binary
    if pa.types.is_string(wkb_column.type) or pa.types.is_large_string(
        wkb_column.type
    ):
        wkb_column = pa.array(
            [bytes.fromhex(v.as_py()) if v.is_valid else None for v in wkb_column],
            type=pa.binary(),
        )

    # --- GPU decode ----------------------------------------------------------
    owned: OwnedGeometryArray = decode_wkb_arrow_array_owned(wkb_column)

    # --- build GeoSeries from OGA -------------------------------------------
    geom_series = geoseries_from_owned(owned, name=geom_col, crs=crs)

    # --- build attribute DataFrame (host-side) ------------------------------
    attr_table = table.drop(geom_col)

    import vibespatial.api as geopandas

    attrs_df = attr_table.to_pandas()
    gdf = geopandas.GeoDataFrame(attrs_df, geometry=geom_series)

    return gdf


# ---------------------------------------------------------------------------
# Public read
# ---------------------------------------------------------------------------


def read_postgis_gpu(
    sql: str,
    con: str | object,
    geom_col: str = "geom",
    crs: str | int | None = None,
    chunksize: int | None = None,
) -> GeoDataFrame | None:
    """Read a PostGIS query into a GeoDataFrame using ADBC + GPU WKB decode.

    Parameters
    ----------
    sql : str
        SQL query or table name.
    con : str or SQLAlchemy engine/connection
        PostgreSQL connection URI (``postgresql://...``) or a SQLAlchemy
        connectable whose ``.url`` can be extracted.
    geom_col : str
        Name of the geometry column in the query result.
    crs : str, int, or None
        CRS to assign. If ``None``, attempts auto-detection from PostGIS
        metadata (simple table queries only).
    chunksize : int or None
        Not yet supported for the GPU path. If set, returns ``None`` so
        the caller falls back to the Shapely chunked reader.

    Returns
    -------
    GeoDataFrame or None
        ``None`` when ADBC is unavailable or any error occurs (caller
        should fall through to the Shapely-based path).
    """
    # Chunked reading is not yet supported in the GPU path.
    if chunksize is not None:
        return None

    # --- resolve connection URI ----------------------------------------------
    con_uri = _get_connection_uri(con)
    if con_uri is None:
        return None

    # --- ensure ADBC is installed --------------------------------------------
    try:
        import adbc_driver_postgresql.dbapi as pg_dbapi
    except ImportError:
        return None

    # --- CRS auto-detection --------------------------------------------------
    if crs is None:
        crs = _detect_crs_from_table(con_uri, sql, geom_col)

    # --- SQL rewriting -------------------------------------------------------
    wrapped_sql = _wrap_sql_for_wkb(sql, geom_col)

    # --- execute via ADBC ----------------------------------------------------
    try:
        with pg_dbapi.connect(con_uri) as conn, conn.cursor() as cur:
            cur.execute(wrapped_sql)
            table = cur.fetch_arrow_table()
    except Exception as exc:
        record_fallback_event(
            surface="vibespatial.io.postgis_gpu",
            reason=f"ADBC query execution failed: {exc}",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/postgis_read",
            d2h_transfer=False,
        )
        return None

    # --- use the __vibes_wkb column if present, drop the original geom ------
    col_names = table.column_names
    if "__vibes_wkb" in col_names:
        # Drop the original geometry column (may be PostGIS bytea or text)
        if geom_col in col_names:
            table = table.drop(geom_col)
        # Rename __vibes_wkb → geom_col
        idx = table.column_names.index("__vibes_wkb")
        table = table.rename_columns(
            [
                geom_col if i == idx else c
                for i, c in enumerate(table.column_names)
            ]
        )

    # --- assemble GeoDataFrame -----------------------------------------------
    try:
        gdf = _arrow_table_to_geodataframe(table, geom_col=geom_col, crs=crs)
    except Exception as exc:
        record_fallback_event(
            surface="vibespatial.io.postgis_gpu",
            reason=f"GPU WKB decode or assembly failed: {exc}",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/postgis_read",
            d2h_transfer=False,
        )
        return None

    record_dispatch_event(
        surface="vibespatial.io.postgis_gpu",
        operation="read",
        implementation="adbc_gpu_wkb_decode",
        reason="ADBC Arrow fetch + GPU WKB decode pipeline",
        selected=ExecutionMode.GPU,
    )
    return gdf


# ---------------------------------------------------------------------------
# Public write
# ---------------------------------------------------------------------------


def to_postgis_gpu(
    gdf: GeoDataFrame,
    name: str,
    con: str | object,
    if_exists: str = "fail",
    schema: str | None = None,
    index: bool = False,
) -> bool:
    """Write a GeoDataFrame to PostGIS using ADBC bulk ingest + GPU WKB encode.

    Parameters
    ----------
    gdf : GeoDataFrame
        The data to write.
    name : str
        Target table name.
    con : str or SQLAlchemy engine/connection
        PostgreSQL connection URI or a SQLAlchemy connectable.
    if_exists : str
        ``"fail"``, ``"replace"``, or ``"append"``.
    schema : str or None
        Database schema. Defaults to ``"public"``.
    index : bool
        Whether to write the DataFrame index as a column.

    Returns
    -------
    bool
        ``True`` if the write succeeded via ADBC; ``False`` if the caller
        should fall back to the Shapely/GeoAlchemy2 path.
    """
    con_uri = _get_connection_uri(con)
    if con_uri is None:
        return False

    try:
        import adbc_driver_postgresql.dbapi as pg_dbapi
    except ImportError:
        return False

    import pyarrow as pa

    # --- extract geometry as WKB bytes ---------------------------------------
    geom_name = gdf.geometry.name
    geom_values = gdf.geometry.values

    # Try GPU WKB encode if owned backing exists
    owned: OwnedGeometryArray | None = getattr(geom_values, "_owned", None)
    wkb_list: list[bytes | None]
    if owned is not None:
        from vibespatial.io.wkb import encode_wkb_owned

        wkb_list = encode_wkb_owned(owned, hex=False)
    else:
        # Shapely fallback (no owned backing — data came from pure Shapely)
        import shapely

        wkb_list = [
            shapely.to_wkb(g) if g is not None and not g.is_empty else None
            for g in geom_values
        ]
        record_fallback_event(
            surface="vibespatial.io.postgis_gpu",
            reason="No OwnedGeometryArray backing; fell back to shapely.to_wkb",
            selected=ExecutionMode.CPU,
            pipeline="io/postgis_write",
            d2h_transfer=False,
        )

    # --- determine SRID ------------------------------------------------------
    srid = 0
    if gdf.crs is not None:
        try:
            for confidence in (100, 70, 25):
                epsg = gdf.crs.to_epsg(min_confidence=confidence)
                if epsg is not None:
                    srid = epsg
                    break
        except Exception:
            pass

    # --- build Arrow table with WKB binary column + attributes ---------------
    import pandas as pd

    df = pd.DataFrame(gdf.drop(columns=[geom_name]), copy=False)
    if index:
        df = df.reset_index()

    wkb_array = pa.array(wkb_list, type=pa.binary())
    attr_table = pa.Table.from_pandas(df, preserve_index=False)
    # Append the WKB column
    write_table = attr_table.append_column(geom_name, wkb_array)

    # --- ADBC ingest ---------------------------------------------------------
    schema_name = schema or "public"
    qualified_name = f"{schema_name}.{name}" if schema else name

    try:
        with pg_dbapi.connect(con_uri) as conn, conn.cursor() as cur:
            # Handle if_exists
            if if_exists == "replace":
                cur.execute(f'DROP TABLE IF EXISTS "{schema_name}"."{name}"')
                conn.commit()
            elif if_exists == "fail":
                cur.execute(
                    "SELECT EXISTS ("
                    "  SELECT FROM information_schema.tables"
                    f"  WHERE table_schema = '{schema_name}'"
                    f"  AND table_name = '{name}'"
                    ")"
                )
                row = cur.fetchone()
                if row is not None and row[0]:
                    raise ValueError(
                        f"Table '{qualified_name}' already exists. "
                        f"Use if_exists='replace' or 'append'."
                    )

            # Determine ADBC ingest mode
            mode = "append" if if_exists == "append" else "create"

            cur.adbc_ingest(name, write_table, mode=mode)
            conn.commit()

            # Convert the bytea column to PostGIS geometry type
            alter_sql = (
                f'ALTER TABLE "{schema_name}"."{name}" '
                f'ALTER COLUMN "{geom_name}" '
                f"TYPE geometry USING ST_GeomFromWKB(\"{geom_name}\", {srid})"
            )
            cur.execute(alter_sql)
            conn.commit()

    except Exception as exc:
        record_fallback_event(
            surface="vibespatial.io.postgis_gpu",
            reason=f"ADBC write failed: {exc}",
            detail=str(exc),
            selected=ExecutionMode.CPU,
            pipeline="io/postgis_write",
            d2h_transfer=False,
        )
        return False

    record_dispatch_event(
        surface="vibespatial.io.postgis_gpu",
        operation="write",
        implementation="adbc_gpu_wkb_encode",
        reason="ADBC bulk ingest + GPU WKB encode pipeline",
        selected=ExecutionMode.GPU if owned is not None else ExecutionMode.CPU,
    )
    return True
