from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from vibespatial import has_gpu_runtime

plc = pytest.importorskip("pylibcudf")
rmm = pytest.importorskip("rmm")

if not has_gpu_runtime():
    pytest.skip("GPU runtime unavailable", allow_module_level=True)


def _table_content(table: pa.Table) -> tuple[list, ...]:
    return tuple(table.column(index).to_pylist() for index in range(table.num_columns))


def _read_parquet_table(
    source,
    *,
    columns: list[str] | None = None,
    row_groups: list[list[int]] | None = None,
    filt=None,
    chunked: bool = False,
) -> pa.Table:
    sources = source if isinstance(source, list) else [source]
    options = plc.io.parquet.ParquetReaderOptions.builder(
        plc.io.types.SourceInfo(sources)
    ).build()
    if columns is not None:
        options.set_columns(columns)
    if row_groups is not None:
        options.set_row_groups(row_groups)
    if filt is not None:
        options.set_filter(filt)
    if not chunked:
        return plc.io.parquet.read_parquet(options).tbl.to_arrow()
    reader = plc.io.parquet.ChunkedParquetReader(options)
    chunks: list[pa.Table] = []
    while reader.has_next():
        chunks.append(reader.read_chunk().tbl.to_arrow())
    if not chunks:
        return pa.table({})
    if len(chunks) == 1:
        return chunks[0]
    return pa.concat_tables(chunks)


def test_pylibcudf_parquet_reader_supports_path_bytes_bytesio_and_devicebuffer(
    tmp_path,
) -> None:
    path = tmp_path / "sample.parquet"
    expected = pa.table({"a": [1, 2, 3], "b": [10, 20, 30]})
    pq.write_table(expected, path)
    raw = path.read_bytes()

    by_path = _read_parquet_table(str(path))
    by_bytes = _read_parquet_table(raw)
    by_bytesio = _read_parquet_table(io.BytesIO(raw))
    by_devicebuffer = _read_parquet_table(rmm.DeviceBuffer.to_device(raw))

    expected_content = _table_content(expected)
    assert _table_content(by_path) == expected_content
    assert _table_content(by_bytes) == expected_content
    assert _table_content(by_bytesio) == expected_content
    assert _table_content(by_devicebuffer) == expected_content


def test_pylibcudf_parquet_reader_supports_row_groups_and_filters(
    tmp_path,
) -> None:
    path_a = tmp_path / "part_a.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": [10, 20]}), path_a, row_group_size=1)
    filt = plc.expressions.to_expression("(a > 2)", ("a", "b"))

    selected = _read_parquet_table(
        str(path_a),
        columns=["a"],
        row_groups=[[1]],
    )
    filtered = _read_parquet_table(
        str(path_a),
        row_groups=[[0, 1]],
        filt=filt,
    )

    assert _table_content(selected) == ([2],)
    assert filtered.num_rows == 0
    assert filtered.num_columns == 2


def test_pylibcudf_parquet_reader_supports_multi_source_filters_and_chunked_reader(
    tmp_path,
) -> None:
    path_a = tmp_path / "part_a.parquet"
    path_b = tmp_path / "part_b.parquet"
    pq.write_table(pa.table({"a": [1, 2], "b": [10, 20]}), path_a, row_group_size=1)
    pq.write_table(pa.table({"a": [3, 4], "b": [30, 40]}), path_b, row_group_size=1)
    filt = plc.expressions.to_expression("(a > 2)", ("a", "b"))
    filtered = _read_parquet_table(
        [str(path_a), str(path_b)],
        row_groups=[[0, 1], [0, 1]],
        filt=filt,
    )
    chunked = _read_parquet_table(
        [str(path_a), str(path_b)],
        row_groups=[[0, 1], [0, 1]],
        filt=filt,
        chunked=True,
    )

    assert _table_content(filtered) == ([3, 4], [30, 40])
    assert _table_content(chunked) == ([3, 4], [30, 40])


def test_pylibcudf_file_uri_is_not_supported_remote_endpoint(tmp_path) -> None:
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.table({"a": [1, 2]}), path)

    with pytest.raises(RuntimeError, match="Unsupported endpoint URL"):
        _read_parquet_table(path.as_uri())
