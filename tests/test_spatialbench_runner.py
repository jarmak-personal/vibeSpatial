from __future__ import annotations

import pandas as pd

from benchmarks.spatialbench.run_benchmark import normalize


def test_normalize_drops_timestamp_timezone_in_utc_for_result_dump() -> None:
    source = pd.DataFrame(
        {
            "when": pd.Series(
                ["2026-01-15 08:30:00-05:00", "2026-07-15 08:30:00-04:00"],
                dtype="datetime64[ns, America/New_York]",
            )
        }
    )

    result = normalize(source)

    assert result["when"].dtype == "datetime64[us]"
    assert result["when"].tolist() == [
        pd.Timestamp("2026-01-15 13:30:00"),
        pd.Timestamp("2026-07-15 12:30:00"),
    ]
