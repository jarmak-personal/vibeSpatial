from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


EVENT_LOG_ENV_VAR = "VIBESPATIAL_EVENT_LOG"


def get_event_log_path() -> Path | None:
    value = os.environ.get(EVENT_LOG_ENV_VAR, "").strip()
    if not value:
        return None
    return Path(value)


def append_event_record(event_type: str, payload: dict[str, Any]) -> None:
    path = get_event_log_path()
    if path is None:
        return
    record = {"event_type": event_type, **payload}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")
    except OSError:
        return


def read_event_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records
