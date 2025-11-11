from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


def safe_mkdirs(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _prepare_path(path: str | Path) -> Path:
    path = Path(path)
    if path.parent:
        safe_mkdirs(path.parent)
    return path


def save_json(path: str | Path, data: Any, *, indent: int = 2) -> None:
    file_path = _prepare_path(path)
    with file_path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent)


def load_json(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def save_pickle(path: str | Path, obj: Any) -> None:
    file_path = _prepare_path(path)
    with file_path.open("wb") as fh:
        pickle.dump(obj, fh)


def load_pickle(path: str | Path) -> Any:
    file_path = Path(path)
    with file_path.open("rb") as fh:
        return pickle.load(fh)


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str] | None = None,
    *,
    append: bool = False,
) -> None:
    rows = list(rows)
    if not rows:
        return

    if fieldnames is None:
        ordered_keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in ordered_keys:
                    ordered_keys.append(key)
        fieldnames = ordered_keys

    csv_path = _prepare_path(path)
    mode = "a" if append else "w"
    write_header = not append or not csv_path.exists() or csv_path.stat().st_size == 0

    with csv_path.open(mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def append_csv_row(path: str | Path, row: Mapping[str, Any], fieldnames: Sequence[str] | None = None) -> None:
    write_csv(path, [row], fieldnames=fieldnames, append=True)


def read_csv(path: str | Path) -> List[Dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return list(reader)


__all__ = [
    "safe_mkdirs",
    "save_json",
    "load_json",
    "save_pickle",
    "load_pickle",
    "write_csv",
    "append_csv_row",
    "read_csv",
]
