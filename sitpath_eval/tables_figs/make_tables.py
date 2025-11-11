from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

TABLE_NAMES = [f"table_{idx}" for idx in range(1, 7)]


def _format_entry(row: pd.Series) -> str:
    mean = row["mean"]
    lower = row.get("ci95_low", mean)
    upper = row.get("ci95_high", mean)
    radius = max(mean - lower, upper - mean)
    return f"{mean:.2f} ± {radius:.2f}"


def _load_source(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "model" not in df.columns:
        df["model"] = csv_path.stem
    return df


def compose_table(table_name: str, sources: Iterable[Path], out_dir: Path) -> None:
    frames = []
    for csv_path in sources:
        if not csv_path.exists():
            continue
        frames.append(_load_source(csv_path))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    pivot_rows = []
    for model, group in df.groupby("model"):
        entry = {"model": model}
        for _, row in group.iterrows():
            metric = row["metric"]
            entry[metric] = _format_entry(row)
        pivot_rows.append(entry)
    result_df = pd.DataFrame(pivot_rows).set_index("model")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_out = out_dir / f"{table_name}.csv"
    md_out = out_dir / f"{table_name}.md"
    result_df.to_csv(csv_out)
    result_df.to_markdown(md_out)


def build_all_tables(root: str | Path = "tables") -> None:
    root = Path(root)
    for table_name in TABLE_NAMES:
        table_dir = root / "sources" / table_name
        sources = sorted(table_dir.glob("*.csv"))
        compose_table(table_name, sources, root)


if __name__ == "__main__":
    build_all_tables()
