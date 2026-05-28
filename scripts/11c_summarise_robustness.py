"""Aggregate robustness and placebo checks across events."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from stressnet.config import load_events, results_root
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_read(path: Path) -> pl.DataFrame:
    return pl.read_csv(path) if path.exists() else pl.DataFrame()


def main() -> None:
    tables_dir = results_root() / "tables"
    rows = []
    for event_id in load_events().keys():
        path = tables_dir / f"table_robustness_{event_id}.csv"
        df = _safe_read(path)
        if df.is_empty() or "check" not in df.columns or "significant_p01" not in df.columns:
            rows.append(
                {
                    "event_id": event_id,
                    "check": "missing",
                    "n_total": 0,
                    "n_significant": 0,
                    "sig_rate": None,
                }
            )
            continue
        summary = (
            df.group_by("check")
            .agg(
                pl.len().alias("n_total"),
                pl.col("significant_p01").sum().alias("n_significant"),
            )
            .with_columns((pl.col("n_significant") / pl.col("n_total")).alias("sig_rate"))
            .with_columns(pl.lit(event_id).alias("event_id"))
            .select(["event_id", "check", "n_total", "n_significant", "sig_rate"])
        )
        rows.extend(summary.iter_rows(named=True))

    out = pl.DataFrame(rows).sort(["event_id", "check"])
    out_path = tables_dir / "table_robustness_summary.csv"
    out.write_csv(out_path)
    logger.info("Wrote %s", out_path)
    print(out)


if __name__ == "__main__":
    main()
