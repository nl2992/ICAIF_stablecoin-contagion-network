"""L2 order-book reconstruction with sequence tracking and manifest diagnostics.

This module extends ``stressnet.reconstruct.orderbook.OrderBook`` with:

- Per-update sequence-ID gap detection
- Resync (snapshot) counting
- Clock-skew computation (exchange_ts vs local_ts)
- Coverage fraction over an analysis window
- A structured ``BookManifest`` dataclass for tier-downgrade decisions

Typical workflow::

    from stressnet.reconstruct.orderbook_l2 import L2BookReconstructor

    recon = L2BookReconstructor(notional_usd=10_000)
    for row in updates:          # sorted by exchange_ts
        recon.apply(row)
    silver = recon.to_silver_df(start_ts_us=..., end_ts_us=..., grid_seconds=60)
    manifest = recon.manifest()

See ``scripts/02b_reconstruct_l2_books.py`` for the batch pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterator

import numpy as np
import polars as pl

from stressnet.reconstruct.bookwalk import bookwalk_vwap, depth_within_bps
from stressnet.reconstruct.orderbook import OrderBook
from stressnet.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# BookUpdate — minimal typed row
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class BookUpdate:
    """One L2 order-book update message (normalised from bronze schema)."""
    exchange_ts:  float          # microseconds since epoch (exchange clock)
    local_ts:     float          # microseconds since epoch (local ingestion clock)
    sequence_id:  int            # exchange sequence number; -1 = not provided
    side:         str            # "bid" | "ask"
    price:        float
    size:         float          # 0 = remove level
    update_type:  str            # "snapshot" | "delta"
    row_position: int            # position within source file (used for ordering ties)


# ---------------------------------------------------------------------------
# BookManifest — diagnostics for tier-downgrade rules
# ---------------------------------------------------------------------------

@dataclass
class BookManifest:
    """Per-window diagnostics computed after full reconstruction.

    All thresholds come from ``docs/tier_assignment_rules.md``.

    Downgrade rules (any single trigger → tier_actual = B):
      - coverage_pct < 0.50
      - gap_rate > 0.01
      - resync_count > 0  AND cumulative_resync_seconds > 300
      - clock_skew_abs_ms > 5_000
    """
    exchange:               str
    symbol:                 str
    start_ts_us:            float
    end_ts_us:              float
    n_messages:             int      = 0
    n_snapshots:            int      = 0        # resyncs triggered
    sequence_gap_count:     int      = 0
    gap_rate:               float    = 0.0      # gap_count / n_messages
    resync_count:           int      = 0        # = n_snapshots - 1 (first is expected)
    cumulative_resync_seconds: float = 0.0
    coverage_pct:           float    = 0.0
    clock_skew_ms:          float    = 0.0      # median (local_ts - exchange_ts) ms
    clock_skew_abs_ms:      float    = 0.0
    depth_source:           str      = "full_l2_book"
    is_executable_bookwalk: bool     = True

    @property
    def tier_actual(self) -> str:
        if self.coverage_pct < 0.50:
            return "B"
        if self.gap_rate > 0.01:
            return "B"
        if self.resync_count > 0 and self.cumulative_resync_seconds > 300:
            return "B"
        if self.clock_skew_abs_ms > 5_000:
            return "B"
        return "A"

    @property
    def tier_downgrade_reason(self) -> str:
        reasons = []
        if self.coverage_pct < 0.50:
            reasons.append("incomplete_coverage")
        if self.gap_rate > 0.01:
            reasons.append("sequence_gaps")
        if self.resync_count > 0 and self.cumulative_resync_seconds > 300:
            reasons.append("resync")
        if self.clock_skew_abs_ms > 5_000:
            reasons.append("clock_unreliable")
        return ",".join(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exchange":                   self.exchange,
            "symbol":                     self.symbol,
            "n_messages":                 self.n_messages,
            "n_snapshots":                self.n_snapshots,
            "sequence_gap_count":         self.sequence_gap_count,
            "gap_rate":                   self.gap_rate,
            "resync_count":               self.resync_count,
            "cumulative_resync_seconds":  self.cumulative_resync_seconds,
            "coverage_pct":               self.coverage_pct,
            "clock_skew_ms":              self.clock_skew_ms,
            "clock_skew_abs_ms":          self.clock_skew_abs_ms,
            "depth_source":               self.depth_source,
            "is_executable_bookwalk":     self.is_executable_bookwalk,
            "tier_actual":                self.tier_actual,
            "tier_downgrade_reason":      self.tier_downgrade_reason,
        }


# ---------------------------------------------------------------------------
# L2BookReconstructor
# ---------------------------------------------------------------------------

class L2BookReconstructor:
    """Apply incremental L2 updates to an in-memory book; record diagnostics.

    Args:
        notional_usd:   Notional USD order size for bookwalk VWAP (default 10 000).
        depth_bps:      Basis-point range for depth computation (default 10).
    """

    def __init__(
        self,
        notional_usd: float = 10_000.0,
        depth_bps: float = 10.0,
    ) -> None:
        self._book = OrderBook()
        self._notional_usd = notional_usd
        self._depth_bps = depth_bps

        # Diagnostics
        self._n_messages = 0
        self._n_snapshots = 0
        self._seq_gap_count = 0
        self._prev_seq: int | None = None
        self._clock_skews: list[float] = []   # local_ts - exchange_ts, ms
        self._resync_ts: list[float] = []     # exchange_ts of each snapshot

        # Silver output rows
        self._silver_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    def apply(self, upd: BookUpdate) -> None:
        """Apply one update to the book and record a silver snapshot."""
        is_snapshot = upd.update_type == "snapshot"

        # Sequence gap detection
        if upd.sequence_id >= 0 and self._prev_seq is not None and not is_snapshot:
            expected = self._prev_seq + 1
            if upd.sequence_id > expected:
                self._seq_gap_count += upd.sequence_id - expected
        if upd.sequence_id >= 0:
            self._prev_seq = upd.sequence_id

        # Snapshot / resync accounting
        if is_snapshot:
            self._n_snapshots += 1
            if self._n_snapshots > 1:
                self._resync_ts.append(upd.exchange_ts)
            # Snapshots reset the book; apply_snapshot expects list of (price, size) tuples
            # In incremental_book_L2 each snapshot row is a single level —
            # we accumulate them and flush when the first delta arrives.
            # For simplicity we treat every snapshot row as apply_update below.

        # Clock skew
        skew_ms = (upd.local_ts - upd.exchange_ts) / 1_000.0
        self._clock_skews.append(skew_ms)
        self._n_messages += 1

        # Apply to book
        self._book.apply_update(upd.side, upd.price, upd.size)

        # Record silver snapshot
        mid = self._book.mid()
        bb  = self._book.best_bid()
        ba  = self._book.best_ask()
        spread_bps = self._book.spread_bps()
        imbalance  = self._book.imbalance(bps=self._depth_bps)
        bid_depth_usd = self._depth_in_usd("bid", mid)
        ask_depth_usd = self._depth_in_usd("ask", mid)
        buy_vwap  = bookwalk_vwap("buy",  self._book, self._notional_usd)
        sell_vwap = bookwalk_vwap("sell", self._book, self._notional_usd)

        self._silver_rows.append({
            "exchange_ts_us":             upd.exchange_ts,
            "local_ts_us":                upd.local_ts,
            "mid_price":                  mid,
            "best_bid":                   bb,
            "best_ask":                   ba,
            "spread_bps":                 spread_bps,
            "depth_10bps_bid_usd":        bid_depth_usd,
            "depth_10bps_ask_usd":        ask_depth_usd,
            "orderbook_imbalance":        imbalance,
            "executable_price_10k_buy":   buy_vwap,
            "executable_price_10k_sell":  sell_vwap,
            "is_snapshot":                is_snapshot,
            "sequence_id":                upd.sequence_id,
            "depth_source":               "full_l2_book",
            "is_executable_bookwalk":     (buy_vwap is not None and sell_vwap is not None),
        })

    def _depth_in_usd(self, side: str, mid: float | None) -> float | None:
        """Depth within depth_bps of best price, expressed in USD."""
        d_native = self._book.depth_within_bps(side, self._depth_bps)
        if mid is None or mid <= 0:
            return None
        return d_native * mid

    # ------------------------------------------------------------------
    def apply_from_df(self, df: pl.DataFrame) -> None:
        """Apply all updates from a bronze DataFrame (sorted by exchange_ts)."""
        for row in df.sort("exchange_ts").iter_rows(named=True):
            upd = BookUpdate(
                exchange_ts  = float(row.get("exchange_ts", 0.0)),
                local_ts     = float(row.get("local_ts", row.get("exchange_ts", 0.0))),
                sequence_id  = int(row.get("sequence_id", -1) or -1),
                side         = str(row.get("side", "bid")).lower(),
                price        = float(row.get("price", 0.0) or 0.0),
                size         = float(row.get("size", 0.0) or 0.0),
                update_type  = str(row.get("update_type", "delta")),
                row_position = int(row.get("row_position", 0) or 0),
            )
            self.apply(upd)

    # ------------------------------------------------------------------
    def to_silver_df(
        self,
        start_ts_us: float | None = None,
        end_ts_us: float | None = None,
        grid_seconds: int = 60,
    ) -> pl.DataFrame:
        """Return silver-level features resampled to a regular grid.

        Args:
            start_ts_us: Window start in microseconds.
            end_ts_us: Window end in microseconds.
            grid_seconds: Grid spacing for OHLCV resampling.

        Returns:
            DataFrame with one row per grid bucket, indexed by ``exchange_ts_us``.
        """
        if not self._silver_rows:
            return pl.DataFrame()

        df = pl.DataFrame(self._silver_rows)

        # Filter to analysis window
        if start_ts_us is not None:
            df = df.filter(pl.col("exchange_ts_us") >= start_ts_us)
        if end_ts_us is not None:
            df = df.filter(pl.col("exchange_ts_us") <= end_ts_us)

        if df.is_empty():
            return df

        # Resample to grid
        grid_us = grid_seconds * 1_000_000.0
        df = df.with_columns(
            (pl.col("exchange_ts_us") // grid_us * grid_us)
            .cast(pl.Float64)
            .alias("grid_ts_us")
        )

        # Aggregate per bucket: last tick-level values, mean for depths/imbalance
        agg_cols = {
            "mid_price":                  "last",
            "best_bid":                   "last",
            "best_ask":                   "last",
            "spread_bps":                 "mean",
            "depth_10bps_bid_usd":        "mean",
            "depth_10bps_ask_usd":        "mean",
            "orderbook_imbalance":        "mean",
            "executable_price_10k_buy":   "last",
            "executable_price_10k_sell":  "last",
            "is_executable_bookwalk":     "max",
        }
        agg_exprs = []
        for col, agg in agg_cols.items():
            if col not in df.columns:
                continue
            expr = getattr(pl.col(col), agg)()
            agg_exprs.append(expr.alias(col))

        silver = (
            df.group_by("grid_ts_us")
            .agg(
                *agg_exprs,
                pl.col("exchange_ts_us").count().alias("n_updates"),
            )
            .sort("grid_ts_us")
            .with_columns(
                pl.lit("full_l2_book").alias("depth_source"),
            )
        )
        return silver

    # ------------------------------------------------------------------
    def manifest(
        self,
        exchange: str = "",
        symbol: str = "",
        start_ts_us: float = 0.0,
        end_ts_us: float = 0.0,
    ) -> BookManifest:
        """Compute and return the BookManifest for the reconstructed window."""
        # Coverage: fraction of expected grid seconds that have at least one update
        if start_ts_us > 0 and end_ts_us > start_ts_us and self._silver_rows:
            window_seconds = (end_ts_us - start_ts_us) / 1_000_000.0
            ts_arr = np.array([r["exchange_ts_us"] for r in self._silver_rows])
            ts_filtered = ts_arr[
                (ts_arr >= start_ts_us) & (ts_arr <= end_ts_us)
            ]
            if len(ts_filtered) > 0:
                # Bin into 60s buckets
                buckets = np.unique(ts_filtered // 60_000_000)
                expected_buckets = max(1, int(window_seconds / 60))
                coverage = min(1.0, len(buckets) / expected_buckets)
            else:
                coverage = 0.0
        else:
            coverage = 1.0 if self._n_messages > 0 else 0.0

        # Clock skew
        if self._clock_skews:
            skew_arr = np.array(self._clock_skews)
            median_skew = float(np.median(skew_arr))
            abs_skew    = float(np.median(np.abs(skew_arr)))
        else:
            median_skew = 0.0
            abs_skew    = 0.0

        # Gap rate
        gap_rate = (
            self._seq_gap_count / self._n_messages
            if self._n_messages > 0 else 0.0
        )

        # Resync duration (rough: 5 min assumed per resync)
        resync_count = max(0, self._n_snapshots - 1)
        cumulative_resync_s = resync_count * 300.0

        return BookManifest(
            exchange               = exchange,
            symbol                 = symbol,
            start_ts_us            = start_ts_us,
            end_ts_us              = end_ts_us,
            n_messages             = self._n_messages,
            n_snapshots            = self._n_snapshots,
            sequence_gap_count     = self._seq_gap_count,
            gap_rate               = gap_rate,
            resync_count           = resync_count,
            cumulative_resync_seconds = cumulative_resync_s,
            coverage_pct           = coverage,
            clock_skew_ms          = median_skew,
            clock_skew_abs_ms      = abs_skew,
            depth_source           = "full_l2_book",
            is_executable_bookwalk = True,
        )
