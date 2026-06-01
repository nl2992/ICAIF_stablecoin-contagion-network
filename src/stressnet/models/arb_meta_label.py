"""CEX arbitrage execution microstructure: kline-proxy labeling and meta-labeling.

Implements a three-tier arbitrage pipeline for the BTC-routed USDC/USDT basis trade:

  Primary filter  : |b_USDC(t)| > threshold_bps
  Oracle label    : y_arb(t) = 1 if net(q,t) > 0 after costs
  Secondary model : LightGBM classifier on price_plus_book features

Since BTC-USDC spot had thin or no liquidity during 2022-2023 stress events, the
BTCUSDC leg is proxied from BTCUSDT × USDCUSDT with a 20% depth haircut.

Cost parameters (from plan):
  taker_fee = 4 bps per leg (8 bps round-trip)
  delta_bps = 5 bps (slippage budget)
  depth_haircut = 0.20 (proxy error bound)
  primary_threshold = 10 bps (minimum gross basis to fire)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TAKER_FEE_BPS: float = 4.0       # per leg
DELTA_BPS: float = 5.0            # timing / slippage budget
DEPTH_HAIRCUT: float = 0.20       # 20% depth haircut for kline proxy
PRIMARY_THRESHOLD_BPS: float = 10.0  # primary signal fires when |b_USDC| > this

KLINE_FULL_COLS = [
    "open_time_ms", "open", "high", "low", "close", "volume",
    "close_time_ms", "quote_volume", "n_trades",
    "taker_buy_base", "taker_buy_quote", "_ignore",
]


# ---------------------------------------------------------------------------
# Kline parsing (extended to include taker columns)
# ---------------------------------------------------------------------------

def parse_klines_full(path) -> pl.DataFrame:
    """Parse 1m kline CSV retaining taker_buy_quote for depth estimation."""
    from pathlib import Path

    df = pl.read_csv(
        Path(path),
        has_header=False,
        new_columns=KLINE_FULL_COLS,
        schema_overrides={
            "open_time_ms": pl.Int64,
            "open": pl.Float64, "high": pl.Float64,
            "low": pl.Float64, "close": pl.Float64,
            "volume": pl.Float64,
            "close_time_ms": pl.Int64,
            "quote_volume": pl.Float64,
            "n_trades": pl.Int64,
            "taker_buy_base": pl.Float64,
            "taker_buy_quote": pl.Float64,
            "_ignore": pl.Utf8,
        },
        ignore_errors=True,
    )
    return df.with_columns(
        (pl.col("open_time_ms") * 1_000).cast(pl.Datetime("us", "UTC")).alias("wall_clock_utc")
    ).select([
        "wall_clock_utc", "open", "high", "low", "close",
        "volume", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote",
    ])


# ---------------------------------------------------------------------------
# Kline ingestion (wraps existing Binance Vision downloader)
# ---------------------------------------------------------------------------

def fetch_klines_api(
    symbol: str,
    start_date,
    end_date,
    cache_dir,
    overwrite: bool = False,
) -> pl.DataFrame | None:
    """Fetch 1m klines via Binance REST API (/api/v3/klines) for date ranges
    where Binance Vision archives are unavailable (post-September 2022 for some pairs).

    Paginates in 1000-candle batches. No API key required.
    """
    from datetime import datetime, timezone, timedelta
    from pathlib import Path

    import requests

    cache_dir = Path(cache_dir)
    parquet_path = cache_dir / f"{symbol}_api_klines.parquet"
    if parquet_path.exists() and not overwrite:
        df = pl.read_parquet(parquet_path)
        return df

    cache_dir.mkdir(parents=True, exist_ok=True)

    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt   = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms   = int(end_dt.timestamp() * 1000)

    all_rows = []
    cursor_ms = start_ms

    while cursor_ms < end_ms:
        url = (
            f"https://api.binance.com/api/v3/klines"
            f"?symbol={symbol}&interval=1m&startTime={cursor_ms}&endTime={end_ms}&limit=1000"
        )
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as exc:
            print(f"    API error ({symbol}): {exc}")
            break

        if not rows:
            break

        all_rows.extend(rows)
        last_open_ms = rows[-1][0]
        cursor_ms = last_open_ms + 60_000  # next minute

        if len(rows) < 1000:
            break

    if not all_rows:
        return None

    # Parse into DataFrame (same schema as Vision klines)
    records = []
    for r in all_rows:
        open_ms = int(r[0])
        records.append({
            "wall_clock_utc": datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc),
            "open":  float(r[1]), "high": float(r[2]),
            "low":   float(r[3]), "close": float(r[4]),
            "volume": float(r[5]),
            "quote_volume": float(r[7]),
            "n_trades": int(r[8]),
            "taker_buy_base":  float(r[9]),
            "taker_buy_quote": float(r[10]),
        })

    df = pl.DataFrame(records).with_columns(
        pl.col("wall_clock_utc").cast(pl.Datetime("us", "UTC"))
    ).unique(subset=["wall_clock_utc"], keep="last").sort("wall_clock_utc")

    df.write_parquet(parquet_path)
    return df


def fetch_klines_range(
    symbol: str,
    start_date,
    end_date,
    cache_dir,
    overwrite: bool = False,
) -> pl.DataFrame | None:
    """Download and concatenate 1m klines for symbol across a date range.

    Returns a polars DataFrame with full kline columns, or None if no data found.
    """
    import io
    import zipfile
    from datetime import timedelta

    import requests

    from stressnet.data.binance import vision_url, download_vision_zip, _VISION_DAILY, _VISION_MONTHLY
    from pathlib import Path

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    from datetime import date as date_type, datetime, timezone

    frames: list[pl.DataFrame] = []
    current = start_date
    while current <= end_date:
        url = vision_url(symbol, "klines/1m", current, monthly=False)
        csv_path = download_vision_zip(url, cache_dir / "daily", overwrite=overwrite)
        if csv_path is not None:
            try:
                df = parse_klines_full(csv_path)
                if df.height > 0:
                    frames.append(df)
            except Exception:
                pass
        current += timedelta(days=1)

    if not frames:
        # Try monthly archives
        months: set[tuple[int, int]] = set()
        current = start_date
        while current <= end_date:
            months.add((current.year, current.month))
            current += timedelta(days=1)
        for year, month in sorted(months):
            from datetime import date as dt_cls
            rep_day = dt_cls(year, month, 1)
            url = vision_url(symbol, "klines/1m", rep_day, monthly=True)
            csv_path = download_vision_zip(url, cache_dir / "monthly", overwrite=overwrite)
            if csv_path is not None:
                try:
                    df = parse_klines_full(csv_path)
                    if df.height > 0:
                        frames.append(df)
                except Exception:
                    pass

    if not frames:
        return None

    from datetime import datetime, timezone

    combined = pl.concat(frames, how="diagonal").sort("wall_clock_utc")
    start_dt = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
    end_dt = datetime(end_date.year, end_date.month, end_date.day, 23, 59, 59, tzinfo=timezone.utc)
    combined = combined.filter(
        (pl.col("wall_clock_utc") >= start_dt) & (pl.col("wall_clock_utc") <= end_dt)
    )
    combined = combined.unique(subset=["wall_clock_utc"], keep="last").sort("wall_clock_utc")
    return combined if combined.height > 0 else None


# ---------------------------------------------------------------------------
# Panel construction
# ---------------------------------------------------------------------------

def build_minute_panel(
    stable_klines: pl.DataFrame,
    btcusdt: pl.DataFrame,
    depth_haircut: float = DEPTH_HAIRCUT,
    basis_mode: str = "stable_direct",
    btcusdc_klines: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Align stablecoin-pair and BTCUSDT klines and compute cost components.

    Args:
        stable_klines: Klines for the stablecoin pair (USDCUSDT, BUSDUSDT, or BTCUSDC).
        btcusdt: BTCUSDT 1m klines.
        depth_haircut: Fraction of BTCUSDC depth unavailable due to proxy error.
        basis_mode: How to compute b_USDC:
            "stable_direct"  - b = stable_close - 1 (USDCUSDT or BUSDUSDT)
            "triangular"     - b = BTCUSDT_close / stable_close - 1 (when stable=BTCUSDC)
        btcusdc_klines: Alias for stable_klines when basis_mode="triangular".

    Returns DataFrame with one row per minute containing basis and cost signals.
    """
    stable_klines = stable_klines.rename(
        {c: f"usdc_{c}" for c in stable_klines.columns if c != "wall_clock_utc"}
    )
    btcusdt = btcusdt.rename(
        {c: f"btc_{c}" for c in btcusdt.columns if c != "wall_clock_utc"}
    )

    df = stable_klines.join(btcusdt, on="wall_clock_utc", how="inner")

    # ---- Basis computation ----
    # "stable_direct": stablecoin pair traded directly vs USDT (USDCUSDT or BUSDUSDT)
    #    b = stable_close - 1   (positive → stablecoin premium; negative → discount)
    # "triangular": stable_klines is BTCUSDC; derive implied USDC/USDT rate from BTC legs
    #    b = BTCUSDT_close / BTCUSDC_close - 1
    #    Positive → USDC is cheap vs USDT (BTCUSDC higher than BTCUSDT implies);
    #    fires when the two BTC price legs diverge > threshold

    MAX_BTC_SPREAD_BPS  = 50.0   # Binance BTCUSDT is very liquid
    MAX_USDC_SPREAD_BPS = 500.0  # stablecoin pair widens during stress

    if basis_mode == "triangular":
        # BTCUSDC close ≈ BTCUSDT / USDCUSDT (rearranging: USDCUSDT = BTCUSDT/BTCUSDC)
        basis_col = (pl.col("btc_close") / (pl.col("usdc_close") + 1e-12) - 1.0) * 10_000.0
    else:
        # Direct stablecoin/USDT price deviation from parity
        basis_col = (pl.col("usdc_close") - 1.0) * 10_000.0

    df = df.with_columns([
        basis_col.alias("basis_signed_bps"),
        basis_col.abs().alias("basis_abs_bps"),
        pl.when(basis_col >= 0.0).then(pl.lit(1.0)).otherwise(pl.lit(-1.0)
                                                               ).alias("basis_direction"),

        # BTCUSDT effective spread: VWAP_buy / VWAP_sell - 1 (capped)
        pl.min_horizontal(
            (
                (pl.col("btc_taker_buy_quote") / (pl.col("btc_taker_buy_base") + 1e-9))
                / ((pl.col("btc_quote_volume") - pl.col("btc_taker_buy_quote"))
                   / (pl.col("btc_volume") - pl.col("btc_taker_buy_base") + 1e-9) + 1e-9)
                - 1.0
            ).abs() * 10_000.0,
            pl.lit(MAX_BTC_SPREAD_BPS),
        ).alias("spread_btcusdt_bps"),

        # Stablecoin-pair effective spread (capped)
        pl.min_horizontal(
            (
                (pl.col("usdc_taker_buy_quote") / (pl.col("usdc_taker_buy_base") + 1e-9))
                / ((pl.col("usdc_quote_volume") - pl.col("usdc_taker_buy_quote"))
                   / (pl.col("usdc_volume") - pl.col("usdc_taker_buy_base") + 1e-9) + 1e-9)
                - 1.0
            ).abs() * 10_000.0,
            pl.lit(MAX_USDC_SPREAD_BPS),
        ).alias("spread_usdcusdt_bps"),

        # Available ask depth in USDT on BTCUSDT leg (proxy for BTCUSDC depth)
        (pl.col("btc_taker_buy_quote") * (1.0 - depth_haircut)).alias("depth_proxy_usdt"),

        # Taker buy pressure ratios
        (pl.col("btc_taker_buy_quote") / (pl.col("btc_quote_volume") + 1e-9)
         ).alias("taker_pressure_btc"),
        (pl.col("usdc_taker_buy_quote") / (pl.col("usdc_quote_volume") + 1e-9)
         ).alias("taker_pressure_usdc"),

        # Activity
        pl.col("btc_n_trades").alias("ntrades_btc"),
        pl.col("usdc_n_trades").alias("ntrades_usdc"),
    ])

    # Total execution cost (bps)
    # Round-trip taker fees: 2 × 4 bps = 8 bps
    # BTC spread (both legs): 2 × spread_btcusdt × (1 + haircut/(1−haircut))
    #   extra term accounts for proxy error in the BTCUSDC leg
    proxy_factor = 1.0 + depth_haircut / (1.0 - depth_haircut)  # = 1.25 for 20% haircut
    df = df.with_columns(
        (
            8.0  # round-trip taker fees
            + 2.0 * pl.col("spread_btcusdt_bps") * proxy_factor
            + pl.col("spread_usdcusdt_bps")
            + DELTA_BPS
        ).alias("total_cost_bps")
    )

    # Net profit if we execute at this minute (oracle: knows direction)
    df = df.with_columns(
        (pl.col("basis_abs_bps") - pl.col("total_cost_bps")).alias("net_bps")
    )

    # Oracle label: profitable trade?
    df = df.with_columns(
        pl.when(pl.col("net_bps") > 0.0).then(1).otherwise(0).cast(pl.Int8).alias("y_arb")
    )

    return df.sort("wall_clock_utc")


# ---------------------------------------------------------------------------
# Primary filter
# ---------------------------------------------------------------------------

def apply_primary_filter(
    panel: pl.DataFrame,
    threshold_bps: float = PRIMARY_THRESHOLD_BPS,
) -> pl.DataFrame:
    """Return only rows where |basis| > threshold (primary signal fires)."""
    return panel.filter(pl.col("basis_abs_bps") > threshold_bps)


# ---------------------------------------------------------------------------
# Feature engineering (price_plus_book feature set)
# ---------------------------------------------------------------------------

_LAG_FEATURES = ["basis_abs_bps", "spread_usdcusdt_bps",
                  "spread_btcusdt_bps", "taker_pressure_btc", "taker_pressure_usdc"]

PRICE_PLUS_BOOK_FEATURES = [
    # Basis signals — use abs so sign-flip between events doesn't break transfer
    "basis_abs_bps",
    "basis_direction",     # +1 if USDC premium, -1 if USDC cheap
    # Book signals
    "spread_usdcusdt_bps", "spread_btcusdt_bps",
    "depth_proxy_usdt", "taker_pressure_btc", "taker_pressure_usdc",
    "ntrades_btc", "ntrades_usdc",
    # Lagged basis (abs — mechanistically symmetric)
    "basis_abs_bps_lag1", "basis_abs_bps_lag2", "basis_abs_bps_lag3",
    "spread_usdcusdt_bps_lag1",
    "taker_pressure_btc_lag1",
    "depth_proxy_usdt_lag1",
    # Derived
    "basis_momentum",      # basis_abs_bps − basis_abs_bps_lag3
    "spread_regime",       # spread_usdcusdt_bps / spread_usdcusdt_bps_lag1
    # Time-of-day
    "hour_sin", "hour_cos",
]


def engineer_features(panel: pl.DataFrame) -> pl.DataFrame:
    """Add lag, momentum, and time features to the full (unfiltered) panel.

    Call BEFORE applying the primary filter so lags are computed across
    the full time series, not just primary-fire minutes.
    """
    # Time features
    panel = panel.with_columns([
        pl.col("wall_clock_utc").dt.hour().alias("_hour"),
    ]).with_columns([
        (pl.col("_hour") * math.pi * 2.0 / 24.0).sin().alias("hour_sin"),
        (pl.col("_hour") * math.pi * 2.0 / 24.0).cos().alias("hour_cos"),
    ]).drop("_hour")

    # Lags (operate on sorted series)
    for col in _LAG_FEATURES:
        if col not in panel.columns:
            continue
        for lag in [1, 2, 3]:
            panel = panel.with_columns(
                pl.col(col).shift(lag).alias(f"{col}_lag{lag}")
            )

    # Momentum (abs basis only — direction-agnostic so it transfers across events)
    if "basis_abs_bps" in panel.columns and "basis_abs_bps_lag3" in panel.columns:
        panel = panel.with_columns(
            (pl.col("basis_abs_bps") - pl.col("basis_abs_bps_lag3")).alias("basis_momentum")
        )
    else:
        panel = panel.with_columns(pl.lit(0.0).alias("basis_momentum"))

    # Spread regime (current vs lagged)
    if "spread_usdcusdt_bps_lag1" in panel.columns:
        panel = panel.with_columns(
            (pl.col("spread_usdcusdt_bps") / (pl.col("spread_usdcusdt_bps_lag1") + 1e-6)
             ).alias("spread_regime")
        )
    else:
        panel = panel.with_columns(pl.lit(1.0).alias("spread_regime"))

    return panel


def get_feature_matrix(
    panel: pl.DataFrame,
    feature_cols: list[str] | None = None,
) -> np.ndarray:
    """Extract numeric feature matrix for a primary-fire DataFrame."""
    if feature_cols is None:
        feature_cols = PRICE_PLUS_BOOK_FEATURES
    available = [c for c in feature_cols if c in panel.columns]
    X = panel.select(available).to_numpy().astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, available


# ---------------------------------------------------------------------------
# Secondary LightGBM classifier
# ---------------------------------------------------------------------------

def build_lgbm(random_state: int = 42) -> Any:
    if not _HAS_LGBM:
        raise ImportError("lightgbm is required for meta-labeling")
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        verbose=-1,
        n_jobs=-1,
    )


def train_secondary(
    train_panel: pl.DataFrame,
    feature_cols: list[str] | None = None,
) -> tuple[Any, list[str]]:
    """Fit secondary LightGBM classifier on primary-fire training data.

    Args:
        train_panel: Primary-fire rows (already filtered) with features + y_arb.
        feature_cols: Feature columns to use.

    Returns:
        (fitted_model, feature_names_used)
    """
    X, used_cols = get_feature_matrix(train_panel, feature_cols)
    y = train_panel["y_arb"].to_numpy()
    model = build_lgbm()
    model.fit(X, y)
    return model, used_cols


def calibrate_threshold(
    model: Any,
    val_panel: pl.DataFrame,
    feature_cols: list[str],
    grid: list[float] | None = None,
) -> float:
    """Find the decision threshold maximising net bps on the validation set.

    Sweeps probability thresholds and picks the one with the highest total
    net_bps on predicted-positive primary-fire minutes.

    Returns:
        Best threshold in [0, 1].
    """
    if grid is None:
        grid = [i / 20.0 for i in range(1, 20)]  # 0.05 to 0.95

    X, _ = get_feature_matrix(val_panel, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    net_bps = val_panel["net_bps"].to_numpy()

    best_thresh = 0.5
    best_net = -np.inf
    for thresh in grid:
        mask = probs >= thresh
        if mask.sum() == 0:
            continue
        net = net_bps[mask].sum()
        if net > best_net:
            best_net = net
            best_thresh = thresh
    return best_thresh


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------

def evaluate_strategy(
    test_panel: pl.DataFrame,
    model: Any,
    feature_cols: list[str],
    threshold: float,
) -> dict:
    """Evaluate meta-labeling strategy on the test primary-fire panel.

    Returns dict with: net_bps, n_trades, oracle_net_bps, oracle_trades,
    oracle_capture, positive_rate_primary.
    """
    X, _ = get_feature_matrix(test_panel, feature_cols)
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    net_bps_arr = test_panel["net_bps"].to_numpy()
    y_arb = test_panel["y_arb"].to_numpy()

    # Strategy: trade when secondary predicts 1
    strategy_mask = preds == 1
    strategy_net = float(net_bps_arr[strategy_mask].sum())
    n_trades = int(strategy_mask.sum())

    # Oracle: trade all primary fires where y_arb=1
    oracle_mask = y_arb == 1
    oracle_net = float(net_bps_arr[oracle_mask].sum())
    oracle_trades = int(oracle_mask.sum())

    # Oracle capture
    if oracle_net > 0:
        oracle_capture = strategy_net / oracle_net
    else:
        oracle_capture = float("nan")

    # Primary positive rate
    positive_rate = float(y_arb.mean()) if len(y_arb) > 0 else 0.0

    return {
        "net_bps": round(strategy_net, 1),
        "n_trades": n_trades,
        "oracle_net_bps": round(oracle_net, 1),
        "oracle_trades": oracle_trades,
        "oracle_capture": round(oracle_capture * 100, 1) if not math.isnan(oracle_capture) else float("nan"),
        "positive_rate_pct": round(positive_rate * 100, 1),
        "n_primary_fires": len(test_panel),
        "threshold_used": round(threshold, 3),
    }


# ---------------------------------------------------------------------------
# Optical summary (cross-event basis table)
# ---------------------------------------------------------------------------

def compute_optical_summary(panel: pl.DataFrame, event_name: str) -> dict:
    """Compute optical positive rate and firing stats for a given event panel.

    'Optical' = percentage of minutes exceeding 10 bps, regardless of cost model.
    """
    total_minutes = panel.height
    fires = panel.filter(pl.col("basis_abs_bps") > PRIMARY_THRESHOLD_BPS)
    n_fires = fires.height
    if n_fires > 0:
        pos_rate = fires["y_arb"].mean()
    else:
        pos_rate = 0.0
    return {
        "event": event_name,
        "total_minutes": total_minutes,
        "n_primary_fires": n_fires,
        "fire_rate_pct": round(n_fires / max(total_minutes, 1) * 100, 2),
        "oracle_positive_rate_pct": round(float(pos_rate) * 100, 1),
        "median_basis_bps_at_fire": round(
            float(fires["basis_abs_bps"].median() or 0.0), 2
        ) if n_fires > 0 else 0.0,
    }
