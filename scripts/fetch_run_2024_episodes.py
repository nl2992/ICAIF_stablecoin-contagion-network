"""Item 9 — 2024-2025 episode extension.

Fetches on-chain Curve 3pool data and Binance USDT/USD market data for two
new episodes, builds gold-schema features, and runs Forbes-Rigobon +
causal HMM analysis.  Results are written to:

    results/tables/table_2024_episodes_detection.json
    results/tables/table_2024_episodes_fr.json
    data/gold/dataset_contagion_features_{event_id}.parquet   (per episode)

Episodes
--------
usdt_curve_2024_aug
    August 4-8 2024, BTC/ETH carry-trade unwind.  ~25 % crypto crash from
    Japan carry-trade reversal + US jobs report miss.  DeFi liquidation
    cascade hit Curve lending markets; 3pool saw elevated USDT flow as
    investors rebalanced.  Mechanism: market_deleveraging_defi_native.
    Hypothesis: on-chain HMM fires (liquidation cascade ≈ endogenous DeFi
    stress), market HMM also fires (macro shock visible in CEX basis).

bybit_hack_2025
    February 21 2025, ByBit exchange hack ($1.5 B ETH stolen — largest
    crypto hack ever).  Exchange-credit mechanism: analogous to FTX 2022.
    Massive CEX withdrawal spike; stablecoin flight on CEX.  Hypothesis:
    on-chain 3pool HMM does NOT fire (3pool uninvolved), market HMM fires
    (CEX USDT basis widens as users rush to withdraw).

Usage
-----
    cd stablecoin-contagion-network
    python scripts/fetch_run_2024_episodes.py
    python scripts/fetch_run_2024_episodes.py --episodes usdt_curve_2024_aug
    python scripts/fetch_run_2024_episodes.py --dry-run   # skip data fetch
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import requests
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Episode definitions
# ---------------------------------------------------------------------------

_EPISODES: dict[str, dict[str, Any]] = {
    "usdt_curve_2024_aug": {
        "name": "USDT/Curve 2024 Aug (BTC carry-trade crash)",
        "mechanism": "market_deleveraging_defi_native",
        "analysis_window": ("2024-07-28", "2024-08-12"),
        "shock_onset_utc": "2024-08-05T00:00:00Z",
        "calm_end_utc": "2024-08-04T23:59:59Z",
        "panic_end_utc": "2024-08-08T23:59:59Z",
        "curve_pool": "curve_3pool",
        "cex_symbol": "USDCUSDT",   # Binance USDC/USDT: close>1 = USDT discount
        "expected_onchain_fires": True,
    },
    "bybit_hack_2025": {
        "name": "ByBit hack 2025 (exchange-credit)",
        "mechanism": "exchange_credit_hack",
        "analysis_window": ("2025-02-16", "2025-03-01"),
        "shock_onset_utc": "2025-02-21T06:00:00Z",
        "calm_end_utc": "2025-02-21T05:59:59Z",
        "panic_end_utc": "2025-02-25T23:59:59Z",
        "curve_pool": "curve_3pool",
        "cex_symbol": "USDTUSDC",
        "expected_onchain_fires": False,
    },
}

# Curve 3pool contract and TokenExchange event topic
_3POOL_ADDRESS = "0xbEbc44782C7dB0a1A60Cb6fe97d0b483032FF1C7"
_TOPIC_TOKEN_EXCHANGE = "0x8b3e96f2b889fa771c53c981b40daf005f63f637f1869f707052d15a3dd97140"
# Token indices in 3pool: 0=DAI(18), 1=USDC(6), 2=USDT(6)
_3POOL_TOKENS = {0: ("DAI", 18), 1: ("USDC", 6), 2: ("USDT", 6)}
_3POOL_SIZE_USD = 300_000_000   # conservative pool size normaliser for 2024

# ---------------------------------------------------------------------------
# Etherscan helpers
# ---------------------------------------------------------------------------

_ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
_RATE_SLEEP = 0.3


def _etherscan_get(params: dict) -> dict:
    params.setdefault("chainid", 1)
    params.setdefault("apikey", os.environ.get("ETHERSCAN_API_KEY", ""))
    try:
        r = requests.get(_ETHERSCAN_BASE, params=params, timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("Etherscan request failed: %s", e)
        return {"status": "0", "result": []}


def _block_by_ts(ts_unix: int, closest: str = "before") -> int:
    data = _etherscan_get({
        "module": "block", "action": "getblocknobytime",
        "timestamp": ts_unix, "closest": closest,
    })
    try:
        return int(data.get("result", "0"))
    except (ValueError, TypeError):
        return 0


def _fetch_curve_logs(from_block: int, to_block: int,
                      page: int = 1, offset: int = 1000) -> list[dict]:
    data = _etherscan_get({
        "module": "logs", "action": "getLogs",
        "address": _3POOL_ADDRESS,
        "topic0": _TOPIC_TOKEN_EXCHANGE,
        "fromBlock": from_block, "toBlock": to_block,
        "offset": offset, "page": page,
    })
    result = data.get("result", [])
    return result if isinstance(result, list) else []


def _fetch_all_curve_logs(from_block: int, to_block: int,
                          max_results: int = 50_000) -> list[dict]:
    """Paginated fetch of 3pool TokenExchange events."""
    all_logs: list[dict] = []
    for page in range(1, 51):
        logs = _fetch_curve_logs(from_block, to_block, page=page)
        all_logs.extend(logs)
        if len(all_logs) >= max_results:
            logger.debug("Capped at %d logs", max_results)
            return all_logs[:max_results]
        if len(logs) < 1000:
            break
        time.sleep(_RATE_SLEEP)
    return all_logs


# ---------------------------------------------------------------------------
# TokenExchange decode (no web3 dependency)
# ---------------------------------------------------------------------------

def _decode_int128(hex_str: str, slot: int) -> int | None:
    try:
        raw = bytes.fromhex(hex_str[2:] if hex_str.startswith("0x") else hex_str)
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start:start + 32], "big", signed=True)
    except Exception:
        return None


def _decode_uint256(hex_str: str, slot: int) -> int | None:
    try:
        raw = bytes.fromhex(hex_str[2:] if hex_str.startswith("0x") else hex_str)
        start = slot * 32
        if len(raw) < start + 32:
            return None
        return int.from_bytes(raw[start:start + 32], "big")
    except Exception:
        return None


def _decode_token_exchange(log: dict) -> dict | None:
    """Return decoded swap amounts or None on failure."""
    data = log.get("data", "0x")
    ts_hex = log.get("timeStamp", "0x0")
    try:
        ts = int(ts_hex, 16)
    except ValueError:
        return None

    sold_id = _decode_int128(data, 0)
    tokens_sold = _decode_uint256(data, 1)
    bought_id = _decode_int128(data, 2)
    tokens_bought = _decode_uint256(data, 3)

    if sold_id is None or tokens_sold is None or bought_id is None:
        return None

    def norm(idx: int, raw: int) -> float:
        _, dec = _3POOL_TOKENS.get(idx, ("UNK", 18))
        return raw / (10 ** dec)

    sold_sym = _3POOL_TOKENS.get(sold_id, ("UNK", 18))[0]
    bought_sym = _3POOL_TOKENS.get(bought_id, ("UNK", 18))[0]

    return {
        "ts": ts,
        "sold_id": sold_id,
        "sold_sym": sold_sym,
        "bought_id": bought_id,
        "bought_sym": bought_sym,
        "sold_amt": norm(sold_id, tokens_sold),
        "bought_amt": norm(bought_id, tokens_bought or 0),
    }


# ---------------------------------------------------------------------------
# Binance public kline fetch (1m USDT/USDC or USDT/USD proxy)
# ---------------------------------------------------------------------------

def _fetch_binance_klines(symbol: str, start_ms: int, end_ms: int,
                           interval: str = "1m") -> list[dict]:
    """Fetch 1-minute Binance kline data (public endpoint, no API key)."""
    url = "https://api.binance.com/api/v3/klines"
    all_klines = []
    t = start_ms
    while t < end_ms:
        params = {
            "symbol": symbol, "interval": interval,
            "startTime": t, "endTime": min(t + 999 * 60_000, end_ms),
            "limit": 1000,
        }
        try:
            r = requests.get(url, params=params, timeout=30)
            r.raise_for_status()
            klines = r.json()
            if not klines:
                break
            for k in klines:
                all_klines.append({
                    "ts_ms": int(k[0]),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]), "n_trades": int(k[8]),
                })
            t = int(klines[-1][0]) + 60_000
            time.sleep(0.1)
        except Exception as e:
            logger.warning("Binance kline fetch failed for %s: %s", symbol, e)
            break
    return all_klines


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _build_pool_features(logs: list[dict],
                          start_utc: datetime,
                          end_utc: datetime,
                          shock_onset: datetime,
                          calm_end: datetime,
                          panic_end: datetime) -> pl.DataFrame:
    """Aggregate decoded 3pool logs to hourly features + event_phase labels."""
    rows = []
    for log in logs:
        dec = _decode_token_exchange(log)
        if dec is None:
            continue
        # Net USDC sold = USDC out of pool → positive means USDC sold (pool gets USDC)
        # From trader POV: sold_sym=USDT → bought_sym=USDC means USDT→USDC swap
        # usdc_net_sold_1h: positive = USDC flowing INTO pool (stress signal for 3pool)
        usdc_delta = 0.0
        if dec["bought_sym"] == "USDC":
            usdc_delta = dec["bought_amt"]    # USDC leaving pool (sold by pool)
        elif dec["sold_sym"] == "USDC":
            usdc_delta = -dec["sold_amt"]     # USDC entering pool

        # Also track USDT → USDC direction for reserve_imbalance proxy
        usdt_delta = 0.0
        if dec["bought_sym"] == "USDT":
            usdt_delta = dec["bought_amt"]
        elif dec["sold_sym"] == "USDT":
            usdt_delta = -dec["sold_amt"]

        rows.append({"ts": dec["ts"], "usdc_delta": usdc_delta, "usdt_delta": usdt_delta,
                     "sold_amt": dec["sold_amt"]})

    if not rows:
        logger.warning("No decoded logs — returning empty features")
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    df = df.with_columns(
        (pl.col("ts") * 1_000_000).cast(pl.Datetime("us", "UTC")).alias("dt")
    )

    # Hourly buckets
    df = df.with_columns(
        (pl.col("ts") // 3600).alias("h")
    )
    agg = df.group_by("h").agg([
        pl.col("usdc_delta").sum().alias("usdc_net_sold_1h"),
        pl.col("usdt_delta").sum().alias("usdt_net_sold_1h"),
        pl.col("sold_amt").sum().alias("total_volume_1h"),
        pl.col("ts").len().cast(pl.UInt32).alias("n_events"),
    ]).sort("h")

    # Fill missing hours with zeros
    h_start = int(start_utc.timestamp()) // 3600
    h_end = int(end_utc.timestamp()) // 3600
    all_hours = pl.DataFrame({"h": list(range(h_start, h_end + 1))})
    agg = all_hours.join(agg, on="h", how="left").with_columns([
        pl.col("usdc_net_sold_1h").fill_null(0.0),
        pl.col("usdt_net_sold_1h").fill_null(0.0),
        pl.col("total_volume_1h").fill_null(0.0),
        pl.col("n_events").fill_null(pl.lit(0, dtype=pl.UInt32)),
    ])

    # Cumulative and derived features
    agg = agg.with_columns([
        pl.col("usdc_net_sold_1h").cum_sum().alias("usdc_net_sold_cum"),
    ])

    # Reserve imbalance proxy: cumulative USDC net sold / pool size
    agg = agg.with_columns([
        (pl.col("usdc_net_sold_cum") / _3POOL_SIZE_USD).alias("reserve_imbalance"),
    ])

    # Implied pool price: approximation from cumulative imbalance
    # (USDT implied price relative to USDC peg)
    agg = agg.with_columns([
        (1.0 - pl.col("reserve_imbalance").clip(-0.3, 0.3)).alias("implied_pool_price"),
    ])

    # Wall clock UTC
    agg = agg.with_columns([
        ((pl.col("h") * 3_600 * 1_000_000).cast(pl.Datetime("us", "UTC"))).alias("wall_clock_utc"),
    ])

    # Event phase labels
    shock_ts = int(shock_onset.timestamp())
    calm_ts = int(calm_end.timestamp())
    panic_ts = int(panic_end.timestamp())

    def phase(h_val: int) -> str:
        ts = h_val * 3600
        if ts < shock_ts:
            return "calm"
        elif ts <= panic_ts:
            return "panic"
        else:
            return "recovery"

    agg = agg.with_columns([
        pl.col("h").map_elements(phase, return_dtype=pl.String).alias("event_phase"),
    ])

    return agg.select([
        "wall_clock_utc", "usdc_net_sold_1h", "usdc_net_sold_cum",
        "n_events", "reserve_imbalance", "implied_pool_price", "event_phase",
        "total_volume_1h",
    ])


def _build_market_features(klines: list[dict],
                            start_utc: datetime,
                            end_utc: datetime,
                            shock_onset: datetime,
                            calm_end: datetime,
                            panic_end: datetime) -> pl.DataFrame:
    """Aggregate 1m Binance klines to hourly USDT/USDC basis + event_phase."""
    if not klines:
        return pl.DataFrame()

    df = pl.DataFrame(klines)
    df = df.with_columns([
        (pl.col("ts_ms") * 1000).cast(pl.Datetime("us", "UTC")).alias("dt"),
    ])
    df = df.with_columns([
        (pl.col("ts_ms") // (3_600_000)).alias("h"),
        # USDCUSDT close: price of USDC in USDT; > 1 means USDT at discount
        # basis_vs_usd here is (close - 1) * 10000 bps, positive = USDT stress
        (pl.col("close") - 1.0).alias("basis_vs_usd"),
    ])

    agg = df.group_by("h").agg([
        pl.col("close").mean().alias("mid_price"),
        pl.col("basis_vs_usd").mean().alias("basis_bps_raw"),
        pl.col("volume").sum().alias("volume_1h"),
        pl.col("n_trades").sum().alias("n_trades_1h"),
    ]).sort("h")

    # Scale basis to bps
    agg = agg.with_columns([
        (pl.col("basis_bps_raw") * 10_000).alias("basis_bps"),
    ])

    h_start = int(start_utc.timestamp()) // 3600
    h_end = int(end_utc.timestamp()) // 3600
    all_hours = pl.DataFrame({"h": list(range(h_start, h_end + 1))})
    agg = all_hours.join(agg, on="h", how="left").with_columns([
        pl.col("mid_price").fill_null(1.0),
        pl.col("basis_bps").fill_null(0.0),
        pl.col("volume_1h").fill_null(0.0),
        pl.col("n_trades_1h").fill_null(0),
    ])

    agg = agg.with_columns([
        ((pl.col("h") * 3_600 * 1_000_000).cast(pl.Datetime("us", "UTC"))).alias("wall_clock_utc"),
    ])

    shock_ts = int(shock_onset.timestamp())
    calm_ts = int(calm_end.timestamp())
    panic_ts = int(panic_end.timestamp())

    def phase(h_val: int) -> str:
        ts = h_val * 3600
        if ts < shock_ts:
            return "calm"
        elif ts <= panic_ts:
            return "panic"
        else:
            return "recovery"

    agg = agg.with_columns([
        pl.col("h").map_elements(phase, return_dtype=pl.String).alias("event_phase"),
    ])

    return agg.select(["wall_clock_utc", "mid_price", "basis_bps", "volume_1h", "event_phase"])


# ---------------------------------------------------------------------------
# Analysis: Forbes-Rigobon
# ---------------------------------------------------------------------------

def _pearson_r(x: np.ndarray, y: np.ndarray) -> float | None:
    from scipy import stats as _stats
    if len(x) < 6 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    r, _ = _stats.pearsonr(x, y)
    return float(r)


def _fisher_z_stat(r_calm: float | None, n_calm: int,
                   r_panic: float | None, n_panic: int) -> float | None:
    if r_calm is None or r_panic is None or min(n_calm, n_panic) < 4:
        return None
    r1c = max(min(r_calm, 0.999), -0.999)
    r2c = max(min(r_panic, 0.999), -0.999)
    z1, z2 = math.atanh(r1c), math.atanh(r2c)
    se = math.sqrt(1.0 / (n_calm - 3) + 1.0 / (n_panic - 3))
    return (z2 - z1) / se


def _block_bootstrap_fr(calm_x: np.ndarray, calm_y: np.ndarray,
                         panic_x: np.ndarray, panic_y: np.ndarray,
                         n_boot: int = 2000, block_hours: int = 24,
                         rng_seed: int = 42) -> dict:
    rng = np.random.default_rng(rng_seed)
    z_obs = _fisher_z_stat(_pearson_r(calm_x, calm_y), len(calm_x),
                            _pearson_r(panic_x, panic_y), len(panic_x))
    if z_obs is None:
        return {"z_fisher": None, "p_onesided": None, "ci_lo": None, "ci_hi": None,
                "pct_positive": None, "n_calm": len(calm_x), "n_panic": len(panic_x)}

    boot_z = []
    for _ in range(n_boot):
        def _resamp(x, y, block):
            n = len(x)
            n_blocks = math.ceil(n / block)
            starts = rng.integers(0, n, size=n_blocks)
            xs, ys = [], []
            for s in starts:
                idx = np.arange(s, s + block) % n
                xs.append(x[idx]); ys.append(y[idx])
            xr = np.concatenate(xs)[:n]
            yr = np.concatenate(ys)[:n]
            return xr, yr

        cx, cy = _resamp(calm_x, calm_y, block_hours)
        px, py = _resamp(panic_x, panic_y, block_hours)
        z = _fisher_z_stat(_pearson_r(cx, cy), len(cx),
                           _pearson_r(px, py), len(px))
        if z is not None:
            boot_z.append(z)

    boot_z_arr = np.array(boot_z)
    pct_pos = float((boot_z_arr > 0).mean())
    p_one_sided = float(1.0 - pct_pos)

    return {
        "z_fisher": round(z_obs, 3),
        "rho_calm": round(float(_pearson_r(calm_x, calm_y) or 0), 3),
        "rho_panic": round(float(_pearson_r(panic_x, panic_y) or 0), 3),
        "n_calm": len(calm_x),
        "n_panic": len(panic_x),
        "p_onesided": round(p_one_sided, 4),
        "pct_positive": round(pct_pos, 4),
        "ci_lo": round(float(np.percentile(boot_z_arr, 2.5)), 3),
        "ci_hi": round(float(np.percentile(boot_z_arr, 97.5)), 3),
    }


def _run_fr(pool_df: pl.DataFrame, market_df: pl.DataFrame) -> dict:
    """Run Forbes-Rigobon on pool flow (on-chain) and market basis."""
    results = {}

    for label, df, feat in [
        ("onchain", pool_df, "usdc_net_sold_1h"),
        ("market",  market_df, "basis_bps"),
    ]:
        if df is None or len(df) == 0 or feat not in df.columns:
            results[label] = {"z_fisher": None, "status": "no_data"}
            continue

        calm = df.filter(pl.col("event_phase") == "calm")[feat].drop_nulls().to_numpy()
        panic = df.filter(pl.col("event_phase") == "panic")[feat].drop_nulls().to_numpy()

        # For FR we need a paired cross-pool correlation — here we use
        # absolute_flow (on-chain) vs absolute_basis (market) as a proxy
        # single-feature self-correlation across regimes (|calm| vs |panic| variance)
        # This gives a regime-switching variance ratio rather than a cross-pool FR stat.
        # Note: the cross-pool FR stat requires TWO pools — for single-feature data we
        # compute the Fisher z comparing the autocorrelation (lag-1) calm vs panic.

        # AR(1) calm vs AR(1) panic as Forbes-Rigobon variant
        def _ar1(x):
            if len(x) < 3:
                return None, None
            return x[:-1], x[1:]

        xc, yc = _ar1(calm)
        xp, yp = _ar1(panic)
        if xc is None or xp is None:
            results[label] = {"z_fisher": None, "status": "insufficient_data",
                              "n_calm": len(calm), "n_panic": len(panic)}
            continue

        r = _block_bootstrap_fr(xc, yc, xp, yp)
        r["feature"] = feat
        r["status"] = "ok"
        results[label] = r

    return results


# ---------------------------------------------------------------------------
# Analysis: HMM stress detection
# ---------------------------------------------------------------------------

def _run_hmm(pool_df: pl.DataFrame, market_df: pl.DataFrame,
             event_id: str) -> dict:
    """Run causal (filtered-posterior) HMM on on-chain and market features."""
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn not available; skipping HMM analysis")
        return {"status": "hmmlearn_not_installed"}

    from sklearn.metrics import roc_auc_score

    results = {}

    for label, df, feats in [
        ("onchain", pool_df,
         ["usdc_net_sold_1h", "implied_pool_price", "reserve_imbalance"]),
        ("market",  market_df, ["mid_price", "basis_bps"]),
    ]:
        if df is None or len(df) == 0:
            results[label] = {"auroc": None, "status": "no_data"}
            continue

        missing = [f for f in feats if f not in df.columns]
        avail = [f for f in feats if f in df.columns]
        if not avail:
            results[label] = {"auroc": None, "status": f"missing_features:{missing}"}
            continue

        X_raw = df.select(avail).to_numpy().astype(float)
        y = (df["event_phase"].to_numpy() == "panic").astype(int)

        # Replace NaN with column mean
        X_raw = np.where(np.isnan(X_raw), np.nanmean(X_raw, axis=0), X_raw)
        if label == "onchain":
            # abs-value for pool features (direction-agnostic stress signal)
            X_raw = np.abs(X_raw)

        # Standardize
        mu, sd = X_raw.mean(0), X_raw.std(0) + 1e-9
        X = (X_raw - mu) / sd

        n_states = 3
        try:
            hmm = GaussianHMM(n_components=n_states, covariance_type="diag",
                              n_iter=200, random_state=0)
            hmm.fit(X)
        except Exception as e:
            results[label] = {"auroc": None, "status": f"hmm_fit_failed:{e}"}
            continue

        # Identify stress state: highest mean absolute deviation from peg
        state_means = hmm.means_  # (n_states, n_features)
        stress_state = int(np.argmax(state_means[:, 0]))  # highest flow/basis state

        # Causal (filtered) posterior
        try:
            log_prob, state_seq = hmm.decode(X, algorithm="viterbi")
            # Filtered posterior via forward algorithm
            _, fwd = hmm._do_forward_pass(hmm._compute_log_likelihood(X))
            log_fwd = fwd  # shape (T, n_states)
            # Normalize to probabilities
            log_sum = np.logaddexp.reduce(log_fwd, axis=1, keepdims=True)
            fwd_prob = np.exp(log_fwd - log_sum)
            stress_prob = fwd_prob[:, stress_state]
        except Exception:
            # Fallback: use viterbi state sequence
            stress_prob = (state_seq == stress_state).astype(float)

        if y.sum() == 0 or y.sum() == len(y):
            results[label] = {"auroc": None, "status": "degenerate_labels",
                              "n_panic": int(y.sum())}
            continue

        try:
            auroc = float(roc_auc_score(y, stress_prob))
        except Exception as e:
            results[label] = {"auroc": None, "status": f"auroc_failed:{e}"}
            continue

        # Detection latency (hours from onset to first alarm at 5% false-alarm)
        calm_probs = stress_prob[y == 0]
        threshold = np.percentile(calm_probs, 95) if len(calm_probs) > 0 else 0.5
        panic_hours = np.where(y == 1)[0]
        alarm_hours = np.where(stress_prob > threshold)[0]
        if len(panic_hours) > 0 and len(alarm_hours) > 0:
            first_panic = int(panic_hours[0])
            first_alarm = int(alarm_hours[0])
            lead_h = first_panic - first_alarm  # positive = early warning
        else:
            lead_h = None

        results[label] = {
            "auroc": round(auroc, 3),
            "stress_state": stress_state,
            "lead_hours": lead_h,
            "n_calm": int((y == 0).sum()),
            "n_panic": int(y.sum()),
            "status": "ok",
        }

    return results


# ---------------------------------------------------------------------------
# Build gold-schema parquet
# ---------------------------------------------------------------------------

_NaN = float("nan")

def _build_gold_parquet(pool_df: pl.DataFrame, market_df: pl.DataFrame,
                         event_id: str, episode: dict) -> pl.DataFrame:
    """Build a gold-schema contagion features parquet (minimal schema for analysis)."""
    rows = []

    if pool_df is not None and len(pool_df) > 0:
        for row in pool_df.iter_rows(named=True):
            rows.append({
                "event_id": event_id,
                "node_id": "curve_3pool",
                "layer": "dex_pool",
                "asset": "USDT",
                "venue": "curve",
                "tier_nominal": "A",
                "tier_actual": "A",
                "wall_clock_utc": row["wall_clock_utc"],
                "event_time_seconds": int(
                    row["wall_clock_utc"].replace(tzinfo=None).timestamp()
                    if hasattr(row["wall_clock_utc"], "replace")
                    else row["wall_clock_utc"]
                ),
                "usdc_net_sold_1h": float(row.get("usdc_net_sold_1h") or 0.0),
                "usdc_net_sold_cum": float(row.get("usdc_net_sold_cum") or 0.0),
                "n_events": int(row.get("n_events") or 0),
                "reserve_imbalance": float(row.get("reserve_imbalance") or 0.0),
                "implied_pool_price": float(row.get("implied_pool_price") or 1.0),
                "event_phase": row.get("event_phase", "calm"),
                "mid_price": _NaN,
                "basis_bps": _NaN,
            })

    if market_df is not None and len(market_df) > 0:
        for row in market_df.iter_rows(named=True):
            rows.append({
                "event_id": event_id,
                "node_id": "usdt_binance",
                "layer": "cex_market",
                "asset": "USDT",
                "venue": "binance",
                "tier_nominal": "B",
                "tier_actual": "B",
                "wall_clock_utc": row["wall_clock_utc"],
                "event_time_seconds": 0,
                "usdc_net_sold_1h": _NaN,
                "usdc_net_sold_cum": _NaN,
                "n_events": 0,
                "reserve_imbalance": _NaN,
                "implied_pool_price": _NaN,
                "event_phase": row.get("event_phase", "calm"),
                "mid_price": float(row.get("mid_price") or 1.0),
                "basis_bps": float(row.get("basis_bps") or 0.0),
            })

    if not rows:
        return pl.DataFrame()

    return pl.from_dicts(rows)


# ---------------------------------------------------------------------------
# Main per-episode pipeline
# ---------------------------------------------------------------------------

def _ts_from_iso(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _dt_from_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=timezone.utc)


def run_episode(event_id: str, episode: dict, dry_run: bool = False,
                out_dir: Path | None = None, data_dir: Path | None = None) -> dict:
    logger.info("=== Episode: %s (%s) ===", event_id, episode["name"])

    start_dt = _dt_from_iso(episode["analysis_window"][0] + "T00:00:00Z")
    end_dt   = _dt_from_iso(episode["analysis_window"][1] + "T23:59:59Z")
    shock_dt = _dt_from_iso(episode["shock_onset_utc"])
    calm_end_dt = _dt_from_iso(episode["calm_end_utc"])
    panic_end_dt = _dt_from_iso(episode["panic_end_utc"])

    start_ts = int(start_dt.timestamp())
    end_ts   = int(end_dt.timestamp())
    start_ms = start_ts * 1000
    end_ms   = end_ts * 1000

    pool_df = market_df = None

    if not dry_run:
        # --- On-chain: Curve 3pool TokenExchange events ---
        logger.info("Fetching block numbers for %s → %s", start_dt.date(), end_dt.date())
        from_block = _block_by_ts(start_ts, "after")
        to_block   = _block_by_ts(end_ts,   "before")
        logger.info("Block range: %d → %d", from_block, to_block)

        if from_block > 0 and to_block > from_block:
            logger.info("Fetching Curve 3pool TokenExchange events ...")
            logs = _fetch_all_curve_logs(from_block, to_block)
            logger.info("  Fetched %d raw logs", len(logs))
            pool_df = _build_pool_features(
                logs, start_dt, end_dt, shock_dt, calm_end_dt, panic_end_dt
            )
            logger.info("  Pool features: %d hourly rows", len(pool_df) if pool_df is not None else 0)
        else:
            logger.warning("Could not resolve block numbers; skipping on-chain fetch")

        # --- Market: Binance USDT/USDC kline ---
        symbol = episode.get("cex_symbol", "USDTUSDC")
        logger.info("Fetching Binance %s 1m klines ...", symbol)
        klines = _fetch_binance_klines(symbol, start_ms, end_ms)
        logger.info("  Fetched %d 1m klines", len(klines))
        if klines:
            market_df = _build_market_features(
                klines, start_dt, end_dt, shock_dt, calm_end_dt, panic_end_dt
            )
        else:
            logger.warning("No Binance klines for %s; trying USDCUSDT reverse", symbol)
            klines2 = _fetch_binance_klines("USDCUSDT", start_ms, end_ms)
            if klines2:
                # Flip: USDC/USDT → USDT/USDC price = 1 / close
                for k in klines2:
                    if k["close"] > 0:
                        k["close"] = 1.0 / k["close"]
                market_df = _build_market_features(
                    klines2, start_dt, end_dt, shock_dt, calm_end_dt, panic_end_dt
                )
                logger.info("  Used USDCUSDT reverse (%d klines)", len(klines2))

    else:
        logger.info("[DRY RUN] Skipping data fetch")

    # --- Save gold parquet ---
    gold_df = _build_gold_parquet(pool_df, market_df, event_id, episode)
    if data_dir and len(gold_df) > 0:
        gold_path = data_dir / f"dataset_contagion_features_{event_id}.parquet"
        gold_df.write_parquet(str(gold_path))
        logger.info("Saved gold parquet: %s (%d rows)", gold_path.name, len(gold_df))

    # --- Run analyses ---
    fr_result  = _run_fr(pool_df, market_df)
    hmm_result = _run_hmm(pool_df, market_df, event_id)

    result = {
        "event_id": event_id,
        "name": episode["name"],
        "mechanism": episode["mechanism"],
        "analysis_window": episode["analysis_window"],
        "shock_onset_utc": episode["shock_onset_utc"],
        "expected_onchain_fires": episode["expected_onchain_fires"],
        "n_pool_rows": len(pool_df) if pool_df is not None else 0,
        "n_market_rows": len(market_df) if market_df is not None else 0,
        "forbes_rigobon": fr_result,
        "hmm": hmm_result,
    }

    # Convenience summary for paper
    onchain_auroc = hmm_result.get("onchain", {}).get("auroc")
    market_auroc  = hmm_result.get("market", {}).get("auroc")
    lead_h        = hmm_result.get("onchain", {}).get("lead_hours")
    detects       = "on-chain" if (
        onchain_auroc and market_auroc and onchain_auroc > market_auroc + 0.05
    ) else "market" if (
        market_auroc and onchain_auroc and market_auroc > onchain_auroc + 0.05
    ) else "either (tie)" if (onchain_auroc and market_auroc) else "insufficient_data"

    result["paper_summary"] = {
        "hmm_auroc_onchain": onchain_auroc,
        "hmm_auroc_market":  market_auroc,
        "lead_hours": lead_h,
        "detects": detects,
        "fr_z_fisher": fr_result.get("onchain", {}).get("z_fisher"),
        "fr_p_onesided": fr_result.get("onchain", {}).get("p_onesided"),
    }

    logger.info("Summary: on-chain AUROC=%s  market AUROC=%s  detects=%s  lead=%sh",
                onchain_auroc, market_auroc, detects, lead_h)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", nargs="+", default=list(_EPISODES.keys()),
                    help="Episode IDs to run (default: all)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Skip API fetches; run analysis on empty data (tests pipeline)")
    ap.add_argument("--n-bootstrap", type=int, default=2000)
    args = ap.parse_args()

    repo_root = Path(__file__).parent.parent
    out_dir  = repo_root / "results" / "tables"
    data_dir = repo_root / "data" / "gold"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for event_id in args.episodes:
        ep = _EPISODES.get(event_id)
        if ep is None:
            logger.error("Unknown episode '%s'. Available: %s", event_id, list(_EPISODES))
            continue
        try:
            r = run_episode(event_id, ep, dry_run=args.dry_run,
                            out_dir=out_dir, data_dir=data_dir)
            all_results[event_id] = r
        except Exception as exc:
            logger.error("Episode %s failed: %s", event_id, exc, exc_info=True)
            all_results[event_id] = {"event_id": event_id, "error": str(exc)}

    # Write results
    out_json = out_dir / "table_2024_episodes_detection.json"
    out_json.write_text(json.dumps(all_results, indent=2, default=str))
    logger.info("Wrote: %s", out_json)

    # Print paper-ready table rows
    print("\n=== PAPER SUMMARY (for tab:detection) ===")
    for eid, r in all_results.items():
        ps = r.get("paper_summary", {})
        print(f"  {r.get('name', eid)}")
        print(f"    Mechanism:    {r.get('mechanism')}")
        print(f"    On-chain AUROC: {ps.get('hmm_auroc_onchain')}")
        print(f"    Market AUROC:   {ps.get('hmm_auroc_market')}")
        print(f"    Lead hours:     {ps.get('lead_hours')}")
        print(f"    Detects:        {ps.get('detects')}")
        print(f"    FR z:           {ps.get('fr_z_fisher')}  p={ps.get('fr_p_onesided')}")
        exp = r.get("expected_onchain_fires")
        conf = (
            "CONFIRMED" if (
                ps.get("hmm_auroc_onchain") is not None and
                (
                    (exp and ps["hmm_auroc_onchain"] > 0.7) or
                    (not exp and ps.get("hmm_auroc_onchain", 1.0) < 0.65)
                )
            ) else "CHECK_RESULTS"
        )
        print(f"    Hypothesis:     {'on-chain fires' if exp else 'on-chain silent'} [{conf}]")
        print()


if __name__ == "__main__":
    main()
