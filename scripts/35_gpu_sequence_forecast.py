"""F4: GPU sequence-model forecast grid.  Does temporal context (a GRU) plus
Tier-A on-chain features beat a Tier-B market-only baseline at FORECASTING
near-future CEX stress?  This attacks the earlier logistic-forecast null with a
stronger model class and multiple horizons.

Honest protocol: strictly temporal 70/30 split (test is the future of train),
forward label = |basis| exceeds 10 bps within the next H minutes, on-chain pool
state asof-joined BACKWARD (no leakage).  Grid: horizon x {market, market+onchain}.
Runs on CUDA if available, else CPU.  Degenerate splits are reported, not hidden.

Usage:  python scripts/35_gpu_sequence_forecast.py
"""
from __future__ import annotations

import csv
import warnings

import numpy as np
import polars as pl

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

EVENTS = ["usdt_curve_2023", "terra_luna_2022", "ftx_2022", "busd_2023", "usdc_svb_2023"]
CEX_NODE = {
    "usdt_curve_2023": "usdt_binance", "terra_luna_2022": "usdt_binance",
    "ftx_2022": "usdt_binance", "busd_2023": "busd_binance",
    "usdc_svb_2023": "usdc_coinbase",
}
ONCHAIN_NODE = "curve_3pool"
LOOKBACK = 30
THRESH_BPS = 10.0
HORIZONS = [15, 30, 60]
MARKET = ["abs_basis", "spread_bps", "orderbook_imbalance", "log_depth",
          "abs_basis_ma15", "abs_basis_chg"]
ONCHAIN = ["oc_pxdev", "oc_imb", "oc_slip", "oc_flow"]


def _frame(ev, horizon):
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    cex = df.filter(pl.col("node_id") == CEX_NODE[ev]).sort("wall_clock_utc").with_columns([
        pl.col("basis_bps").abs().alias("abs_basis"),
        (pl.col("depth_10bps_bid_usd") + pl.col("depth_10bps_ask_usd")).log1p().alias("log_depth"),
    ])
    cex = cex.with_columns([
        pl.col("abs_basis").rolling_mean(15, min_periods=1).alias("abs_basis_ma15"),
        pl.col("abs_basis").diff().fill_null(0).alias("abs_basis_chg"),
    ])
    rev = cex.select("abs_basis").reverse()
    fwd = rev.select(pl.col("abs_basis").rolling_max(horizon, min_periods=1)).reverse().to_series()
    cex = cex.with_columns((fwd.shift(-1) > THRESH_BPS).cast(pl.Int8).alias("y"))
    oc = (df.filter(pl.col("node_id") == ONCHAIN_NODE).sort("wall_clock_utc").with_columns([
        (pl.col("implied_pool_price") - 1.0).abs().alias("oc_pxdev"),
        pl.col("reserve_imbalance").abs().alias("oc_imb"),
        pl.col("pool_slippage_10k").alias("oc_slip"),
        pl.col("usdc_net_sold_1h").abs().alias("oc_flow"),
    ]).select(["wall_clock_utc", *ONCHAIN]))
    m = cex.join_asof(oc, on="wall_clock_utc", strategy="backward").sort("wall_clock_utc")
    if m.height > horizon:
        m = m[: m.height - horizon]
    return m


def _seq(M, y, lookback):
    X = np.stack([M[i - lookback:i] for i in range(lookback, len(M))])
    return X.astype(np.float32), y[lookback:].astype(np.float32)


def _run_gru(Xtr, ytr, Xte, yte, device, epochs=12):
    import torch
    import torch.nn as nn
    g = torch.Generator().manual_seed(0)

    class GRU(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.gru = nn.GRU(d, 32, batch_first=True)
            self.fc = nn.Sequential(nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        def forward(self, x):
            _, h = self.gru(x)
            return self.fc(h[-1]).squeeze(-1)

    torch.manual_seed(0)
    net = GRU(Xtr.shape[-1]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    pos = float(ytr.sum()); neg = float(len(ytr) - ytr.sum())
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
    Xte_t = torch.tensor(Xte, device=device)
    n = len(Xtr_t); bs = 512
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(device)
        net.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            out = net(Xtr_t[idx])
            loss = lossf(out, ytr_t[idx])
            loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xte_t)).cpu().numpy()
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(yte, p))


def main():
    import torch
    from sklearn.preprocessing import StandardScaler
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("device = %s", device)
    rows = []
    for ev in EVENTS:
        for H in HORIZONS:
            m = _frame(ev, H)
            y = m["y"].to_numpy()
            n = len(y); cut = int(n * 0.70)
            if len(np.unique(y[:cut])) < 2 or len(np.unique(y[cut:])) < 2:
                logger.warning("%-16s H=%2d degenerate split, skip", ev, H)
                rows.append({"event": ev, "horizon": H, "auroc_market": None,
                             "auroc_both": None, "lift": None, "note": "degenerate"})
                continue
            for fs_name, cols in [("market", MARKET), ("both", MARKET + ONCHAIN)]:
                Mf = np.nan_to_num(m.select(cols).to_numpy().astype(float))
                sc = StandardScaler().fit(Mf[:cut])
                Mf = sc.transform(Mf)
                X, yy = _seq(Mf, y, LOOKBACK)
                c2 = cut - LOOKBACK
                au = _run_gru(X[:c2], yy[:c2], X[c2:], yy[c2:], device)
                if fs_name == "market":
                    au_m = au
                else:
                    rows.append({"event": ev, "horizon": H, "auroc_market": round(au_m, 4),
                                 "auroc_both": round(au, 4), "lift": round(au - au_m, 4), "note": ""})
                    logger.info("F4 %-16s H=%2d market=%.3f both=%.3f lift=%+.3f",
                                ev, H, au_m, au, au - au_m)
    TDIR = results_root() / "tables"; TDIR.mkdir(parents=True, exist_ok=True)
    with (TDIR / "grid_f4_gru_forecast.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    logger.info("wrote grid_f4_gru_forecast.csv (%d rows)", len(rows))


if __name__ == "__main__":
    main()
