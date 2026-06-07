"""Publication figures for the five core findings of the paper.

Generates, from the committed result tables and gold panels:
  fig_regime_switch.pdf     - regime-switching contagion (calm vs panic coupling)
  fig_arbitrage_flip.pdf    - stabilizing->amplifying arbitrage sign flip
  fig_price_discovery.pdf   - on-chain vs CEX price discovery + deviation ratio
  fig_hmm_detection.pdf     - unsupervised HMM stress posterior over time
  fig_ml_diagnosis.pdf      - cross-event transfer matrix (concept shift)

Outputs to results/paper/figures/ as both .pdf (paper) and .png (preview).

Usage:
    python scripts/29_make_finding_figures.py
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stressnet.config import gold_root, results_root
from stressnet.utils.logging import get_logger

warnings.filterwarnings("ignore")
logger = get_logger(__name__)

FIGDIR = results_root() / "paper" / "figures"
TABLES = results_root() / "tables"

# Consistent house style
plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "figure.dpi": 150, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
})
C_CALM, C_PANIC, C_POS, C_NEG = "#5B8DEF", "#E0533D", "#27AE60", "#C0392B"
SHORT = {"usdt_curve_2023": "USDT/\nCurve", "terra_luna_2022": "Terra/\nLUNA",
         "ftx_2022": "FTX", "busd_2023": "BUSD", "usdc_svb_2023": "USDC/\nSVB"}


def _save(fig, name):
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png")
    plt.close(fig)
    logger.info("wrote %s.pdf/.png", name)


def fig_regime_switch():
    df = pl.read_csv(TABLES / "table_regime_contagion.csv")
    ev = df["event_id"].to_list()
    pre = df["rho_pre"].to_list(); pan = df["rho_panic"].to_list()
    x = np.arange(len(ev)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w/2, pre, w, label="calm (pre)", color=C_CALM)
    ax.bar(x + w/2, pan, w, label="acute (panic)", color=C_PANIC)
    for i, r in enumerate(df.iter_rows(named=True)):
        if r["contagion_regime_shift"]:
            ax.annotate("*", (x[i] + w/2, (r["rho_panic"] or 0) + 0.03),
                        ha="center", fontsize=14, color="black")
    ax.axhline(0, color="#888", lw=0.7)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[e] for e in ev])
    ax.set_ylabel(r"cross-pool flow correlation $\hat{\rho}$")
    ax.set_title("Regime-switching contagion: coupling activates in panic\n"
                 "(* = significant calm$\\to$panic shift, Fisher $z$, $p<0.05$)")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_regime_switch")


def fig_arbitrage_flip():
    df = pl.read_csv(TABLES / "table_arbitrage_regime.csv")
    ev = df["event_id"].to_list()
    rc = [x if x is not None else np.nan for x in df["r_calm"].to_list()]
    rp = [x if x is not None else np.nan for x in df["r_panic"].to_list()]
    x = np.arange(len(ev)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    ax.bar(x - w/2, rc, w, label="calm", color=C_CALM)
    ax.bar(x + w/2, rp, w, label="panic",
           color=[C_POS if (v or 0) > 0 else C_NEG for v in rp])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[e] for e in ev])
    ax.set_ylabel("flow$\\,\\leftrightarrow\\,$price coupling")
    ax.set_title("Arbitrage flips: stabilizing ($-$) in calm,\n"
                 "amplifying ($+$) in panic for supply/regulatory shocks")
    # annotate the panic bar color meaning
    ax.bar(np.nan, np.nan, color=C_POS, label="panic $+$ (amplifying)")
    ax.bar(np.nan, np.nan, color=C_NEG, label="panic $-$ (stabilizing)")
    ax.legend(frameon=False, fontsize=7, loc="lower left", ncol=2)
    _save(fig, "fig_arbitrage_flip")


def fig_price_discovery():
    df = pl.read_csv(TABLES / "table_price_discovery.csv")
    ev = df["event_id"].to_list()
    onc = df["onchain_leads_mean_r"].to_list(); cex = df["cex_leads_mean_r"].to_list()
    ratio = df["onchain_dev_ratio"].to_list()
    x = np.arange(len(ev)); w = 0.38
    fig, ax = plt.subplots(figsize=(5.8, 3.1))
    ax.bar(x - w/2, onc, w, label="on-chain leads", color=C_POS)
    ax.bar(x + w/2, cex, w, label="CEX leads", color=C_PANIC)
    ax.set_xticks(x); ax.set_xticklabels([SHORT[e] for e in ev])
    ax.set_ylabel("mean lead-lag correlation")
    ax.set_title("On-chain price discovery: the pool leads the exchange\n"
                 "for DeFi-native stress (annotated: pool/CEX deviation ratio)")
    for i, rr in enumerate(ratio):
        if rr is not None and not (isinstance(rr, float) and np.isnan(rr)):
            top = max(onc[i], cex[i])
            ax.annotate(f"{rr:.0f}$\\times$", (x[i], top + 0.015),
                        ha="center", fontsize=8, color="#333")
    ax.legend(frameon=False, loc="upper right")
    _save(fig, "fig_price_discovery")


def fig_hmm_detection():
    # HMM posterior over time for usdt_curve, with the true panic window shaded
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logger.warning("hmmlearn missing; skipping HMM figure.")
        return
    ev = "usdt_curve_2023"
    df = pl.read_parquet(gold_root() / f"dataset_contagion_features_{ev}.parquet")
    d = (df.filter(pl.col("node_id") == "curve_3pool")
           .with_columns((pl.col("event_time_seconds") // 3600).alias("h"))
           .group_by("h").agg(
               pl.col("usdc_net_sold_1h").sum().alias("flow"),
               pl.col("implied_pool_price").mean().alias("px"),
               pl.col("reserve_imbalance").mean().alias("imb"),
               pl.col("event_phase").first().alias("phase")).sort("h"))
    h = d["h"].to_numpy()
    af = np.abs(np.nan_to_num(d["flow"].to_numpy()))
    pxd = np.abs(np.nan_to_num(d["px"].to_numpy(), nan=1.0) - 1.0)
    imb = np.abs(np.nan_to_num(d["imb"].to_numpy()))
    X = np.column_stack([af, pxd, imb]); X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = (d["phase"].to_numpy() == "panic").astype(int)
    m = GaussianHMM(n_components=3, covariance_type="diag", n_iter=300, random_state=0).fit(X)
    s = int(np.argmax(m.means_[:, 1]))
    post = m.predict_proba(X)[:, s]

    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    # shade true panic hours
    inpanic = False
    for i in range(len(h)):
        if y[i] and not inpanic:
            start = h[i]; inpanic = True
        if inpanic and (i == len(h) - 1 or not y[i]):
            ax.axvspan(start, h[i], color=C_PANIC, alpha=0.12)
            inpanic = False
    ax.plot(h, post, color="#222", lw=1.3, label="HMM P(stress state)")
    ax.axhline(0.5, ls="--", color="#999", lw=0.7)
    ax.set_xlabel("hours relative to shock onset")
    ax.set_ylabel("P(stress | on-chain state)")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Unsupervised HMM recovers the stress regime (USDT/Curve, AUROC 0.93)\n"
                 "shaded = true panic window; model fit with no labels")
    ax.legend(frameon=False, loc="center left")
    _save(fig, "fig_hmm_detection")


def fig_tvpvar():
    # Per-window FEVD share for crvUSD/USDT -> 3pool, showing post-onset concentration
    p = TABLES / "table_tvp_var_spillovers_usdt_curve_2023.csv"
    if not p.exists():
        logger.warning("TVP-VAR spillovers table missing; skipping.")
        return
    df = pl.read_csv(p)
    sub = df.filter((pl.col("caused_node") == "curve_3pool") &
                    (pl.col("causing_node") == "curve_crvusd_usdt")).sort("window_center")
    if sub.height == 0:
        logger.warning("no crvUSD->3pool rows; skipping TVP-VAR fig.")
        return
    x = (sub["window_center"].to_numpy() / 3600.0)   # seconds -> hours rel. onset
    yv = sub["fevd_share"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    ax.axvline(0, color=C_PANIC, ls="--", lw=0.9, label="shock onset")
    ax.plot(x, yv, "-o", color="#222", lw=1.4, ms=4)
    ax.set_xlabel("rolling-window centre (hours rel. onset)")
    ax.set_ylabel("FEVD share\n(crvUSD/USDT $\\to$ 3pool)")
    ax.set_ylim(-0.03, 1.0)
    ax.set_title("TVP-VAR: cross-pool spillover is transient and post-onset\n"
                 "(94\\% concentrated $\\approx$+140h after onset, not before)")
    ax.legend(frameon=False, loc="upper left")
    _save(fig, "fig_tvpvar")


def fig_ml_diagnosis():
    df = pl.read_csv(TABLES / "table_transfer_matrix.csv")
    ev = [c for c in df.columns if c != "train_event"]
    M = np.array([[df.filter(pl.col("train_event") == tr)[te][0] for te in ev] for tr in ev], float)
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    im = ax.imshow(M, cmap="RdYlGn", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(ev))); ax.set_yticks(range(len(ev)))
    ax.set_xticklabels([SHORT[e].replace("\n", "") for e in ev], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([SHORT[e].replace("\n", "") for e in ev], fontsize=7)
    for i in range(len(ev)):
        for j in range(len(ev)):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    ax.set_xlabel("test event"); ax.set_ylabel("train event")
    ax.set_title("Cross-event transfer AUROC\n"
                 "diagonal (within) high; off-diagonal $\\approx$ chance (concept shift)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="AUROC")
    _save(fig, "fig_ml_diagnosis")


def main():
    for fn in (fig_regime_switch, fig_arbitrage_flip, fig_price_discovery,
               fig_hmm_detection, fig_ml_diagnosis, fig_tvpvar):
        try:
            fn()
        except Exception as exc:
            logger.error("%s failed: %s", fn.__name__, exc, exc_info=True)
    logger.info("Finding figures written to %s", FIGDIR)


if __name__ == "__main__":
    main()
